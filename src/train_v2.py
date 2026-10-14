"""EmbeddingGemma-2 Arabic ASR, reader v2.

Frozen audio encoder run live (no feature cache), learned softmax mix over all hidden layers, 6-layer transformer
reader, CTC over Arabic characters. Labels: Cohere and a second commercial recogniser's transcripts as two samples per window (two teachers,
different errors). Model selection on held-out calls; final score on the 72 human-corrected windows, scored exactly
like the reference Cohere decoder (mean per-window WER/CER after norm()).
"""
import json, math, os, random, re, sys, time, unicodedata
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoModel, AutoProcessor

OUT = os.environ.get("OUT", "runs/v2")
SEL = [6, 8, 10, 11, 12]          # probe: upper layers carry the speech content (12 best, then 11, 10)
EPOCHS = int(os.environ.get("EPOCHS", 20))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 6000))     # per batch, encoder frames (25 fps)
os.makedirs(OUT, exist_ok=True)
random.seed(0); torch.manual_seed(0)


def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("ـ", "")
    for d in range(0x064B, 0x0653): s = s.replace(chr(d), "")
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
         .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي").replace("ء", ""))
    s = re.sub(r"(.)\1{2,}", r"\1\1", s); s = re.sub(r"[^؀-ۿ0-9\s]", " ", s)
    return " ".join(s.split())


def _lev(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev = cur
    return d[len(h)]


def wer(r, h): a, b = norm(r).split(), norm(h).split(); return 0.0 if not a else _lev(a, b) / len(a)
def cer(r, h): a, b = list(norm(r).replace(" ", "")), list(norm(h).replace(" ", "")); return 0.0 if not a else _lev(a, b) / len(a)


# ---------------- data ----------------
call = lambda p: os.path.basename(p).split("_w")[0]
gold = json.load(open("data/calls/test_windows.json"))
gold_calls = {call(g["audio_filepath"]) for g in gold}
coh = {r["audio_filepath"]: r for r in json.load(open("data/calls/train_windows_cohere.json"))}
asr2 = {r["audio_filepath"]: r for r in json.load(open("data/calls/train_windows_asr2.json"))}
wins = [p for p in coh if call(p) not in gold_calls]
calls = sorted({call(p) for p in wins}); random.shuffle(calls)
val_calls = set(calls[:max(20, len(calls) // 20)])
train, val = [], []
for p in wins:
    labels = [t for t in (coh[p].get("cohere"), asr2.get(p, {}).get("asr2")) if t and norm(t)]
    if not labels: continue
    item = {"audio": p, "dur": coh[p]["duration"]}
    if call(p) in val_calls:
        val.append({**item, "text": labels[0], "alt": labels[-1]})
    else:
        train.append({**item, "labels": labels})                 # one encoder pass per window; teacher picked per epoch
chars = sorted({c for r in train for t in r["labels"] for c in norm(t)})
vocab = ["<b>"] + chars; c2i = {c: i for i, c in enumerate(vocab)}
enc = lambda t: [c2i[c] for c in norm(t) if c in c2i]
print(f"train windows {len(train)} ({sum(len(r['labels']) for r in train)} labels) | val windows {len(val)} "
      f"({len(val_calls)} calls) | gold {len(gold)} | vocab {len(vocab)}", flush=True)

# ---------------- frozen encoder ----------------
mid = "google/embeddinggemma-2"
fe = AutoProcessor.from_pretrained(mid).feature_extractor
full = AutoModel.from_pretrained(mid, dtype=torch.bfloat16)
encoder = full.audio_tower.cuda().eval(); del full; torch.cuda.empty_cache()
for p in encoder.parameters(): p.requires_grad_(False)

_wav_cache = {}
def load_wav(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    if sr != 16000:
        import librosa; w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return w


@torch.no_grad()
def encode(wavs):
    f = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    o = encoder(f["input_features"].cuda().to(torch.bfloat16), f["input_features_mask"].cuda(), output_hidden_states=True)
    hs = [o.hidden_states[i] for i in SEL] + [o.last_hidden_state]   # layers differ in width (1024 / 1536)
    # frame lengths after subsampling: count valid output frames from the mask the encoder returns
    m = getattr(o, "attention_mask", None)
    if m is None:
        lens = torch.full((hs.shape[1],), hs.shape[2], dtype=torch.long)
    else:
        lens = m.long().sum(-1).cpu()
    return hs, lens


# ---------------- reader ----------------
class Reader(nn.Module):
    def __init__(self, dims, d=512, n=6, v=len(vocab)):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(len(dims)))
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d)) for k in dims])
        self.drop = nn.Dropout(0.1)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, v))

    def forward(self, hs, lens, augment=False):
        w = torch.softmax(self.mix, 0)
        x = sum(w[i] * self.proj[i](h.float()) for i, h in enumerate(hs))
        if augment:                                               # SpecAugment-style masks on the mixed features
            B, T, D = x.shape; x = x.clone()
            for b in range(B):
                for _ in range(2):
                    t = random.randint(0, max(1, T // 20)); s0 = random.randint(0, max(0, T - t)); x[b, s0:s0 + t] = 0
        x = self.drop(x)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        T = x.shape[1]; pad = torch.arange(T, device=x.device)[None, :] >= lens.to(x.device)[:, None]
        x = self.tf(x, src_key_padding_mask=pad)
        return self.out(x)


probe_hs, probe_len = encode([load_wav(train[0]["audio"]), load_wav(train[-1]["audio"])])  # length check
print("batched length check (frames):", probe_len.tolist(), "vs duration*25:", [int(train[0]["dur"] * 25), int(train[-1]["dur"] * 25)], flush=True)
reader = Reader([h.shape[-1] for h in probe_hs]).cuda()
print("layers mixed:", SEL + ["out"], "dims", [h.shape[-1] for h in probe_hs],
      "| reader params (M):", round(sum(p.numel() for p in reader.parameters()) / 1e6, 1), flush=True)
opt = torch.optim.AdamW(reader.parameters(), lr=5e-4, weight_decay=1e-2)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)


def batches(items, max_frames=MAX_FRAMES):
    items = sorted(items, key=lambda r: r["dur"]); out, cur, mx = [], [], 0
    for r in items:
        fr = int(r["dur"] * 25) + 5; mx2 = max(mx, fr)
        if cur and mx2 * (len(cur) + 1) > max_frames: out.append(cur); cur, mx2 = [], fr
        cur.append(r); mx = mx2
    if cur: out.append(cur)
    random.shuffle(out); return out


def greedy(logits, lens):
    outs = []
    for b in range(logits.shape[0]):
        ids = logits[b, :int(lens[b])].argmax(-1).tolist(); s, prev = [], 0
        for t in ids:
            if t != prev and t != 0: s.append(vocab[t])
            prev = t
        outs.append("".join(s))
    return outs


@torch.no_grad()
def transcribe(paths, bs=4):
    reader.eval(); hyps = []
    for i in range(0, len(paths), bs):
        hs, lens = encode([load_wav(p) for p in paths[i:i + bs]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = reader(hs, lens)
        hyps += greedy(lg.float(), lens); del hs
    reader.train(); return hyps


steps_per_epoch = len(batches(train)); total = EPOCHS * steps_per_epoch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=total, pct_start=0.08)
best, t0 = (1e9, 0), time.time()
for ep in range(1, EPOCHS + 1):
    tot = n = 0
    for b in batches(train):
        hs, lens = encode([load_wav(r["audio"]) for r in b])
        ys = [torch.tensor(enc(random.choice(r["labels"]))) for r in b]
        keep = [k for k, y in enumerate(ys) if 0 < len(y) <= int(lens[k])]
        if not keep: continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = reader([h[keep] for h in hs], lens[keep], augment=True)
        lp = logits.float().log_softmax(-1).transpose(0, 1)
        del hs
        loss = ctc(lp, torch.cat([ys[k] for k in keep]), lens[keep], torch.tensor([len(ys[k]) for k in keep]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(reader.parameters(), 5.0); opt.step(); sched.step()
        tot += loss.item(); n += 1
    vh = transcribe([r["audio"] for r in val])
    vc = 100 * np.mean([min(cer(r["text"], h), cer(r["alt"], h)) for r, h in zip(val, vh)])
    vw = 100 * np.mean([min(wer(r["text"], h), wer(r["alt"], h)) for r, h in zip(val, vh)])
    mix = torch.softmax(reader.mix, 0).detach().cpu().numpy()
    print(f"ep {ep:2d} loss {tot / max(n, 1):.3f}  val CER {vc:5.2f}% WER {vw:5.2f}%  top layers "
          f"{[(SEL + ['out'])[i] for i in np.argsort(-mix)[:3]]}  ({time.time() - t0:.0f}s)", flush=True)
    if vc < best[0]:
        best = (vc, ep); torch.save({"reader": reader.state_dict(), "vocab": vocab}, f"{OUT}/reader_best.pt")

ck = torch.load(f"{OUT}/reader_best.pt"); reader.load_state_dict(ck["reader"])
gh = transcribe([g["audio_filepath"] for g in gold])
W = 100 * np.mean([wer(g["corrected"], h) for g, h in zip(gold, gh)])
C = 100 * np.mean([cer(g["corrected"], h) for g, h in zip(gold, gh)])
print(f"\nGOLD 72 windows (best epoch {best[1]}, val CER {best[0]:.2f}%):  WER {W:.2f}%  CER {C:.2f}%   "
      f"[Cohere: WER 40.48%  CER 20.46%]", flush=True)
print("layer mix:", np.round(torch.softmax(reader.mix, 0).detach().cpu().numpy(), 3).tolist(), flush=True)
json.dump([{"audio": g["audio_filepath"], "ref": g["corrected"], "hyp": h} for g, h in zip(gold, gh)],
          open(f"{OUT}/gold_hyp.json", "w"), ensure_ascii=False, indent=1)
print("TRAIN_DONE", flush=True)
