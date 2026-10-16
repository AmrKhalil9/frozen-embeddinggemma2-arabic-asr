"""EmbeddingGemma-2 Arabic ASR, reader v5c = staged hybrid CTC/attention + self-conditioned CTC (research only).
Gemini Experiment-1 recipe: 1024-d layers [4,6,8,9,10,11] (no layer 12 / output), SC-CTC at reader layers 2 and 4,
STAGE=dec (decoder only, everything else frozen) then STAGE=joint (0.3 CTC + 0.5 att + 0.2 inter-CTC), EOS-aware smoothing,
per-window weights from Cohere/second-recogniser disagreement, SADA inaudible rows removed.

Reader: layer mix -> proj 512 -> conv (stride SUB, default 1 = keep 25 fps) -> gated sinusoidal positions -> 6-layer transformer
-> CTC head  +  4-layer autoregressive transformer decoder over BPE pieces. Loss = CTC_W*CTC + (1-CTC_W)*attention (LS 0.1).
Original v4 notes:

Adds three switches on top of v3 (same data, same scoring):
  UNIT=char|bpe     CTC over characters or SentencePiece BPE pieces (VOCAB=512/1024)
  LORA=0|1          LoRA (r=16) on q/k/v/post of encoder layers LORA_LAYERS, gradient checkpointing
  FILTER=0|1        drop pseudo-labels whose speaking rate is implausible (chars/s outside [5, 24])
Layers mixed: SEL (+ encoder output). Telephony degradation of SADA as in v3 MODE=tel.
"""
import json, os, random, re, sys, time, unicodedata
import numpy as np, soundfile as sf, torch, torch.nn as nn
from transformers import AutoModel, AutoProcessor

OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
SUB = int(os.environ.get("SUB", 1)); UNIT = "bpe"; CTC_W = float(os.environ.get("CTC_W", 0.5)); WARM = os.environ.get("WARM", ""); SP_MODEL = os.environ.get("SP_MODEL", ""); VOCAB = int(os.environ.get("VOCAB", 1024))
LORA = os.environ.get("LORA", "0") == "1"; LORA_LAYERS = [int(x) for x in os.environ.get("LORA_LAYERS", "6,7,8,9,10").split(",")]
LORA_R = int(os.environ.get("LORA_R", 16)); LORA_LR = float(os.environ.get("LORA_LR", 1e-4))
FILTER = os.environ.get("FILTER", "1") == "1"
SEL = [int(x) for x in os.environ.get("SEL", "4,6,8,9,10,11").split(",")]
USE_OUT = os.environ.get("USE_OUT", "0") == "1"; STAGE = os.environ.get("STAGE", "joint"); INTER = [2, 4]
W_CTC, W_ATT, W_INT = 0.3, 0.5, 0.2; SMOKE = int(os.environ.get("SMOKE", 0))
INIT = os.environ.get("INIT", "")                      # warm start (char only, same SEL)
EPOCHS = int(os.environ.get("EPOCHS", 8)); MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 3000))
SADA_JSON = "data/sada/train.json"; SADA_HOURS = float(os.environ.get("SADA_HOURS", 60))
sys.path.insert(0, "src"); from telephony import degrade
random.seed(0); torch.manual_seed(0)
json.dump({k: v for k, v in dict(UNIT=UNIT, VOCAB=VOCAB, LORA=LORA, LORA_LAYERS=LORA_LAYERS, LORA_R=LORA_R, FILTER=FILTER,
                                  SEL=SEL, INIT=INIT, EPOCHS=EPOCHS, MAX_FRAMES=MAX_FRAMES, SADA_HOURS=SADA_HOURS).items()},
          open(f"{OUT}/config.json", "w"), indent=1)


def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("ـ", "")
    for d in range(0x064B, 0x0653): s = s.replace(chr(d), "")
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
         .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي").replace("ء", ""))
    s = re.sub(r"[،؍؛؟٪-٭۔]", " ", s)   # strip Arabic punctuation (inside the Arabic block)
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


# ---------------- data (identical split to v2/v3) ----------------
call = lambda p: os.path.basename(p).split("_w")[0]
gold = json.load(open("data/calls/test_windows.json"))
gold_calls = {call(g["audio_filepath"]) for g in gold}
coh = {r["audio_filepath"]: r for r in json.load(open("data/calls/train_windows_cohere.json"))}
asr2 = {r["audio_filepath"]: r for r in json.load(open("data/calls/train_windows_asr2.json"))}
wins = [p for p in coh if call(p) not in gold_calls]
calls = sorted({call(p) for p in wins}); random.shuffle(calls)
val_calls = set(calls[:max(20, len(calls) // 20)])
train, val, n_lab, n_drop = [], [], 0, 0
DIS = json.load(open("data/calls/teacher_disagreement.json"))
def tw(p):                                                  # weight from Cohere/second-recogniser disagreement (thresholds from our data)
    d = DIS.get(p)
    if d is None: return 0.5
    return 1.0 if d <= 0.25 else 0.5 if d <= 0.35 else 0.25 if d <= 0.45 else 0.0
for p in wins:
    labels = [t for t in (coh[p].get("cohere"), asr2.get(p, {}).get("asr2")) if t and norm(t)]
    if not labels: continue
    dur = coh[p]["duration"]; item = {"audio": p, "dur": dur}
    if call(p) in val_calls:
        val.append({**item, "text": labels[0], "alt": labels[-1]}); continue
    n_lab += len(labels)
    if FILTER:
        ok = [t for t in labels if 5 <= len(norm(t)) / max(dur, 0.1) <= 24]; n_drop += len(labels) - len(ok); labels = ok
    w_ = tw(p)
    if labels and w_ > 0: train.append({**item, "labels": labels, "w": w_})
EXTRA = os.environ.get("EXTRA_JSON", ""); EXTRA_REPEAT = int(os.environ.get("EXTRA_REPEAT", 1))   # e.g. voicebot calls
if EXTRA:
    n0 = len(train)
    for l in open(EXTRA):
        if l.strip():
            r = json.loads(l); labels = [t for t in r["labels"] if t and norm(t)]
            if labels: train += [{"audio": r["audio_filepath"], "dur": float(r["duration"]), "labels": labels}] * EXTRA_REPEAT
    print(f"EXTRA {EXTRA}: +{len(train) - n0} windows (x{EXTRA_REPEAT})", flush=True)
sada = []
for l in open(SADA_JSON):
    if l.strip():
        r = json.loads(l)
        if "غير واضح" in r["text"] or "غيرواضح" in r["text"]: continue      # SADA inaudible marker
        if norm(r["text"]): sada.append({"audio": r["audio_filepath"], "dur": float(r["duration"]), "labels": [r["text"]], "tel": True})
print(f"call train windows {len(train)} | labels kept {n_lab - n_drop}/{n_lab} (filter={FILTER}) | val {len(val)} | gold {len(gold)} "
      f"| SADA {len(sada)} windows, {SADA_HOURS} h/epoch", flush=True)

# ---------------- output units ----------------
if UNIT == "bpe":
    import sentencepiece as spm
    corpus = f"{OUT}/sp_corpus.txt"
    with open(corpus, "w") as f:
        for r in train:
            for t in r["labels"]: f.write(norm(t) + "\n")
        for r in sada: f.write(norm(r["labels"][0]) + "\n")
    if SP_MODEL: import shutil; shutil.copy(SP_MODEL, f"{OUT}/sp.model")
    else: spm.SentencePieceTrainer.train(input=corpus, model_prefix=f"{OUT}/sp", vocab_size=VOCAB, model_type="bpe",
                                   character_coverage=1.0, unk_id=0, bos_id=-1, eos_id=-1, num_threads=16)
    sp = spm.SentencePieceProcessor(model_file=f"{OUT}/sp.model")
    vocab = ["<b>"] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    enc = lambda t: [i + 1 for i in sp.encode(norm(t))]
    detok = lambda ids: sp.decode([i - 1 for i in ids])
else:
    chars = sorted({c for r in train for t in r["labels"] for c in norm(t)} | {c for r in sada for c in norm(r["labels"][0])})
    vocab = torch.load(INIT)["vocab"] if INIT else ["<b>"] + chars
    c2i = {c: i for i, c in enumerate(vocab)}
    enc = lambda t: [c2i[c] for c in norm(t) if c in c2i]
    detok = lambda ids: "".join(vocab[i] for i in ids)
print(f"UNIT={UNIT} vocab {len(vocab)}", flush=True)

# ---------------- encoder (+ LoRA) ----------------
mid = "google/embeddinggemma-2"
fe = AutoProcessor.from_pretrained(mid).feature_extractor
full = AutoModel.from_pretrained(mid, dtype=torch.bfloat16)
encoder = full.audio_tower.cuda().eval(); del full; torch.cuda.empty_cache()
for p in encoder.parameters(): p.requires_grad_(False)


class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32, drop=0.05):
        super().__init__()
        self.base = base; self.scale = alpha / r; self.drop = nn.Dropout(drop)
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device)); nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (self.drop(x).to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


lora_params = []
if LORA:
    for li in LORA_LAYERS:
        attn = encoder.layers[li].self_attn
        for name in ["q_proj", "k_proj", "v_proj", "post"]:
            holder = getattr(attn, name)
            assert isinstance(holder.linear, nn.Linear), type(holder.linear)
            holder.linear = LoRALinear(holder.linear, r=LORA_R, alpha=2 * LORA_R)
            lora_params += [holder.linear.A, holder.linear.B]
    # Gemma4AudioModel does not support HF gradient checkpointing -> LoRA runs use a smaller MAX_FRAMES instead
    print(f"LoRA r={LORA_R} on layers {LORA_LAYERS}: {sum(p.numel() for p in lora_params) / 1e6:.2f} M params", flush=True)


def load_wav(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    if sr != 16000:
        import librosa; w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return w


def encode(wavs, grad=False):
    f = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    encoder.train(grad and LORA)
    with torch.set_grad_enabled(grad and LORA), torch.autocast("cuda", dtype=torch.bfloat16):
        o = encoder(f["input_features"].cuda().to(torch.bfloat16), f["input_features_mask"].cuda(), output_hidden_states=True)
    hs = [o.hidden_states[i] for i in SEL] + ([o.last_hidden_state] if USE_OUT else [])
    lens = o.attention_mask.long().sum(-1).cpu()
    return hs, lens



def sinusoid(T, d, device):
    pos = torch.arange(T, device=device).float()[:, None]; i = torch.arange(0, d, 2, device=device).float()
    ang = pos / (10000 ** (i / d)); pe = torch.zeros(T, d, device=device); pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang)
    return pe


SOS = len(vocab); NV = len(vocab) + 1          # decoder vocabulary = BPE pieces (+1 offset, 0 unused) + <sos/eos>


class Reader(nn.Module):
    def __init__(self, dims, d=512, n=6, ndec=4):
        super().__init__()
        self.d = d
        self.mix = nn.Parameter(torch.zeros(len(dims)))
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d)) for k in dims])
        self.drop = nn.Dropout(0.1)
        if SUB == 1: self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)          # identical to v4 -> warm start
        else: self.sub = nn.Sequential(nn.Conv1d(d, d, 5, stride=SUB, padding=2), nn.GELU())
        self.pe_gate = nn.Parameter(torch.zeros(1))      # gated positions: starts as "no PE", learns how much to use
        self.sub_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n)
        self.enc_norm = nn.LayerNorm(d)
        self.ctc_out = nn.Linear(d, len(vocab))
        self.back = nn.Linear(len(vocab), d, bias=False); nn.init.zeros_(self.back.weight)   # SC-CTC feedback, starts as a no-op
        self.emb = nn.Embedding(NV, d)
        dl = nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, ndec)
        self.dec_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NV))

    def encode(self, hs, lens, augment=False):
        w = torch.softmax(self.mix, 0)
        x = sum(w[i] * self.proj[i](h.float()) for i, h in enumerate(hs))
        if augment:
            B, T, D = x.shape; x = x.clone()
            for b in range(B):
                for _ in range(2):
                    t = random.randint(0, max(1, T // 20)); s0 = random.randint(0, max(0, T - t)); x[b, s0:s0 + t] = 0
        x = self.drop(x)
        if SUB == 1: x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        else: x = self.sub_norm(self.sub(x.transpose(1, 2)).transpose(1, 2))
        lens2 = (lens + SUB - 1) // SUB
        x = x + self.pe_gate * sinusoid(x.shape[1], self.d, x.device)
        pad = torch.arange(x.shape[1], device=x.device)[None, :] >= lens2.to(x.device)[:, None]
        inter = []
        for i, layer in enumerate(self.tf.layers, 1):
            x = layer(x, src_key_padding_mask=pad)
            if i in INTER:                                   # self-conditioned CTC: shared head, posterior fed back
                z = self.ctc_out(self.enc_norm(x)); inter.append(z)
                x = x + self.back(z.float().softmax(-1).to(x.dtype))
        mem = self.enc_norm(x)
        return mem, pad, lens2, inter

    def decode(self, ys_in, mem, pad):
        L = ys_in.shape[1]
        y = self.emb(ys_in) * (self.d ** 0.5) + sinusoid(L, self.d, ys_in.device)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ys_in.device), 1)
        return self.dec_out(self.dec(y, mem, tgt_mask=causal, memory_key_padding_mask=pad))


with torch.no_grad():
    probe_hs, _ = encode([load_wav(train[0]["audio"])])
reader = Reader([h.shape[-1] for h in probe_hs]).cuda()
if WARM:                                                     # v4 BPE reader -> v5 encoder/CTC branch; decoder starts fresh
    w = torch.load(WARM); sd = {}
    old_sel = list(w.get("sel", [6, 8, 10, 11, 12])) + ["out"]; new_sel = SEL + (["out"] if USE_OUT else [])
    pmap = {old_sel.index(L): j for j, L in enumerate(new_sel) if L in old_sel}
    same_layers = old_sel[:len(new_sel)] == new_sel and len(w["reader"]["mix"]) == len(new_sel)
    for k, v in w["reader"].items():
        if k == "mix" and not same_layers: continue        # mixer only transfers when the layer set is identical
        if k.startswith("proj."):
            i = int(k.split(".")[1])
            if i not in pmap: continue
            k = "proj." + str(pmap[i]) + k[len("proj." + str(i)):]
        if k.startswith("out.0."): k = "enc_norm." + k[6:]          # v4 checkpoints only
        elif k.startswith("out.1."): k = "ctc_out." + k[6:]
        sd[k] = v
    print("projection map old->new:", {old_sel[i]: new_sel[j] for i, j in pmap.items()}, flush=True)
    miss = reader.load_state_dict(sd, strict=False)
    print("warm start from", WARM, "| missing (new):", sorted({m.split('.')[0] for m in miss.missing_keys}), "| unused:", miss.unexpected_keys, flush=True)
    if LORA and w.get("lora"):
        assert len(w["lora"]) == len(lora_params)
        for p_, v in zip(lora_params, w["lora"]): p_.data.copy_(v.to(p_.device))
        print("LoRA adapters loaded:", len(w["lora"]), flush=True)
if INIT:                                                     # continue a previous v5 run (same SentencePiece model required)
    ci = torch.load(INIT); reader.load_state_dict(ci["reader"]); print("continue from", INIT, flush=True)
    if LORA and ci.get("lora"):
        assert len(ci["lora"]) == len(lora_params)
        for p_, v in zip(lora_params, ci["lora"]): p_.data.copy_(v.to(p_.device))
print(f"layers {SEL + (['out'] if USE_OUT else [])} | reader {sum(p.numel() for p in reader.parameters()) / 1e6:.1f} M params | CTC_W {CTC_W}", flush=True)
dec_params = [p for n, p in reader.named_parameters() if n.split(".")[0] in ("emb", "dec", "dec_out")]
rest_params = [p for n, p in reader.named_parameters() if n.split(".")[0] not in ("emb", "dec", "dec_out")]
if STAGE == "dec":                                          # stage 1: only the decoder learns, on frozen reader states
    for p in rest_params + lora_params: p.requires_grad_(False)
    groups = [{"params": dec_params, "lr": 5e-4}]
else:                                                       # stage 2: everything, decoder fastest, encoder side slowest
    groups = [{"params": dec_params, "lr": 3e-4}, {"params": rest_params, "lr": 1.5e-4}] + \
             ([{"params": lora_params, "lr": LORA_LR, "weight_decay": 0.0}] if LORA else [])
print(f"STAGE={STAGE} | trainable {sum(p.numel() for g in groups for p in g['params']) / 1e6:.2f} M | losses ctc {W_CTC} att {W_ATT} inter {W_INT}", flush=True)
opt = torch.optim.AdamW(groups, weight_decay=1e-2)
ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")


def ctc_w(logits, targets, in_len, tgt_len, wts):         # per-window CTC, length-normalised, label-quality weighted
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    l = ctc(lp, targets, in_len, tgt_len) / tgt_len.clamp(min=1).to(lp.device)
    return (l * wts).sum() / wts.sum().clamp(min=1e-6)


def att_w(att, ys_out, wts):                               # EOS-aware smoothing: 0.05 on content, 0 on EOS, EOS x1.5
    logp = att.float().log_softmax(-1); m = ys_out != -100; t = ys_out.clamp(min=0)
    nll = -logp.gather(-1, t[..., None])[..., 0]; smooth = -logp.mean(-1); eos = t == SOS
    eps = torch.where(eos, torch.zeros_like(nll), torch.full_like(nll, 0.05))
    tok = ((1 - eps) * nll + eps * smooth) * torch.where(eos, torch.full_like(nll, 1.5), torch.ones_like(nll))
    per = (tok * m).sum(1) / m.sum(1).clamp(min=1)
    return (per * wts).sum() / wts.sum().clamp(min=1e-6)
ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)


def batches(items, max_frames=MAX_FRAMES):
    items = sorted(items, key=lambda r: r["dur"]); out, cur, mx = [], [], 0
    for r in items:
        fr = int(r["dur"] * 25) + 5; mx2 = max(mx, fr)
        if cur and mx2 * (len(cur) + 1) > max_frames: out.append(cur); cur, mx2 = [], fr
        cur.append(r); mx = mx2
    if cur: out.append(cur)
    random.shuffle(out); return out


def ctc_greedy(logits, lens):
    outs = []
    for b in range(logits.shape[0]):
        ids = logits[b, :int(lens[b])].argmax(-1).tolist(); s, prev = [], 0
        for t in ids:
            if t != prev and t != 0: s.append(t)
            prev = t
        outs.append(detok(s))
    return outs


@torch.no_grad()
def att_greedy(mem, pad, lens2, cap=None):
    B = mem.shape[0]; ys = torch.full((B, 1), SOS, dtype=torch.long, device=mem.device)
    done = torch.zeros(B, dtype=torch.bool, device=mem.device); maxlen = int(lens2.max()) // 2 + 10
    if cap is not None: maxlen = min(maxlen, max(int(max(cap) * 1.5) + 5, int(lens2.max()) // 8 + 10))
    for _ in range(maxlen):
        nxt = reader.decode(ys, mem, pad)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, SOS), nxt)
        ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == SOS
        if done.all(): break
    outs = []
    for b in range(B):
        s = []
        for t in ys[b, 1:].tolist():
            if t == SOS: break
            if t > 0: s.append(t)
        outs.append(detok(s))
    return outs


@torch.no_grad()
def transcribe(paths, bs=4):
    reader.eval(); hc, ha = [], []
    for i in range(0, len(paths), bs):
        hs, lens = encode([load_wav(p) for p in paths[i:i + bs]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem, pad, lens2, _ = reader.encode(hs, lens)
            hcb = ctc_greedy(reader.ctc_out(mem).float(), lens2); hc += hcb
            ha += att_greedy(mem, pad, lens2, cap=[len(enc(h)) for h in hcb])
        del hs
    reader.train(); return hc, ha


def save(path):
    torch.save({"reader": reader.state_dict(), "vocab": vocab, "unit": UNIT, "sel": SEL, "lora_layers": LORA_LAYERS if LORA else [],
                "lora_r": LORA_R, "lora": [p.detach().cpu() for p in lora_params], "arch": "hybrid_v5"}, path)


def epoch_items():
    random.shuffle(sada); pick, h = [], 0.0
    for r in sada:
        if h >= SADA_HOURS * 3600: break
        pick.append(r); h += r["dur"]
    return train + pick


EXPECT = float(os.environ.get("EXPECT_CTC", "0"))
if EXPECT > 0:
    bc, _ = transcribe([r["audio"] for r in val])
    b0 = 100 * np.mean([min(wer(r["text"], h), wer(r["alt"], h)) for r, h in zip(val, bc)])
    print(f"BASELINE CHECK: warm-started CTC val WER {b0:.2f}% (expected {EXPECT:.2f}%)", flush=True)
    if abs(b0 - EXPECT) > 1.0:
        print("BASELINE CHECK FAILED: warm start does not reproduce the source model; stopping.", flush=True); sys.exit(2)
steps_per_epoch = len(batches(epoch_items())); total = EPOCHS * steps_per_epoch + 50
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total,
                                            pct_start=min(0.3, 1500 / total) if STAGE == "dec" else 0.05)
trainable = [p for g in groups for p in g["params"]]
best, t0 = (1e9, 0), time.time()
for ep in range(1, EPOCHS + 1):
    tot = n = 0
    for b in batches(epoch_items()):
        wavs = [degrade(load_wav(r["audio"])).astype(np.float32) if r.get("tel") else load_wav(r["audio"]) for r in b]
        hs, lens = encode(wavs, grad=(STAGE != "dec"))
        ys = [enc(random.choice(r["labels"])) for r in b]
        lens2 = (lens + SUB - 1) // SUB
        keep = [k for k, y in enumerate(ys) if 0 < len(y) and len(y) + sum(a == b for a, b in zip(y, y[1:])) <= int(lens2[k])]
        if not keep: continue
        yk = [ys[k] for k in keep]; L = max(len(y) for y in yk) + 1
        ys_in = torch.full((len(yk), L), SOS, dtype=torch.long); ys_out = torch.full((len(yk), L), -100, dtype=torch.long)
        for j, y in enumerate(yk):
            ys_in[j, 1:len(y) + 1] = torch.tensor(y); ys_out[j, :len(y)] = torch.tensor(y); ys_out[j, len(y)] = SOS
        wts = torch.tensor([b[k].get("w", 1.0) for k in keep], dtype=torch.float32).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if STAGE == "dec":
                reader.eval()                                # frozen reader: no dropout, exactly the features seen at test time
                with torch.no_grad(): mem, pad, l2, inter = reader.encode([h[keep] for h in hs], lens[keep])
                reader.train()
            else:
                mem, pad, l2, inter = reader.encode([h[keep] for h in hs], lens[keep], augment=True)
            att = reader.decode(ys_in.cuda(), mem, pad)
        loss_att = att_w(att, ys_out.cuda(), wts)
        if STAGE == "dec":
            loss = loss_att
        else:
            tg, tl = torch.tensor(sum(yk, [])), torch.tensor([len(y) for y in yk])
            loss_ctc = ctc_w(reader.ctc_out(mem.float()), tg, l2, tl, wts)
            loss_int = sum(ctc_w(z, tg, l2, tl, wts) for z in inter) / max(len(inter), 1)
            loss = W_CTC * loss_ctc + W_ATT * loss_att + W_INT * loss_int
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(trainable, 5.0); opt.step()
        if sched.last_epoch < total - 1: sched.step()
        tot += loss.item(); n += 1; del hs, mem, att
        if SMOKE and n >= SMOKE:
            print(f"SMOKE ok: {n} steps, last loss {loss.item():.3f}, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True); sys.exit(0)
    vc_h, va_h = transcribe([r["audio"] for r in val])
    vw_c = 100 * np.mean([min(wer(r["text"], h), wer(r["alt"], h)) for r, h in zip(val, vc_h)])
    vw_a = 100 * np.mean([min(wer(r["text"], h), wer(r["alt"], h)) for r, h in zip(val, va_h)])
    va_c = 100 * np.mean([min(cer(r["text"], h), cer(r["alt"], h)) for r, h in zip(val, va_h)])
    ratio = np.median([len(h.split()) / max(1, len(r["text"].split())) for r, h in zip(val, va_h)])
    mix = torch.softmax(reader.mix, 0).detach().cpu().numpy()
    print(f"ep {ep:2d} loss {tot / max(n, 1):.3f}  val WER ctc {vw_c:5.2f}%  att {vw_a:5.2f}% (att CER {va_c:5.2f}%, att/ref length {ratio:.2f})  top layers "
          f"{[(SEL + (['out'] if USE_OUT else []))[i] for i in np.argsort(-mix)[:3]]}  ({time.time() - t0:.0f}s)", flush=True)
    vw = vw_a if STAGE == "dec" else min(vw_c, vw_a)
    if vw < best[0]:
        best = (vw, ep); save(f"{OUT}/reader_best.pt")

ck = torch.load(f"{OUT}/reader_best.pt")
reader.load_state_dict(ck["reader"])
for p, v in zip(lora_params, ck["lora"]): p.data.copy_(v.to(p.device))
gc, ga = transcribe([g["audio_filepath"] for g in gold])
for tag, gh in [("ctc greedy", gc), ("attention greedy", ga)]:
    W = 100 * np.mean([wer(g["corrected"], h) for g, h in zip(gold, gh)]); C = 100 * np.mean([cer(g["corrected"], h) for g, h in zip(gold, gh)])
    print(f"GOLD 72 windows {tag:17s} (best epoch {best[1]}):  WER {W:.2f}%  CER {C:.2f}%   [Cohere: WER 40.48%  CER 20.46%]", flush=True)
json.dump([{"audio": g["audio_filepath"], "ref": g["corrected"], "ctc": a, "att": b} for g, a, b in zip(gold, gc, ga)],
          open(f"{OUT}/gold_hyp.json", "w"), ensure_ascii=False, indent=1)
print("layer mix:", np.round(torch.softmax(reader.mix, 0).detach().cpu().numpy(), 3).tolist(), flush=True)
print("TRAIN_DONE", flush=True)
