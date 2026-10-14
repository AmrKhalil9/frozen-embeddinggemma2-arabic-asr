"""Score a reader checkpoint on the research-only Saudi/Gulf eval sets (paper only).
Windowed sets: one utterance per row. SaudiTalk: long clips -> Silero VAD segments (<=25 s), hyps joined per clip.
Greedy + KenLM (lm_calls_sada, alpha/beta from the checkpoint's eval_lm.log)."""
import json, os, re, numpy as np, torch
from multiprocessing import get_context
CK = os.environ["CK"]; OUTD = os.path.dirname(CK); E = "data/eval"
os.environ.setdefault("EPOCHS", "0")
src = open("src/train_v2.py").read()
ns = {"__name__": "eval"}; exec(src.split("steps_per_epoch = len(batches(train))")[0], ns)
reader, encode, load_wav, vocab, wer, cer = ns["reader"], ns["encode"], ns["load_wav"], ns["vocab"], ns["wer"], ns["cer"]
ck = torch.load(CK); reader.load_state_dict(ck["reader"]); reader.eval(); assert ck["vocab"] == vocab
import sys; sys.path.insert(0, "src"); from telephony import degrade
from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad(onnx=True)

alpha, beta = 0.5, 1.0
m = re.search(r"LM calls_sada\s+alpha ([\d.]+) beta ([\d.]+)", open(os.path.join(OUTD, "eval_lm.log")).read())
if m: alpha, beta = float(m.group(1)), float(m.group(2))


@torch.no_grad()
def logprobs(wavs, bs=int(os.environ.get("BS", "4"))):
    out = []
    for i in range(0, len(wavs), bs):
        hs, lens = encode(wavs[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = reader(hs, lens)
        lp = lg.float().log_softmax(-1).cpu().numpy()
        out += [lp[b, :int(lens[b])] for b in range(lp.shape[0])]
        del hs
    return out


def greedy(lp):
    ids = lp.argmax(-1).tolist(); s, prev = [], 0
    for t in ids:
        if t != prev and t != 0: s.append(vocab[t])
        prev = t
    return "".join(s)


_dec = None
def _init():
    global _dec
    from pyctcdecode import build_ctcdecoder
    uni = open("lm/unigrams_calls_sada.txt").read().split("\n")
    _dec = build_ctcdecoder([""] + vocab[1:], kenlm_model_path="lm/lm_calls_sada.bin", unigrams=uni, alpha=alpha, beta=beta)
def _beam(lp): return _dec.decode(lp, beam_width=64)


def segments(w):
    ts = get_speech_timestamps(torch.from_numpy(w), vad, sampling_rate=16000, min_silence_duration_ms=250, speech_pad_ms=150, max_speech_duration_s=25)
    return [w[t["start"]:t["end"]] for t in ts]


SETS = [(n, f) for n in ["nexdata_cs_sample", "casablanca_uae"] for f in ["test.json", "test_tel.json"]]
SETS += [("sauditalk_longform", "test.json"), ("sauditalk_longform", "tel")]
res, hyps_out = {}, {}
for name, f in SETS:
    rows = [json.loads(l) for l in open(f"{E}/{name}/{'test.json' if f == 'tel' else f}") if l.strip()]
    tag = f"{name}|{'tel8k' if 'tel' in f else 'clean16k'}"
    if name == "sauditalk_longform":
        seg_of, wavs = [], []
        for k, r in enumerate(rows):
            w = load_wav(r["audio_filepath"])
            if f == "tel": w = degrade(w)
            s = segments(w.astype(np.float32)); wavs += s; seg_of += [k] * len(s)
    else:
        wavs = [load_wav(r["audio_filepath"]) for r in rows]; seg_of = list(range(len(rows)))
    L = logprobs(wavs)
    with get_context("fork").Pool(12, initializer=_init) as pool:
        b = pool.map(_beam, L, chunksize=8)
    for dec, hs in [("greedy", [greedy(x) for x in L]), ("lm_calls_sada", b)]:
        joined = [" ".join(h for h, k in zip(hs, seg_of) if k == i) for i in range(len(rows))]
        W = 100 * np.mean([wer(r["text"], h) for r, h in zip(rows, joined)]); C = 100 * np.mean([cer(r["text"], h) for r, h in zip(rows, joined)])
        res[f"{tag}|{dec}"] = (round(W, 2), round(C, 2)); hyps_out[f"{tag}|{dec}"] = joined
        print(f"{tag:32s} {dec:11s} WER {W:6.2f}%  CER {C:6.2f}%   (n={len(rows)})", flush=True)
json.dump(res, open(os.path.join(OUTD, "research_results.json"), "w"), indent=1)
json.dump(hyps_out, open(os.path.join(OUTD, "research_hyps.json"), "w"), ensure_ascii=False)
print("RESEARCH_EVAL_DONE", flush=True)
