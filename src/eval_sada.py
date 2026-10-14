"""Score a reader checkpoint on the SADA22 test set (research only), clean 16 kHz and telephony-degraded 8 kHz.
Greedy + KenLM beam search (lm_calls_sada; alpha/beta = the values chosen on validation calls by eval_lm.py)."""
import json, os, re, numpy as np, torch
from multiprocessing import get_context
CK = os.environ["CK"]; OUTD = os.path.dirname(CK)
os.environ.setdefault("EPOCHS", "0")
src = open("src/train_v2.py").read()
ns = {"__name__": "eval"}; exec(src.split("steps_per_epoch = len(batches(train))")[0], ns)
reader, encode, load_wav, vocab, wer, cer = ns["reader"], ns["encode"], ns["load_wav"], ns["vocab"], ns["wer"], ns["cer"]
ck = torch.load(CK); reader.load_state_dict(ck["reader"]); reader.eval(); assert ck["vocab"] == vocab

alpha, beta = 0.5, 1.0                                   # fallback; replaced by eval_lm.py's choice when available
log = os.path.join(OUTD, "eval_lm.log")
if os.path.exists(log):
    m = re.search(r"LM calls_sada\s+alpha ([\d.]+) beta ([\d.]+)", open(log).read())
    if m: alpha, beta = float(m.group(1)), float(m.group(2))


@torch.no_grad()
def logprobs(paths, bs=4):
    out = []
    for i in range(0, len(paths), bs):
        hs, lens = encode([load_wav(p) for p in paths[i:i + bs]])
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


res = {}
for name, man in [("sada_test_clean16k", "data/sada/test.json"), ("sada_test_tel8k", "data/sada/test_tel.json")]:
    rows = [json.loads(l) for l in open(man) if l.strip()]
    L = logprobs([r["audio_filepath"] for r in rows])
    g = [greedy(x) for x in L]
    with get_context("fork").Pool(12, initializer=_init) as pool:
        b = pool.map(_beam, L, chunksize=8)
    for tag, hyps in [("greedy", g), ("lm_calls_sada", b)]:
        W = 100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hyps)]); C = 100 * np.mean([cer(r["text"], h) for r, h in zip(rows, hyps)])
        res[f"{name}|{tag}"] = (round(W, 2), round(C, 2))
        print(f"{name:20s} {tag:11s} WER {W:6.2f}%  CER {C:6.2f}%   (n={len(rows)}, alpha {alpha} beta {beta})", flush=True)
json.dump(res, open(os.path.join(OUTD, "sada_test_results.json"), "w"), indent=1)
print("SADA_EVAL_DONE", flush=True)
