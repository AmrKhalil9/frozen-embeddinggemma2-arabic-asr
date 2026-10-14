"""Decode the v2 reader with CTC beam search + KenLM. alpha/beta are tuned on the held-out VALIDATION calls only;
the 72 gold windows are scored once with the chosen values (no test-set tuning)."""
import json, os, sys, itertools, numpy as np, torch
os.environ.setdefault("EPOCHS", "0")
CK = os.environ.get("CK", "runs/v2/reader_best.pt"); OUTD = os.path.dirname(CK)

# reuse data/model definitions from the training script without training (EPOCHS=0 path is skipped below)
src = open("src/train_v2.py").read()
head = src.split("steps_per_epoch = len(batches(train))")[0]
ns = {"__name__": "eval"}; exec(head, ns)
reader, encode, load_wav, vocab, val, gold = ns["reader"], ns["encode"], ns["load_wav"], ns["vocab"], ns["val"], ns["gold"]
wer, cer = ns["wer"], ns["cer"]
ck = torch.load(CK); reader.load_state_dict(ck["reader"]); reader.eval()
assert ck["vocab"] == vocab, "vocab mismatch"


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


from pyctcdecode import build_ctcdecoder
labels = [""] + vocab[1:]                                       # blank first, characters (incl. space)
V = logprobs([r["audio"] for r in val]); G = logprobs([g["audio_filepath"] for g in gold])


def greedy(lp):
    ids = lp.argmax(-1).tolist(); s, prev = [], 0
    for t in ids:
        if t != prev and t != 0: s.append(vocab[t])
        prev = t
    return "".join(s)


def score_val(hyps): return 100 * np.mean([min(wer(r["text"], h), wer(r["alt"], h)) for r, h in zip(val, hyps)])
def score_gold(hyps):
    return (100 * np.mean([wer(g["corrected"], h) for g, h in zip(gold, hyps)]),
            100 * np.mean([cer(g["corrected"], h) for g, h in zip(gold, hyps)]))


res = {"greedy": score_gold([greedy(x) for x in G])}
print(f"greedy            gold WER {res['greedy'][0]:.2f}%  CER {res['greedy'][1]:.2f}%   (val WER {score_val([greedy(x) for x in V]):.2f}%)", flush=True)
for name in ["calls", "calls_sada"]:
    lm = f"lm/lm_{name}.bin"; uni = open(f"lm/unigrams_{name}.txt").read().split("\n")
    best = (1e9, None)
    for a, b in itertools.product([0.3, 0.5, 0.7, 1.0], [0.0, 1.0, 2.0]):
        dec = build_ctcdecoder(labels, kenlm_model_path=lm, unigrams=uni, alpha=a, beta=b)
        vw = score_val([dec.decode(x, beam_width=64) for x in V])
        if vw < best[0]: best = (vw, (a, b))
    a, b = best[1]
    dec = build_ctcdecoder(labels, kenlm_model_path=lm, unigrams=uni, alpha=a, beta=b)
    gh = [dec.decode(x, beam_width=128) for x in G]
    res[name] = score_gold(gh)
    print(f"LM {name:12s}  alpha {a} beta {b} (val WER {best[0]:.2f}%)  ->  gold WER {res[name][0]:.2f}%  CER {res[name][1]:.2f}%", flush=True)
    json.dump([{"ref": g["corrected"], "hyp": h} for g, h in zip(gold, gh)], open(f"{OUTD}/gold_hyp_lm_{name}.json", "w"), ensure_ascii=False, indent=1)
print("Cohere reference:  gold WER 40.48%  CER 20.46%", flush=True)
print("EVAL_DONE", flush=True)
