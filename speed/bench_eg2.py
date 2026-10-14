"""Frozen EmbeddingGemma-2 encoder + 22.6M CTC reader (v3_tel): encoder+reader+greedy timed on DEV; 4-gram beam search timed on CPU."""
import os, sys, json, re, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__)); from common import *
torch.set_num_threads(THREADS)
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, "src"); from cohere_decode_lib import wer, cer
SEL = [6, 8, 10, 11, 12]; mid = "google/embeddinggemma-2"
dt = torch.bfloat16 if DEV == "cuda" else torch.float32
fe = AutoProcessor.from_pretrained(mid).feature_extractor
full = AutoModel.from_pretrained(mid, dtype=dt); encoder = full.audio_tower.to(DEV).eval(); del full
ck = torch.load("runs/v3_tel/reader_best.pt", map_location="cpu"); vocab = ck["vocab"]


class Reader(nn.Module):   # identical to train_v3.py
    def __init__(self, dims, d=512, n=6, v=len(vocab)):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(len(dims)))
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d)) for k in dims])
        self.drop = nn.Dropout(0.1); self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n); self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, v))
    def forward(self, hs, lens):
        w = torch.softmax(self.mix, 0); x = sum(w[i] * self.proj[i](h.float()) for i, h in enumerate(hs))
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        T = x.shape[1]; pad = torch.arange(T, device=x.device)[None, :] >= lens.to(x.device)[:, None]
        return self.out(self.tf(x, src_key_padding_mask=pad))


reader = Reader([1024, 1024, 1024, 1024, 1536, 1536]); reader.load_state_dict(ck["reader"]); reader.to(DEV).eval()
sync = torch.cuda.synchronize if DEV == "cuda" else (lambda: None)


FE_S = [0.0]; ENC_S = [0.0]
@torch.inference_mode()
def run(wavs):
    out = []
    for i in range(0, len(wavs), BS):
        t0 = time.perf_counter(); f = fe(wavs[i:i + BS], sampling_rate=16000, return_tensors="pt", padding=True); FE_S[0] += time.perf_counter() - t0
        x, xm = f["input_features"].to(DEV, dt), f["input_features_mask"].to(DEV); sync(); t1 = time.perf_counter()
        o = encoder(x, xm, output_hidden_states=True); sync(); ENC_S[0] += time.perf_counter() - t1
        hs = [o.hidden_states[k] for k in SEL] + [o.last_hidden_state]; lens = o.attention_mask.long().sum(-1)
        lp = reader(hs, lens).float().log_softmax(-1).cpu().numpy()
        out += [lp[b, :int(lens[b])] for b in range(lp.shape[0])]
    return out


def greedy(lp):
    s, prev = [], 0
    for t in lp.argmax(-1).tolist():
        if t != prev and t != 0: s.append(vocab[t])
        prev = t
    return "".join(s)


rows, wavs = load_set(); audio = sum(len(w) for w in wavs) / 16000
run(wavs[:BS])                                                    # warm-up
if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
FE_S[0] = 0.0; ENC_S[0] = 0.0
with Timer(sync) as t: L = run(wavs); hyp = [greedy(l) for l in L]
peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else None
extra = {"n": len(rows), "peak_gpu_gib": round(peak, 2) if peak else None, "precision": str(dt).split(".")[-1], "feature_extraction_s": round(FE_S[0], 2), "encoder_only_s": round(ENC_S[0], 2), "encoder_only_rtfx": round(audio / ENC_S[0], 1),
         "params_m": round((sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in reader.parameters())) / 1e6, 1),
         "wer": round(100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hyp)]), 2)}
save("eg2_greedy", audio, t.s, dict(extra))
# 4-gram beam search on CPU (single process), same settings as the paper
from pyctcdecode import build_ctcdecoder
uni = open("lm/unigrams_calls_sada.txt").read().split("\n")
dec = build_ctcdecoder([""] + vocab[1:], kenlm_model_path="lm/lm_calls_sada.bin", unigrams=uni, alpha=0.5, beta=2.0)
dec.decode(L[0], beam_width=64)
with Timer(lambda: None) as t2: hl = [dec.decode(l, beam_width=64) for l in L]
save("eg2_lm_beam_only_cpu1", audio, t2.s, {"n": len(rows), "wer": round(100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hl)]), 2),
                                             "note": "pyctcdecode beam 64 + 4-gram, one CPU process, on precomputed log-probs"})
