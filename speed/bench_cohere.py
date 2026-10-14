"""Cohere Transcribe Arabic 2B: greedy generate with the paper's settings, batched, same audio (no VAD: utterances are short)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
torch.set_num_threads(THREADS)
os.environ["HF_HOME"] = "hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
sys.path.insert(0, "src"); from cohere_decode_lib import wer
MID = "CohereLabs/cohere-transcribe-arabic-07-2026"; DT = torch.float16 if DEV == "cuda" else torch.float32
proc = AutoProcessor.from_pretrained(MID); model = CohereAsrForConditionalGeneration.from_pretrained(MID, torch_dtype=DT).to(DEV).eval()
sync = torch.cuda.synchronize if DEV == "cuda" else (lambda: None)


@torch.inference_mode()
def run(wavs):
    out = []
    for i in range(0, len(wavs), BS):
        inp = proc(wavs[i:i + BS], sampling_rate=16000, return_tensors="pt", language="ar").to(DEV, dtype=DT)
        g = model.generate(**inp, max_new_tokens=256, num_beams=1, do_sample=False, no_repeat_ngram_size=4, repetition_penalty=1.2)
        out += [x.strip() for x in proc.batch_decode(g, skip_special_tokens=True)]
    return out


rows, wavs = load_set(); audio = sum(len(w) for w in wavs) / 16000
run(wavs[:BS])
if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
with Timer(sync) as t: hyp = run(wavs)
peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else None
save("cohere", audio, t.s, {"n": len(rows), "peak_gpu_gib": round(peak, 2) if peak else None, "precision": str(DT).split(".")[-1],
                            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                            "wer": round(100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hyp)]), 2)})
