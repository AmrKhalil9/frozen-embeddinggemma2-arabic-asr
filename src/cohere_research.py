"""Cohere-Arabic-2B reference on the same research eval sets + SADA test (clean / tel).
Same config as the gold reference (greedy, Silero VAD max 25 s, no_repeat_ngram 4, rep penalty 1.2) and the same norm/WER."""
import json, os, sys, numpy as np, soundfile as sf, torch, librosa
sys.path.insert(0, "src"); from cohere_decode_lib import wer, cer   # norm/WER copied verbatim from cohere_decode.py
sys.path.insert(0, "src"); from telephony import degrade
os.environ.setdefault("HF_HOME", "hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
MID = "CohereLabs/cohere-transcribe-arabic-07-2026"
proc = AutoProcessor.from_pretrained(MID)
DEV = os.environ.get("DEV", "cuda"); DT = torch.float16 if DEV == "cuda" else torch.float32
torch.set_num_threads(int(os.environ.get("THREADS", "40")))
model = CohereAsrForConditionalGeneration.from_pretrained(MID, torch_dtype=DT).to(DEV).eval()
vad = load_silero_vad(onnx=True)
E, OUT = "data/eval", "results/cohere_research.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def load(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    return librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a


def transcribe(a):
    ts = get_speech_timestamps(torch.from_numpy(a), vad, sampling_rate=16000, min_silence_duration_ms=250, speech_pad_ms=150, max_speech_duration_s=25)
    segs = [a[t["start"]:t["end"]] for t in ts]; out_txt = []
    for i in range(0, len(segs), 16):
        inp = proc(segs[i:i + 16], sampling_rate=16000, return_tensors="pt", language="ar").to(DEV, dtype=DT)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=256, num_beams=1, do_sample=False, no_repeat_ngram_size=4, repetition_penalty=1.2)
        out_txt += [x.strip() for x in proc.batch_decode(out, skip_special_tokens=True) if x.strip()]
    return " ".join(out_txt)


SETS = [(f"{n}|{t}", f"{E}/{n}/{m}", False) for n in ["nexdata_cs_sample", "casablanca_uae"]
        for t, m in [("clean16k", "test.json"), ("tel8k", "test_tel.json")]]
SETS += [("sauditalk_longform|clean16k", f"{E}/sauditalk_longform/test.json", False), ("sauditalk_longform|tel8k", f"{E}/sauditalk_longform/test.json", True)]
SETS += [("sada_test|clean16k", "data/sada/test.json", False), ("sada_test|tel8k", "data/sada/test_tel.json", False)]
for tag, man, tel in SETS:
    if tag in res: continue
    rows = [json.loads(l) for l in open(man) if l.strip()]
    hyp = []
    for r in rows:
        a = load(r["audio_filepath"])
        hyp.append(transcribe(degrade(a).astype(np.float32) if tel else a))
    W = 100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hyp)]); C = 100 * np.mean([cer(r["text"], h) for r, h in zip(rows, hyp)])
    res[tag] = {"wer": round(W, 2), "cer": round(C, 2), "n": len(rows), "hyp": hyp}
    json.dump(res, open(OUT, "w"), ensure_ascii=False)
    print(f"COHERE {tag:32s} WER {W:6.2f}%  CER {C:6.2f}%  (n={len(rows)})", flush=True)
print("COHERE_RESEARCH_DONE", flush=True)
