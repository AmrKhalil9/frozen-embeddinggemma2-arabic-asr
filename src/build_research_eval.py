"""Build 16 kHz mono eval manifests (+ 8 kHz telephony twins) from the research-only sets in data/raw.
Research/paper evaluation only (licences are non-commercial or unstated)."""
import glob, io, json, os, re, sys, csv
import numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq
sys.path.insert(0, "src"); from telephony import degrade
D, O = "data/raw", "data/eval"
MAX_S = 60.0


def save(name, i, y, sr, text, rows):
    if not text.strip(): return
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)
    dur = len(y) / 16000
    if dur < 0.5 or dur > MAX_S: return
    p = f"{O}/{name}/wav/{i}.wav"; sf.write(p, y, 16000)
    rows.append({"audio_filepath": p, "text": text.strip(), "duration": round(dur, 2)})


def write(name, rows):
    with open(f"{O}/{name}/test.json", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(f"{O}/{name}/wav_tel", exist_ok=True)
    with open(f"{O}/{name}/test_tel.json", "w") as f:
        for r in rows:
            y, _ = sf.read(r["audio_filepath"], dtype="float32")
            t = r["audio_filepath"].replace("/wav/", "/wav_tel/"); sf.write(t, degrade(y), 16000)
            f.write(json.dumps({**r, "audio_filepath": t}, ensure_ascii=False) + "\n")
    print(f"{name:22s} {len(rows):5d} utts  {sum(r['duration'] for r in rows) / 3600:5.2f} h", flush=True)


def from_parquet(name, pattern, tcol, clean=lambda s: s):
    os.makedirs(f"{O}/{name}/wav", exist_ok=True); rows = []
    for f in sorted(glob.glob(f"{D}/{pattern}")):
        t = pq.read_table(f, columns=["audio", tcol])
        for k, (a, txt) in enumerate(zip(t.column("audio").to_pylist(), t.column(tcol).to_pylist())):
            y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if y.ndim > 1: y = y.mean(1)
            save(name, f"{os.path.basename(f)[:-8]}_{k}", y, sr, clean(txt or ""), rows)
    write(name, rows)

from_parquet("casablanca_uae", "UBC-NLP__Casablanca/UAE/test*.parquet", "transcription")

# Nexdata Saudi customer-service sample: one leg, human timestamps -> per-turn segments
name = "nexdata_cs_sample"; os.makedirs(f"{O}/{name}/wav", exist_ok=True); rows = []
for w in glob.glob(f"{D}/Nexdata*/**/*.wav", recursive=True):
    y, sr = sf.read(w, dtype="float32"); txt = glob.glob(os.path.join(os.path.dirname(w), "*.txt"))[0]
    for k, line in enumerate(open(txt)):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4: save(name, k, y[int(float(p[0]) * sr):int(float(p[1]) * sr)], sr, p[3], rows)
write(name, rows)

# SaudiTalk: whole-clip human transcripts (avg ~2.7 min, no timestamps) -> long-form list only, not windowed
st = list(csv.DictReader(open(f"{D}/SaudiTalk__SaudiTalk/dataset.csv")))
os.makedirs(f"{O}/sauditalk_longform", exist_ok=True)
with open(f"{O}/sauditalk_longform/test.json", "w") as f:
    for r in st:
        p = f"{D}/SaudiTalk__SaudiTalk/{r['audio']}"
        if os.path.exists(p): f.write(json.dumps({"audio_filepath": p, "text": r["transcript"], "dialect": r["dialect"]}, ensure_ascii=False) + "\n")
print("sauditalk_longform     ", len(st), "clips (long-form; needs VAD windows + alignment or long-form decoding)")
print("BUILD_DONE")
