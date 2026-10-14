"""Shared protocol for the speed benchmark: same audio for every system, loaded into RAM before timing,
warm-up excluded, wall-clock timing with device synchronisation. RTFx = seconds of audio processed per second of wall time."""
import json, os, time, numpy as np, soundfile as sf
MAN = "data/eval/casablanca_uae/test.json"
OUTD = "results/speed"; os.makedirs(OUTD, exist_ok=True)
DEV = os.environ.get("DEV", "cuda"); THREADS = int(os.environ.get("THREADS", "16")); BS = int(os.environ.get("BS", "16"))
N = int(os.environ.get("N", "0"))   # 0 = all utterances


def load_set():
    rows = [json.loads(l) for l in open(MAN) if l.strip()]
    if N: rows = rows[:N]
    wavs = []
    for r in rows:
        w, sr = sf.read(r["audio_filepath"], dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        assert sr == 16000, sr
        wavs.append(w)
    if os.environ.get("SORT") == "1":   # length-sorted batches (less padding); applied identically to every system
        o = sorted(range(len(wavs)), key=lambda i: len(wavs[i])); rows = [rows[i] for i in o]; wavs = [wavs[i] for i in o]
    return rows, wavs


def save(name, audio_s, wall_s, extra):
    rec = {"system": name, "device": DEV, "sorted": os.environ.get("SORT") == "1", "threads": THREADS, "batch": BS, "n": extra.pop("n"), "audio_s": round(audio_s, 1),
           "wall_s": round(wall_s, 2), "rtfx": round(audio_s / wall_s, 1), **extra}
    f = f"{OUTD}/{name}_{DEV}_bs{BS}{'_sorted' if os.environ.get('SORT') == '1' else ''}.json"; json.dump(rec, open(f, "w"), indent=1); print("RESULT", json.dumps(rec), flush=True)


class Timer:
    def __init__(self, sync): self.sync = sync
    def __enter__(self): self.sync(); self.t = time.perf_counter(); return self
    def __exit__(self, *a): self.sync(); self.s = time.perf_counter() - self.t
