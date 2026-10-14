"""Build 4-gram word LMs for CTC beam search. Test (gold) calls are excluded from every corpus.
lm_calls      : call-centre transcripts only (Cohere + second commercial recogniser labels)        -> commercially safe
lm_calls_sada : calls + SADA22 train transcripts (CC BY-NC-SA, research only)  -> paper experiments only
"""
import json, os, re, subprocess, unicodedata

D = "lm"; os.makedirs(D, exist_ok=True)
LMPLZ, BIN = "kenlm/build/bin/lmplz", "kenlm/build/bin/build_binary"


def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("ـ", "")
    for d in range(0x064B, 0x0653): s = s.replace(chr(d), "")
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
         .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي").replace("ء", ""))
    s = re.sub(r"(.)\1{2,}", r"\1\1", s); s = re.sub(r"[^؀-ۿ0-9\s]", " ", s)
    return " ".join(s.split())


call = lambda p: os.path.basename(p).split("_w")[0]
gold_calls = {call(g["audio_filepath"]) for g in json.load(open("data/calls/test_windows.json"))}
calls = []
for f, key in [("data/calls/train_windows_cohere.json", "cohere"), ("data/calls/train_windows_asr2.json", "asr2")]:
    for r in json.load(open(f)):
        if call(r["audio_filepath"]) not in gold_calls and r.get(key):
            t = norm(r[key])
            if t: calls.append(t)
sada = []
for l in open("data/sada/train_text.json"):
    if l.strip():
        t = norm(json.loads(l)["text"])
        if t: sada.append(t)
open(f"{D}/calls.txt", "w").write("\n".join(calls) + "\n")
open(f"{D}/calls_sada.txt", "w").write("\n".join(calls + sada) + "\n")
print(f"call lines {len(calls)} words {sum(len(x.split()) for x in calls)} | SADA lines {len(sada)} words {sum(len(x.split()) for x in sada)}")
for name in ["calls", "calls_sada"]:
    subprocess.run(f"{LMPLZ} -o 4 --discount_fallback -S 20% -T /tmp < {D}/{name}.txt > {D}/lm_{name}.arpa 2> {D}/lm_{name}.log", shell=True, check=True)
    subprocess.run(f"{BIN} {D}/lm_{name}.arpa {D}/lm_{name}.bin > /dev/null 2>&1", shell=True, check=True)
    words = sorted({w for line in open(f"{D}/{name}.txt") for w in line.split()})
    open(f"{D}/unigrams_{name}.txt", "w").write("\n".join(words))
    print(f"lm_{name}.bin: {os.path.getsize(f'{D}/lm_{name}.bin') / 1e6:.1f} MB, vocab {len(words)} words")
print("LM_DONE")
