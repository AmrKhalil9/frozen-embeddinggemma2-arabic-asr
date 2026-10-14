# -*- coding: utf-8 -*-
import json, os, re, sys, time, unicodedata
import numpy as np, soundfile as sf, torch
REF = "data/calls/test_windows.json"
def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("ـ", "")
    for d in range(0x064B, 0x0653): s = s.replace(chr(d), "")
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ٱ","ا")
           .replace("ى","ي").replace("ة","ه").replace("ؤ","و").replace("ئ","ي").replace("ء",""))
    s = re.sub(r"(.)\1{2,}", r"\1\1", s); s = re.sub(r"[^؀-ۿ0-9\s]", " ", s)
    return " ".join(s.split())
def _lev(r,h):
    d=list(range(len(h)+1))
    for i in range(1,len(r)+1):
        prev,d[0]=d[0],i
        for j in range(1,len(h)+1):
            cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(r[i-1]!=h[j-1])); prev=cur
    return d[len(h)]
def wer(r,h): a,b=norm(r).split(),norm(h).split(); return 0.0 if not a else _lev(a,b)/len(a)
def cer(r,h): a,b=list(norm(r).replace(" ","")),list(norm(h).replace(" ","")); return 0.0 if not a else _lev(a,b)/len(a)


