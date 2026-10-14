"""Nemotron 3.5 ASR streaming 0.6B (SADA fine-tuned), att context [56,13], beam 8 (malsd) as in the paper, plus greedy."""
import os, sys, re, copy, tempfile, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
torch.set_num_threads(THREADS)
from omegaconf import open_dict
import nemo.collections.asr as na
import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as P
for v in vars(P).values():
    if isinstance(v, type) and hasattr(v, "_get_prompt_index_for_cut"):
        v._get_prompt_index_for_cut = lambda self, cut: self._get_prompt_index("ar-AR")
sys.path.insert(0, "src"); from cohere_decode_lib import wer
m = na.models.ASRModel.restore_from("models/nemotron_sada_tel.nemo", map_location=DEV); m.eval()
m.encoder.set_default_att_context_size([56, 13]); base = copy.deepcopy(m.cfg.decoding)
sync = torch.cuda.synchronize if DEV == "cuda" else (lambda: None)
rows, wavs = load_set(); audio = sum(len(w) for w in wavs) / 16000
MODES = [x for x in (("nemotron_ft_greedy", "greedy_batch", 1), ("nemotron_ft_beam8", "malsd_batch", 8))
         if x[0].split("_")[-1] in os.environ.get("MODES", "greedy,beam8").split(",")]
for name, strat, beam in MODES:
    dc = copy.deepcopy(base)
    with open_dict(dc):
        dc.strategy = strat
        if beam > 1: dc.beam.beam_size = beam; dc.beam.return_best_hypothesis = True
    m.change_decoding_strategy(dc, verbose=False)
    tx = lambda ws: m.transcribe(ws, batch_size=BS, target_lang="ar-AR", verbose=False)
    tx(wavs[:BS])
    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
    with Timer(sync) as t: out = tx(wavs)
    out = out[0] if isinstance(out, tuple) else out
    hyp = [re.sub(r"<[^>]+>", " ", o.text if hasattr(o, "text") else str(o)) for o in out]
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else None
    save(name, audio, t.s, {"n": len(rows), "peak_gpu_gib": round(peak, 2) if peak else None, "precision": "float32",
                            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                            "wer": round(100 * np.mean([wer(r["text"], h) for r, h in zip(rows, hyp)]), 2)})
