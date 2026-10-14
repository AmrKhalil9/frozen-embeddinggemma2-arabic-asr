# Frozen Gemma 4 / EmbeddingGemma 2 speech encoder for Saudi and Gulf Arabic ASR

Code and public-set results for the paper *How Far Can a Frozen Gemma 4 Speech Encoder Go? Saudi and Gulf Arabic Speech Recognition
with the Encoder Shared by Gemma 4 and EmbeddingGemma 2* (Amr Abdelkhalik, 2026).

The audio encoder of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (305M parameters) is kept frozen. It is
bit-identical to the audio encoder of [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B): 751 of 752 tensors match exactly, and the
only difference is the projection into each model's text embedding space (`src/compare_towers.py` reproduces the check). A
22.6M-parameter reader learns a softmax-weighted mix of encoder layers 6, 8, 10, 11, 12 and the output, then applies a depth-wise
convolution, 6 transformer layers and a CTC output over 68 Arabic characters. A word 4-gram KenLM model is used in pyctcdecode beam search.

**What is not here.** The in-house call-centre recordings, their transcripts, the call-centre results and the trained reader weights
are not released. The training scripts expect your own call windows under `data/calls/`. SADA (CC BY-NC-SA) and Casablanca
(CC BY-NC-ND) are research-only datasets, so anything trained on them is for research use only.

## Layout

| Path | Contents |
|---|---|
| `src/train_v2.py` | Reader trained on call windows (two machine labels per window, one sampled per epoch) |
| `src/train_v3.py` | Continued training on calls + SADA, with `MODE=tel` (G.711 µ-law on the fly) or `MODE=wide` |
| `src/telephony.py` | Deterministic 16 kHz → 8 kHz → G.711 µ-law → 16 kHz degradation |
| `src/build_lm.py`, `src/eval_lm.py` | 4-gram LMs; LM weight and word bonus tuned on validation calls only |
| `src/eval_sada.py`, `src/eval_research.py` | Scoring on SADA test and the public Saudi/Gulf sets |
| `src/build_research_eval.py` | Builds the evaluation manifests and their telephone-coded versions |
| `src/cohere_research.py`, `src/cohere_decode_lib.py` | Cohere Transcribe Arabic reference run, plus the shared normalisation and WER/CER code |
| `speed/` | Speed benchmark: same audio for every system, length-sorted batches, warm-up excluded |
| `results/public_results.json` | All public-set scores and speed measurements |

The scoring normalisation is NFKC, then removal of diacritics and tatweel, then unification of alef, alef maqsura, taa marbuta and
hamza carriers. WER and CER are the mean over utterances; CER ignores spaces.

## Results on public test sets (WER / CER %, telephone-trained reader + 4-gram LM)

| Test set | Cohere Transcribe Arabic 2B | Frozen EG2 + reader |
|---|---|---|
| Saudi customer-service sample (clean) | 27.77 / 11.91 | 40.11 / 14.91 |
| Saudi customer-service sample (telephone) | 30.11 / 13.19 | 41.16 / 15.73 |
| Casablanca, Emirati (clean) | 54.77 / 22.46 | 60.50 / 23.64 |
| Casablanca, Emirati (telephone) | 59.25 / 26.49 | 64.11 / 27.66 |
| SaudiTalk long-form (clean) | 58.49 / 30.07 | 65.46 / 29.66 |
| SaudiTalk long-form (telephone) | – | 66.72 / 31.41 |

### SADA test (4,972 utterances), WER / CER % with LM

| Reader | Clean 16 kHz | Telephone 8 kHz |
|---|---|---|
| v2 | 59.85 / 32.44 | 63.56 / 36.12 |
| v3_tel | 48.54 / 24.16 | 51.57 / 26.85 |
| v3_wide | 48.35 / 24.40 | 52.24 / 28.14 |

`v2` = calls only; `v3_tel` = + SADA degraded to telephone quality; `v3_wide` = + clean SADA.

## Speed (RTFx = seconds of audio per second of wall time)

GPU: one Quadro RTX 6000, all 813 Casablanca Emirati test utterances. CPU: 16 threads of a Xeon Gold 6230, 200 utterances.

| System | GPU RTFx | GPU memory (GiB) | CPU RTFx |
|---|---|---|---|
| Frozen EG2 encoder + reader (greedy) | 257.2 | 1.49 | 31.1 |
| Cohere Transcribe Arabic 2B, batch 16 | 157.4 | 4.76 | 9.2 |
| Nemotron 3.5 ASR 0.6B fine-tuned (greedy) | – | – | 22.6 |

On its own, the frozen encoder runs at 348× real time on the GPU. Nemotron's batched GPU inference failed in its toolkit on the full set, so it was timed on CPU only.

## Licence

The code is MIT-licensed. Models and datasets keep their own licences.
Run every script from the repository root (`python src/train_v3.py`, `python speed/bench_eg2.py`); paths are relative to it.
