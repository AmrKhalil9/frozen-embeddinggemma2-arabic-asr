# Gemma 4 speech encoder for Saudi and Gulf Arabic ASR, without changing its weights

Code and public-set results for *How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic
Speech Recognition with a Frozen and an Adapter-Augmented Encoder* (Amr Abdelkhalik, 2026).

The 305M-parameter audio encoder shared by [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and
[google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) (751 of 752 tensors bit-identical; `src/compare_towers.py`) is never
modified. Two systems are built on it:

* **Frozen**: a 22.6M CTC reader over a learned layer mix (`src/train_v2.py`, `src/train_v3.py`).
* **Adapted**: 3.1M LoRA parameters on encoder layers 6-11, a 41.5M hybrid CTC/attention reader with self-conditioned CTC, and joint
  CTC/attention decoding with a sub-word LM (`adapted/train_v5c.py`, `adapted/train_v5d.py`, `adapted/joint_decode.py`,
  `adapted/fast_joint.py` = batched, KV-cached joint search: 1.7x -> 139x real time on one RTX 6000).

**Not released:** in-house call-centre audio, transcripts and results, and trained weights. SADA (CC BY-NC-SA) and Casablanca
(CC BY-NC-ND) are research-only, so models trained on them are research artefacts.

**Scoring:** NFKC, diacritics/tatweel removed, alef/yaa/taa-marbuta/hamza unified, Arabic punctuation removed, mean per-utterance
WER/CER; references that are mostly Latin-script translations are excluded. All numbers below come from `results/public_results.json`.

## Human-labelled public test sets (WER / CER %)

| Test set | Cohere 2B | Nemotron 0.6B (SADA-ft) | Frozen | Adapted, joint + LM |
|---|---|---|---|---|
| Saudi customer-service call (clean) | 19.27 / 10.05 | 24.12 / 8.58 | 29.62 / 11.91 | 19.20 / 7.77 |
| Saudi customer-service call (telephone) | 20.53 / 10.92 | 25.41 / 9.85 | 30.63 / 12.72 | 21.02 / 8.39 |
| Casablanca, Emirati (clean) | 48.35 / 20.35 | 53.27 / 17.09 | 54.63 / 21.26 | 46.98 / 14.89 |
| Casablanca, Emirati (telephone) | 53.15 / 24.38 | 56.56 / 19.81 | 58.82 / 25.38 | 50.01 / 17.40 |
| SaudiTalk long-form (clean) | 56.75 / 29.51 | 61.25 / 25.76 | 65.29 / 29.60 | 58.79 / 25.58 |
| SaudiTalk long-form (telephone) | 55.80 / 28.40 | 62.63 / 27.22 | 66.56 / 31.36 | 60.45 / 26.65 |

## SADA test (WER / CER %)

| System | Clean | Telephone |
|---|---|---|
| Frozen reader | 48.55 / 24.16 | 51.58 / 26.85 |
| Adapted (v5d), CTC + LM | 40.38 / 21.31 | 42.51 / 23.34 |
| Nemotron 0.6B (fine-tuned on SADA) | 46.50 / 21.21 | 48.22 / 22.62 |
| Cohere Transcribe Arabic 2B | 46.85 / 27.59 | 47.88 / 29.00 |

Our readers and Nemotron were trained on SADA train; whether Cohere was is unknown.

## Blind public call-centre sample (DataoceanAI, human-corrected, blind prefill; WER / CER %, clips not drafted by the system or ours)

| System | Clean | Telephone |
|---|---|---|
| GPT-Transcribe | 18.84 / 8.14 | 19.21 / 9.13 |
| Chirp 3 | 26.50 / 9.78 | 25.70 / 10.12 |
| MAI-Transcribe-2 | 27.28 / 9.12 | 24.32 / 9.11 |
| **Adapted (ours)** | 29.89 / 11.30 | 31.51 / 11.92 |
| Grok STT 1.0 | 31.23 / 17.12 | 31.94 / 14.25 |
| Gemini 3.5 Transcribe | 31.33 / 13.05 | 29.73 / 11.62 |
| AssemblyAI Universal-3.5 Pro | 32.15 / 12.83 | 32.80 / 13.89 |
| Cohere Transcribe Arabic 2B | 35.06 / 12.11 | 33.07 / 14.63 |
| Deepgram Nova-3 | 36.03 / 13.45 | 36.97 / 16.22 |
| Voxtral Mini Transcribe | 38.29 / 16.64 | 39.16 / 17.91 |
| GPT-4o Transcribe | 38.65 / 27.63 | 44.83 / 30.54 |
| Fish Audio Transcribe 1 Pro | 39.50 / 21.11 | 38.61 / 21.09 |
| ElevenLabs Scribe v2 | 39.71 / 15.02 | 39.38 / 17.39 |
| Voxtral Small 24B | 39.74 / 16.71 | 44.86 / 20.53 |
| Whisper large-v3 | 44.35 / 22.84 | 46.74 / 22.57 |

Small sample (92 clips, 8.3 min) and one annotator: read as a ranking, see the paper for intervals.
List prices with sources and dates: `results/prices.json`.

## Layout

| Path | Contents |
|---|---|
| `src/` | frozen system: training, LM building, evaluation, telephone simulation, Cohere reference run, tower comparison |
| `adapted/` | adapted system: LoRA + hybrid reader training, joint CTC/attention search (reference and fast batched versions) |
| `speed/` | frozen-system speed benchmark |
| `results/` | every public-set number in the paper, and the sourced price list |

Run scripts from the repository root; paths are relative to it.

## Licence

Code: MIT. Models and datasets keep their own licences.
