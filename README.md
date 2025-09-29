

# SomaliWhisper — README

**Project:** SomaliWhisper
**Purpose:** Evaluate and fine-tune OpenAI’s Whisper (small) for Somali automatic speech recognition (ASR) using LoRA parameter-efficient fine-tuning. This repository/notebook contains baseline evaluation, EDA on Somali datasets, LoRA fine-tuning, and evaluation of the fine-tuned model.



## Table of Contents

* [Summary / Highlights](#summary--highlights)
* [What’s included](#whats-included)
* [Environment & installation](#environment--installation)
* [Datasets used](#datasets-used)
* [Exploratory Data Analysis (EDA) — key stats](#exploratory-data-analysis-eda---key-stats)
* [Baseline (untuned) evaluation — results](#baseline-untuned-evaluation---results)
* [LoRA fine-tuning — training summary](#lora-fine-tuning---training-summary)
* [LoRA evaluation — results](#lora-evaluation---results)
* [Comparison & interpretation](#comparison--interpretation)
* [Caveats, warnings & known issues](#caveats-warnings--known-issues)
* [Reproducibility / How to run (quick start)](#reproducibility--how-to-run-quick-start)
* [Recommended next steps](#recommended-next-steps)
* [Files / Notebook organization](#files--notebook-organization)
* [Acknowledgements & license](#acknowledgements--license)

---

## Summary / Highlights

* Performed baseline evaluation of `openai/whisper-small` on two Somali datasets from Hugging Face.
* Performed EDA across local Somali TTS/ASR manifests (text and audio summaries).
* Trained a LoRA adapter on top of `whisper-small` (parameter-efficient training).
* After LoRA fine-tuning, the model showed **substantial improvements** on the small `adityaedy01/somali-voice` subset and measurable improvements in CER for `nurfarah57/somali_asr`. Results are based on the actual runs and logs below.
* Important caution: some evaluation splits are extremely small (2–3 samples), so metrics should be treated as illustrative, not definitive.

---

## What’s included

* `SomaliWhisper.ipynb` — main Colab notebook covering:

  * Dependencies installation
  * Baseline evaluation (Whisper-small)
  * Optimized EDA
  * LoRA fine-tuning (PEFT)
  * Evaluation of LoRA-finetuned model
* Helper scripts/snippets for inference and evaluation (embedded in notebook).
* CSV manifests (expected) and audio stored on Google Drive (project assumes a Drive mount for training artifacts).

---

## Environment & installation

Recommended: Google Colab (GPU). The notebook uses GPU if available.

Install dependencies (example Colab cell):

```bash
# basic
!pip install --upgrade torch torchaudio

# HF stack
!pip install transformers datasets evaluate jiwer

# LoRA / PEFT training
!pip install peft accelerate

# misc
!pip install sentencepiece soundfile librosa tqdm pandas numpy torchcodec
```

---

## Datasets used

* `adityaedy01/somali-voice` (Hugging Face)
* `nurfarah57/somali_asr` (Hugging Face)
* Local manifests (used for fine-tuning) e.g.:

  * `somali_tts_dataset.csv` (TTS-style manifest)
  * `soomali_asr_dataset_shortened.csv` (ASR manifest)

*Note:* The two Hugging Face datasets use different reference text column names:

* `adityaedy01/somali-voice` → `sentence`
* `nurfarah57/somali_asr` → `transcription`

The notebook handles the column mismatch.

---

## Exploratory Data Analysis (EDA) — key stats

EDA was run on local manifests (sampled for audio metadata to avoid long read times). The sample sizes shown reflect the dataset manifests and subsampling used by the EDA code.

### `somali_tts` (local TTS manifest)

* **Total samples:** 1719
* **Text length (characters):**

  * mean = **58.64**
  * std = **31.13**
  * min = 3, 25% = 39, median = 53, 75% = 69, max = 175
* **Audio metadata (200 sampled files):**

  * mean duration = **4.395 s**
  * std = 2.026 s
  * min = 0.709 s, max = 11.373 s
  * sample rate (sampled files): **44100 Hz**

*Examples (random sample):*

* `"Adeegsiga_saabuunta_wanaagsan_waxay_caawisaa_fayodhowrk"`
* `"Fadlan su’aasha ku celi"`
* `"Ha ilaawin inaad maareeyso waqtigaaga"`

---

### `soomali_asr` (local ASR manifest)

* **Total samples:** 4999
* **Text length (characters):**

  * mean = **14.01**
  * std = **10.33**
  * min = 2, 25% = 7, median = 11, 75% = 18, max = 218
* **Audio metadata (200 sampled files):**

  * mean duration = **1.303 s**
  * std = 0.782 s
  * min = 0.438 s, max = 4.975 s
  * sample rate (sampled files): **16000 Hz**

*Examples (random sample):* `"ishiisii"`, `"niman"`, `"Waa la dhisay."`, `"liin"`, `"tuugada"`

---

## Baseline (untuned) evaluation — results

**Model:** `openai/whisper-small` (untuned)
**Language hint:** `language='so'` (Somali), using `pipeline("automatic-speech-recognition")` with `chunk_length_s=30` (note experimental warning).

### Per-dataset results (baseline)

|                    Dataset | Samples evaluated |              WER |              CER |
| -------------------------: | ----------------: | ---------------: | ---------------: |
| `adityaedy01/somali-voice` |                 2 | **1.0909090909** | **0.9666666667** |
|    `nurfarah57/somali_asr` |                 3 |          **1.0** | **0.6981132075** |

**Representative transcriptions (baseline)**

* `adityaedy01/somali-voice`:

  * REF: `geed gaab iyo geed qodaxeed` → HYP: `get up, your gate called ahead.` (English/garbled)
  * REF: `laba iyo lixdan laxaad oo dhalaya` → HYP: `لَبَئِي اللِّهِدَنُ الْحَادَ وَدَرَيَّ` (non-Latin script noise)
* `nurfarah57/somali_asr` examples show similarly mis-recognized outputs (Arabic script fragments / transliteration errors).

**Interpretation:** baseline `whisper-small` produced many non-Somali tokens or translations due to language detection/translation behavior and tokenization mismatch; WER > 1 indicates many insertions or severe misalignment.

---

## LoRA fine-tuning — training summary

**Approach:** Freeze base model weights and train LoRA adapters (targeting attention projections). Training via `Seq2SeqTrainer` with FP16 and gradient accumulation.

**LoRA configuration (selected):**

* `r = 16`, `lora_alpha = 32`
* `target_modules = ["q_proj", "v_proj"]`
* `lora_dropout = 0.1`
* `task_type = "SEQ_2_SEQ_LM"`

**Training run (selected log lines):**

* Several per-step losses printed during training (examples included in notebook).
* Final training metrics:

  * `train_runtime = 4888.1344 s` (~81.5 minutes)
  * `train_samples_per_second = 5.497`
  * `train_steps_per_second = 1.375`
  * `train_loss = 3.2927211765732083`
  * `epoch = 4.0`

**Parameter efficiency:**

* Trainable params: **1,769,472**
* Total params: **243,504,384**
* Fraction trainable: **0.73%**

---

## LoRA evaluation — results

**Model:** `whisper-small` with LoRA adapters loaded (from `whisper-lora-trained`)
**Same evaluation harness** as baseline.

### Per-dataset results (LoRA-finetuned)

|                    Dataset | Samples evaluated |              WER |              CER |
| -------------------------: | ----------------: | ---------------: | ---------------: |
| `adityaedy01/somali-voice` |                 2 | **0.1818181818** | **0.0333333333** |
|    `nurfarah57/somali_asr` |                 3 | **0.7777777778** | **0.2452830189** |

**Representative transcriptions (LoRA)**

* `adityaedy01/somali-voice`:

  * REF: `geed gaab iyo geed qodaxeed` → HYP: `geed qaab iyo geed qodaheed` (small orthographic errors)
  * REF: `laba iyo lixdan laxaad oo dhalaya` → HYP: `laba iyo lixdan laxaad oo dhalaya` (correct)
* `nurfarah57/somali_asr` examples improved CER and partial word recognition, but some outputs still include non-Latin tokens or minor orthographic errors.

---

## Comparison & interpretation

**Absolute improvements (baseline → LoRA):**

* `adityaedy01/somali-voice` (2 samples):

  * WER: **1.0909 → 0.1818** (∆ = −0.9091, ≈ **83.4% relative improvement**)
  * CER: **0.9667 → 0.0333** (∆ = −0.9333, ≈ **96.6% relative improvement**)
  * Interpretation: dramatic improvement on these two examples — transcription becomes Somali-orthographic and corrects language detection issues.

* `nurfarah57/somali_asr` (3 samples):

  * WER: **1.0 → 0.7778** (∆ = −0.2222, ≈ **22.2% relative improvement**)
  * CER: **0.6981 → 0.2453** (∆ = −0.4528, ≈ **64.9% relative improvement**)
  * Interpretation: LoRA reduced character errors significantly while word-level errors remain moderate; sample-level variability suggests more tuning/data may further improve WER.

**Important caveats on interpretation:**

* **Tiny evaluation sizes (2 and 3 samples)** for those HF datasets — extreme caution required. Metrics are noisy for such tiny sample counts. Use larger held-out sets for reliable estimates.
* Baseline produced outputs in English/Arabic script for some examples (language detection issue). LoRA reduced this behavior on the small set tested.
* Some warnings in logs (see next section) point to attention/tokenization configuration mismatches that may affect evaluation fairness.

---

## Caveats, warnings & known issues (from your runs)

* `chunk_length_s` with seq2seq pipelines is **experimental**. Pipeline chunking can lead to inaccuracies on long-form audio; using the model’s `generate()` manually is preferable for long audio (Whisper original paper uses its own chunking).
* **Forced decoder ids deprecated warning**: the library warns that `forced_decoder_ids` is deprecated in favor of the `task` and `language` config options; behavior may change in future HF releases.
* **PEFT warning**: `Already found a peft_config attribute in the model. This will lead to having multiple adapters in the model.` — indicates repeated wrapping or loading adapters; ensure you load PEFT only once and clear state if reloading.
* **Attention mask warning:** `The attention mask is not set and cannot be inferred from input because pad token is same as eos token.` — set `attention_mask` explicitly if you see unexpected behavior.
* **WER > 1** in baseline indicates a lot of insertions — this is possible with small reference lengths and algorithmic details; inspect outputs manually.
* **Different sample rates** across datasets: `somali_tts` sampled files were 44.1 kHz, `soomali_asr` sampled files 16 kHz. Ensure consistent resampling to 16 kHz for Whisper inputs (the scripts do resample where needed).
* **Evaluation normalization**: tokenization, punctuation, casing, and diacritics can heavily influence WER/CER. Consistent normalization (lowercasing, removing punctuation or diacritics depending on goal) is recommended for fair comparisons.

---

## Reproducibility / How to run (quick start)

1. **Mount Google Drive** (if running on Colab) and set paths to manifests and audio folders.

2. **Install dependencies** (run once):

```bash
!pip install --upgrade torch torchaudio
!pip install transformers datasets evaluate jiwer peft accelerate sentencepiece
!pip install soundfile librosa tqdm pandas numpy torchcodec
```

3. **Baseline evaluation** (cell):

* Set `MODEL_NAME = "openai/whisper-small"` and run the baseline evaluation cell included in the notebook. It will:

  * Load processor/model
  * Create pipeline(`automatic-speech-recognition`)
  * Load HF datasets by id
  * Evaluate and print WER/CER + example transcriptions

4. **EDA**: run the optimized EDA cell (samples audio metadata only by default). Adjust `max_samples` if you want more files analyzed.

5. **LoRA training**:

* Ensure `AUDIO_DIR`, `TTS_MANIFEST`, and `ASR_MANIFEST` point to your files in Drive.
* Configure `LoraConfig`, `Seq2SeqTrainingArguments` as needed (batch sizes, epochs).
* Run training cell. The notebook saves adapter + processor in `OUTPUT_DIR`.

6. **LoRA evaluation**:

* Set `MODEL_DIR` to `OUTPUT_DIR` (where LoRA weights & processor saved).
* Run the LoRA evaluation cell (uses `PeftModel.from_pretrained(...)` to load adapters).
* Prints WER/CER and sample transcriptions.

7. **Generate/Save outputs**: you may add small modifications to write predictions and references to CSV for later inspection.

---

## Recommended next steps

1. **Larger evaluation splits** — increase held-out test size to get statistically meaningful metrics (100s–1k examples). The current per-dataset sample sizes used for HF datasets (2 and 3) are too small to draw robust conclusions.
2. **Normalization** — implement consistent normalization on references and hypotheses (lowercase, unify punctuation/diacritics) before computing WER/CER.
3. **Attention mask fix** — ensure `attention_mask` is passed where relevant to avoid the warning and potential decode anomalies.
4. **Merge LoRA weights** (optional) — once satisfied, merge LoRA adapter into base model for simpler inference (`peft` provides utilities).
5. **Compare against base model in a single script** — run side-by-side comparisons in one table (baseline vs LoRA).
6. **Tune generation hyperparameters** — `num_beams`, `no_repeat_ngram_size`, `length_penalty`, and `max_length` can affect WER/CER; run a small grid search.
7. **More LoRA targets / data augmentation** — try including more layers (or different `target_modules`), augment data, or synthesize Somali audio for low-resource gains.

---

## Files / Notebook organization

* `SomaliWhisper.ipynb` — Colab notebook (core pipeline)
* `README.md` — this file (documentation & report)
* `datasets/` — CSV manifests and local audio (expected Google Drive mount)
* `whisper-lora-trained/` — output folder for PEFT adapters & processor (after training)

---

## Acknowledgements & license

* Datasets used are from Hugging Face dataset authors:

  * `adityaedy01/somali-voice`
  * `nurfarah57/somali_asr`
* Whisper model: OpenAI / Hugging Face (`openai/whisper-small`)
* PEFT/LoRA: Hugging Face PEFT
* This code and notebook are for research/educational use. Datasets retain their original licensing — check the HF pages for usage terms.

---


