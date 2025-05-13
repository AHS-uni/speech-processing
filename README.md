# Hybrid Transformer ASR with CTC & Seq2Seq on LJSpeech

## Project Overview

This project implements an **automatic speech‑recognition (ASR)** system that couples a **shared Transformer encoder** with **two task‑specific heads**:

1. **Connectionist Temporal Classification (CTC) Head** – frame‑level, alignment‑free speech recognition.
2. **Autoregressive Decoder Head** – sequence‑to‑sequence Transformer decoder conditioned on the encoder output.

The dual‑headed design allows us to train the model in *CTC‑only*, *Seq2Seq‑only*, or *multi‑task* modes, combining the complementary strengths of both objectives.

## Dataset

* **LJSpeech 1.1** – 13,100 English audio clips (≈24 h) of a single female speaker reading public‑domain texts.
* Each clip is paired with a normalized text transcription.
* Download: [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/) (≈2.5 GB).

Run the following Command to download the dataset

``` shell
mkdir -p /content/data/LJSpeech \
&& wget -q https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 \
       -O /content/data/LJSpeech/LJSpeech-1.1.tar.bz2 \
  && tar -xjf /content/data/LJSpeech/LJSpeech-1.1.tar.bz2 \
           -C /content/data/LJSpeech \
  && echo "✅ Downloaded and extracted to /content/data/LJSpeech"
```

## Repository Layout

```
.
├── notebooks/                # Executable, interactive steps
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training.ipynb
│   ├── 04_inference_and_decoding.ipynb
│   ├── 05_experiments_and_ablation.ipynb
│   └── 06_demo.ipynb
├── scripts/                  # Non‑notebook code
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── config.yaml
├── utils/                    # Helper modules (metrics, text processing, etc.)
├── data/                     # Pre‑processed features, tokenizer files
└── README.md                 # (this file)
```

## Requirements

### Software

* Python ≥ 3.10
* **PyTorch** ≥ 2.2 (+ CUDA 11.8 on GPU).
* `torchaudio`, `sentencepiece`, `jiwer`, `PyYAML`, `pandas`, `matplotlib`.
* (Optional) **PyTorch Lightning** and **Weights & Biases** for streamlined training / logging.

Install core dependencies with:

```bash
pip install torch torchaudio sentencepiece jiwer pyyaml pandas matplotlib
```

### Hardware (minimum viable)

| Component | Minimum                   | Recommended                             |
| --------- | ------------------------- | --------------------------------------- |
| GPU       | 8 GB VRAM (e.g. RTX 3060) | ≥16 GB VRAM (RTX 3080, A40)             |
| RAM       | 16 GB                     | ≥32 GB                                  |
| Disk      | 30 GB                     | ≥50 GB (dataset, features, checkpoints) |

Kaggle provides A100 40 GB GPUs for 30 h/week, which exceed the recommended spec.

## Deliverables (Milestone 1)

1. **Pre‑processing pipeline** that converts raw LJSpeech audio to log‑Mel spectrograms + tokenized text.
2. **PyTorch implementation** of the shared Transformer encoder, CTC head, and decoder head.
3. **Training notebook** (03\_training.ipynb) demonstrating:

* CTC‑only training run.
* Multi‑task joint run with adjustable loss weights.
4. **Inference notebook** (04\_inference\_and\_decoding.ipynb) evaluating WER & CER on a held‑out set.
5. **Demo notebook** (06\_demo.ipynb) loading a pretrained checkpoint and transcribing sample audio.

> **Note:** Further polish (hyper‑parameter sweeps, SpecAugment, Conformer encoder swap, etc.) will be added in later iterations.

## Getting Started (quick test)

```bash
# 1. Clone or download the repo on Kaggle / local
# 2. Install deps (see above)
# 3. Run data prep notebook and cache features
# 4. Kick off a small CTC‑only training run:
python scripts/train.py --config scripts/config.yaml --ctc_only true
```

## License

TBD – likely MIT.

## Contributors

* **Your Name** – project lead
* **Team Member A** – legacy code migration
* **Team Member B** – data engineering

---

*Last updated: 2025‑05‑12*
