# Hybrid Transformer ASR with CTC & Seq2Seq on LJSpeech

## Project Overview

This project implements an **automatic speech‑recognition (ASR)** system that couples a **shared Transformer encoder** with **two task‑specific heads**:

1. **Connectionist Temporal Classification (CTC) Head** – frame‑level, alignment‑free speech recognition.
2. **Autoregressive Decoder Head** – sequence‑to‑sequence Transformer decoder conditioned on the encoder output.

## Dataset

* **LJSpeech 1.1** – 13,100 English audio clips (≈24 h) of a single female speaker reading public‑domain texts.
* Each clip is paired with a normalized text transcription.
* Download: [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/) (≈2.5 GB).

Run the following Command to download the dataset

``` shell
mkdir -p data/LJSpeech \
&& wget -q https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 \
       -O data/LJSpeech/LJSpeech-1.1.tar.bz2 \
  && tar -xjf  data/LJSpeech/LJSpeech-1.1.tar.bz2 \
           -C data/LJSpeech
```


## Requirements

* Python ≥ 3.10
* **PyTorch** ≥ 2.2 (+ CUDA 11.8 on GPU).
* `torchaudio`, `sentencepiece`, `jiwer`, `PyYAML`, `pandas`, `matplotlib`.

Install core dependencies with:

```bash
pip install torch torchaudio sentencepiece jiwer pyyaml pandas matplotlib
```
