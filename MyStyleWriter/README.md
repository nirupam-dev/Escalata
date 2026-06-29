# ✍️ MyStyle Writer — Style Fine-Tuning with LoRA

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-00B4D8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Colab-T4_GPU-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **Project #27** — AI/ML Portfolio Project  
> Fine-tune **Qwen2.5-3B-Instruct** using **LoRA + QLoRA 4-bit quantization** on the **databricks-dolly-15k** dataset, then compare base vs fine-tuned outputs with a professional dark-themed Gradio web UI featuring live metrics (Perplexity, BLEU, Sentence Length, Vocabulary Diversity).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run-step-by-step)
- [Metrics Explanation](#-metrics-explanation)
- [Sample Output](#-sample-output-comparison)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Overview

**MyStyle Writer** demonstrates production-grade LoRA fine-tuning on a 3B-parameter language model. The project uses the **databricks/databricks-dolly-15k** dataset in Alpaca instruction format to teach the model a specific writing style, then provides comprehensive evaluation metrics and an interactive comparison UI.

Key highlights:
- **Parameter-efficient**: Only trains ~0.5% of model parameters via LoRA (r=16, α=32)
- **Memory-efficient**: 4-bit QLoRA quantization fits in Colab's free T4 GPU (15GB VRAM)
- **Anti-overfitting**: SHA256 deduplication before splitting + EarlyStopping callback
- **Full pipeline**: Data prep → Training → Evaluation → Interactive UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MyStyle Writer Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Raw Text │───>│ dataset_prep │───>│ HuggingFace Dataset│    │
│  │ (.txt)   │    │    .py       │    │ (Arrow format)     │    │
│  └──────────┘    └──────────────┘    └────────┬───────────┘    │
│                                               │                 │
│                                               ▼                 │
│  ┌──────────────┐    ┌──────────┐    ┌────────────────────┐    │
│  │ Qwen2.5-3B   │───>│ train.py │───>│  LoRA Adapter      │    │
│  │ (Base Model)  │    │ (SFT +   │    │  (best_adapter/)   │    │
│  │              │    │  LoRA)   │    │  + loss_curve.png  │    │
│  └──────────────┘    └──────────┘    └───┬────────┬───────┘    │
│                                          │        │             │
│                                          ▼        ▼             │
│                                  ┌───────────┐ ┌─────────┐     │
│                                  │evaluate.py│ │ app.py  │     │
│                                  │(Metrics + │ │(Gradio  │     │
│                                  │ Report)   │ │  UI)    │     │
│                                  └───────────┘ └─────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦙 **Qwen2.5-3B-Instruct** | Strong 3B parameter model from HuggingFace |
| 🔧 **LoRA Fine-tuning** | Only trains ~0.5% of parameters (r=16, alpha=32, q_proj+v_proj) |
| ⚡ **QLoRA 4-bit** | NF4 quantization + double quant fits in T4 GPU |
| 📊 **4 Metrics** | Perplexity, BLEU, Avg Sentence Length, Vocab Diversity |
| 🎨 **Dark Gradio UI** | Professional interface with charts, badges, and style scoring |
| 🛡️ **Anti-Leakage** | SHA256 dedup before split + EarlyStopping callback |
| 📈 **Loss Curves** | Auto-generated training & validation loss plots |
| 🆓 **No Paid APIs** | Everything runs on free Colab T4 GPU |

---

## 📁 Folder Structure

```
MyStyleWriter/
│
├── dataset_prep.py        ← Script 1: Raw text → Alpaca format → Arrow dataset
├── train.py               ← Script 2: LoRA fine-tuning with SFTTrainer
├── evaluate.py            ← Script 3: Base vs fine-tuned model evaluation
├── app.py                 ← Script 4: Professional dark Gradio web UI
├── requirements.txt       ← Script 5: All dependencies with versions
├── README.md              ← This file
│
├── data/
│   ├── raw_text.txt       ← User provides writing samples here
│   └── prepared/          ← Created by dataset_prep.py (Arrow format)
│       ├── train/
│       └── validation/
│
└── best_adapter/          ← Created by train.py (LoRA weights)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── training_history.csv
    ├── loss_curve.png
    └── tokenizer files...
```

---

## 🛠️ Setup & Installation

### Option A: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → T4 GPU**
3. Upload the project files or clone from GitHub
4. Install dependencies:

```python
!pip install peft trl bitsandbytes accelerate gradio transformers datasets evaluate nltk
```

### Option B: Local Machine

```bash
# Clone or download the project
git clone https://github.com/YOUR_USERNAME/MyStyleWriter.git
cd MyStyleWriter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run (Step by Step)

### Step 1: Prepare Your Dataset (Optional)

```bash
python dataset_prep.py
```

> **Note:** `train.py` downloads `databricks/databricks-dolly-15k` automatically from HuggingFace Hub. Run `dataset_prep.py` only if you want to prepare your own custom text from `data/raw_text.txt`.

**What it does:**
- Loads `./data/raw_text.txt`
- Cleans text (removes special chars, short lines)
- Converts to Alpaca instruction format
- Deduplicates with SHA256
- Splits 80/20 train/validation
- Saves to `./data/prepared/`

---

### Step 2: Train with LoRA (Run on Colab T4)

```bash
python train.py
```

**What it does:**
- Downloads Qwen2.5-3B-Instruct + Dolly-15k from HuggingFace
- Applies LoRA adapters (r=16, alpha=32, q_proj + v_proj)
- Trains with SFTTrainer + EarlyStopping
- Saves adapter to `best_adapter/`
- Generates `training_history.csv` and `loss_curve.png`

⏱️ **Training time:** ~30-60 minutes on T4 GPU

---

### Step 3: Evaluate Both Models

```bash
python evaluate.py
```

**What it does:**
- Loads base model + fine-tuned model (with adapter)
- Runs 10 test prompts through both
- Calculates Perplexity, BLEU, Avg Sentence Length, Vocab Diversity
- Prints formatted comparison table
- Saves `evaluation_report.txt`

---

### Step 4: Launch the Gradio UI

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

**UI Features:**
- ✍️ Dark professional theme with gradient accents
- 📄 Side-by-side Base vs Fine-tuned output panels
- 📊 4 live metric badges (PPL, BLEU, AvgLen, VocabDiv)
- 🎯 Style Shift Score progress bar
- 📈 Loss curve image display
- ⚡ Example prompts (click to try)

---

## 📏 Metrics Explanation

| Metric | What It Measures | Better Direction |
|--------|-----------------|-----------------|
| **Perplexity** | How confident/fluent the model is about the text | ↓ Lower is better |
| **BLEU Score** | N-gram overlap similarity to reference style | ↑ Higher is better |
| **Avg Sentence Length** | Average words per sentence | ↓ Shorter = more concise |
| **Vocabulary Diversity** | Unique words / total words ratio | Context-dependent |

---

## 📸 Sample Output Comparison

### Prompt: *"The soldier walked into the battlefield."*

| Model | Output (sample) |
|-------|----------------|
| **Base Qwen2.5** | The soldier walked into the battlefield, his heart pounding with a mixture of fear and determination as the sounds of explosions echoed across the scarred landscape... |
| **Fine-tuned** | The soldier walked into the battlefield. It was quiet. The smoke hung low over the ground. He moved forward. He did not look back. |

### Average Metrics

| Metric | Base Model | Fine-tuned | Winner |
|--------|-----------|------------|--------|
| Perplexity ↓ | ~45.0 | ~28.0 | Fine-tuned ✅ |
| BLEU Score ↑ | ~0.012 | ~0.085 | Fine-tuned ✅ |
| Avg Sent Len ↓ | ~18.4 | ~7.2 | Fine-tuned ✅ |
| Vocab Diversity | ~72% | ~61% | — |

*Note: Actual results vary based on training duration and data.*

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Qwen/Qwen2.5-3B-Instruct** | Base language model (3B params) |
| **databricks/databricks-dolly-15k** | Training dataset (Alpaca instruction format) |
| **LoRA / PEFT** | Parameter-efficient fine-tuning (r=16, α=32) |
| **QLoRA (BitsAndBytes)** | 4-bit NF4 quantization for memory efficiency |
| **TRL SFTTrainer** | Supervised fine-tuning with validation |
| **HuggingFace Datasets** | Data loading, processing, Arrow format |
| **Gradio** | Professional dark-themed web UI |
| **PyTorch** | Deep learning backend |
| **Matplotlib** | Training loss curves and metric charts |
| **Google Colab (T4)** | Free GPU for training and inference |

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ as an AI/ML portfolio project</strong><br>
  <em>Project #27 — MyStyle Writer: Style Fine-Tuning with LoRA</em>
</p>
