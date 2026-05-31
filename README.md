<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=120&section=header&text=AutoLLM%20Forge&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Efficient%20LLM%20Fine-Tuning%20Platform%20with%20QLoRA&descAlignY=60&descSize=15&descColor=a5b4fc" width="100%"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-1000%2B%20Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![VRAM](https://img.shields.io/badge/VRAM%20Reduction-75%25-14b8a6?style=flat-square)]()

</div>

---

## Overview

AutoLLM Forge is a production-grade fine-tuning platform for large language models from 2B to 70B parameters. It implements QLoRA (Quantized Low-Rank Adaptation) to make fine-tuning accessible without enterprise-grade hardware, achieving **75% VRAM reduction** compared to full fine-tuning while maintaining output quality.

The platform abstracts the full fine-tuning workflow model selection, dataset validation, hyperparameter optimization, real-time training monitoring, and artifact export into a guided 5-step pipeline with a live dashboard.

> *Running a 70B model shouldn't require a $10K GPU setup.*

---

## Engineering Design

**The core insight:** efficient fine-tuning is not about compute it is about which parameters actually matter. QLoRA freezes base model weights and trains only low-rank adapter matrices, reducing the trainable parameter count by orders of magnitude.

```
Base Model (frozen)
      │
      ▼
┌─────────────────┐
│  Quantization   │   4-bit NF4 quantization (bitsandbytes)
│  (QLoRA)        │   75% VRAM reduction vs full fine-tuning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LoRA Adapters  │   Low-rank matrices injected into attention layers
│  (PEFT)         │   Only adapters are trained, base weights frozen
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training Loop  │   Real-time loss, VRAM, throughput monitoring
│  (PyTorch)      │   WebSocket streaming to Next.js dashboard
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Artifact Export│   Merged model, LoRA weights, config templates
│                 │   Ready for inference deployment
└─────────────────┘
```

---

## Performance

| Metric | Value |
|---|---|
| **VRAM Reduction** | 75% vs full fine-tuning |
| **Efficiency Improvement** | ~30% training throughput gain |
| **Supported Model Range** | 2B to 70B parameters |
| **HuggingFace Models** | 1000+ compatible |
| **Setup Time** | Under 5 minutes |
| **Pipeline Stages** | 5 (Inspect, Prepare, Optimize, Train, Ship) |

---

## Features

| Feature | Description |
|---|---|
| **QLoRA Fine-Tuning** | 4-bit quantization with LoRA adapters via PEFT |
| **Smart Hyperparameters** | Automated defaults based on model size and dataset |
| **Real-Time Monitoring** | Live loss curves, VRAM usage, and throughput via WebSockets |
| **Dataset Validation** | Automated format checking and preprocessing |
| **Artifact Export** | Merged model weights, LoRA adapters, and inference configs |
| **Model Browser** | Search and load 1000+ HuggingFace models directly |
| **Multi-Format Support** | Instruction tuning, completion, chat template formats |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Fine-Tuning** | QLoRA, PEFT, bitsandbytes |
| **Model Framework** | PyTorch, HuggingFace Transformers |
| **Backend** | FastAPI, WebSockets |
| **Frontend** | Next.js, TypeScript |
| **Streaming** | Server-Sent Events |

---

## Getting Started

```bash
# Clone
git clone https://github.com/royxlead/autollmforge-python.git
cd autollmforge-python

# Backend setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd frontend && npm install

# Run
# Terminal 1
python run.py

# Terminal 2
cd frontend && npm run dev
```

Open `http://localhost:3000` and launch the workspace.

---

## The 5-Step Pipeline

**01 Inspect** : Select and analyze your base model. View parameter count, architecture, and VRAM requirements.

**02 Prepare** : Upload and validate your training dataset. Automated format detection and preprocessing.

**03 Optimize** : Configure QLoRA parameters with smart defaults. Override any hyperparameter manually.

**04 Train** : Real-time training dashboard with live loss curves, VRAM profiling, and throughput metrics.

**05 Ship** : Export merged model, standalone LoRA weights, and inference code templates.

---

## Related Work

- [Auto-Researcher](https://github.com/royxlead/auto-researcher-python) - Multi-agent academic research system
- [CURA](https://github.com/royxlead/cura-python) - RAG-based medical QA
- [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python) - Uncertainty estimation research

---

<div align="center">

**[Portfolio](https://royxlead.netlify.app) · [LinkedIn](https://linkedin.com/in/royxlead) · [ORCID](https://orcid.org/0009-0009-6582-2295)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=80&section=footer" width="100%"/>

</div>
