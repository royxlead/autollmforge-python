# ⚒️ AutoLLM Forge

**Forge Your Perfect Model - A beautiful, production-ready full-stack platform for automated LLM fine-tuning with QLoRA, AI-powered hyperparameter optimization, and real-time training monitoring.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)
![Next.js](https://img.shields.io/badge/next.js-14.2-black.svg)
![TypeScript](https://img.shields.io/badge/typescript-5.3-blue.svg)
![QLoRA](https://img.shields.io/badge/QLoRA-4bit-22c55e.svg)

<div align="center">
  
🚀 **[Live Demo](#)** | 📖 **[Documentation](#)** | 🐛 **[Report Bug](https://github.com/royxlead/autollmforge-python/issues)** | ✨ **[Request Feature](https://github.com/royxlead/autollmforge-python/issues)**

</div>

## ✨ Features

### 🎯 Core Capabilities

- **⚡ QLoRA Fine-Tuning**: 4-bit quantization with NormalFloat (NF4) for 75% memory reduction
- **🤖 Model Analysis**: Deep inspection of 1000+ Hugging Face models with VRAM estimation
- **🧠 AI Hyperparameter Tuning**: 8-tier intelligent recommendations based on model size and hardware
- **📊 Real-Time Progress**: Live training metrics with detailed progress messages every 2 seconds
- **💾 Memory Optimized**: Train 7B models on 16GB VRAM, 13B on 24GB, 70B on 80GB
- **💻 Code Generation**: Production-ready inference scripts, Gradio apps, FastAPI servers, and documentation
- **📦 Complete Export**: One-click download of fine-tuned models with all deployment files
- **🎨 Modern UI**: Beautiful dark theme with glassmorphism effects and smooth animations
- **🔐 Secure**: Hugging Face token authentication for gated models (Llama, Gemma, etc.)

### 🏗️ Technical Stack

**Backend (Python)**
- **FastAPI 0.109** - High-performance async API framework
- **PyTorch 2.1+** - Deep learning framework with CUDA support
- **Transformers 4.37+** - Hugging Face model loading and training
- **PEFT 0.8+** - Parameter-Efficient Fine-Tuning (LoRA)
- **bitsandbytes 0.42+** - 4-bit/8-bit quantization
- **Accelerate 0.26+** - Distributed training utilities
- **Pydantic 2.5** - Data validation and settings

**Frontend (TypeScript)**
- **Next.js 14.2** - React framework with App Router
- **TypeScript 5.3** - Type-safe JavaScript
- **Tailwind CSS 3.4** - Utility-first styling
- **Radix UI** - Accessible component primitives
- **Zustand 4.5** - State management
- **Lucide React** - Beautiful icons
- **Framer Motion 11** - Smooth animations

**Fine-Tuning Pipeline**
- **QLoRA**: 4-bit NF4 quantization + double quantization
- **Paged AdamW 8-bit**: Memory-efficient optimizer
- **Gradient Checkpointing**: Reduce memory footprint
- **Mixed Precision**: FP16/BF16 training

## 🎬 Demo & Screenshots

### 5-Step Pipeline

1. **🔍 Model Analysis** - Select and analyze any Hugging Face model
2. **📊 Dataset Upload** - Upload JSON datasets with validation
3. **⚙️ Hyperparameter Tuning** - Get AI-powered recommendations
4. **🔥 Training** - Monitor real-time progress with live metrics
5. **📦 Code Export** - Download production-ready deployment code

<div align="center">
  <img src="https://via.placeholder.com/800x450?text=Welcome+Screen" alt="Welcome Screen" width="800"/>
  <p><i>Beautiful welcome screen explaining the complete workflow</i></p>
</div>

<div align="center">
  <img src="https://via.placeholder.com/800x450?text=Training+Progress" alt="Training Progress" width="800"/>
  <p><i>Real-time training progress with live metrics and detailed status messages</i></p>
</div>

## 🎯 Use Cases

| User | Benefit |
|------|---------|
| **ML Engineers** | Fine-tune 7B-70B models on consumer GPUs (RTX 3090/4090) |
| **Researchers** | Rapid experimentation with 8-tier hyperparameter optimization |
| **Startups** | Deploy custom models without expensive cloud infrastructure |
| **Educators** | Interactive teaching tool for LLM fine-tuning concepts |
| **Enterprises** | Standardized, reproducible fine-tuning pipelines |

## 💡 Why QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** makes large model fine-tuning accessible:

| Feature | Benefit | Example |
|---------|---------|---------|
| **75% Memory Reduction** | 4-bit vs FP16 | 7B model: 28GB → 7GB VRAM |
| **NF4 Quantization** | Minimal accuracy loss | Maintains model quality |
| **Paged Optimizers** | No OOM errors | Stable training on consumer GPUs |
| **Double Quantization** | Extra memory savings | Nested quantization of quantization constants |
| **Consumer Hardware** | No cloud costs | RTX 3090/4090 sufficient for 13B models |

### Supported Models

✅ **Llama 2/3** (7B, 13B, 70B)  
✅ **Mistral** (7B, Mixtral 8x7B)  
✅ **Gemma** (2B, 7B)  
✅ **Falcon** (7B, 40B)  
✅ **GPT-2/Neo/J**  
✅ **Bloom** (1.7B, 3B, 7B)  
✅ Any Hugging Face causal LM model

## 📋 Requirements

### Minimum System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Linux (Ubuntu 20.04+), macOS |
| **RAM** | 16GB (32GB recommended) |
| **Storage** | 20GB free space |
| **GPU** | CUDA-capable (optional but recommended) |
| **Python** | 3.10+ |
| **Node.js** | 18+ |

### GPU Recommendations

| GPU | VRAM | Max Model Size | Training Speed |
|-----|------|----------------|----------------|
| **RTX 3060** | 12GB | 7B | ~2h/epoch |
| **RTX 3090** | 24GB | 13B | ~1h/epoch |
| **RTX 4090** | 24GB | 13B | ~45min/epoch |
| **A100 (40GB)** | 40GB | 33B | ~30min/epoch |
| **A100 (80GB)** | 80GB | 70B | ~2h/epoch |
| **CPU Only** | N/A | 7B | ~12h/epoch ⚠️ |

### Software Dependencies

**Backend:**
- Python 3.10+
- CUDA 11.8+ (for GPU)
- PyTorch 2.1+ with CUDA
- bitsandbytes 0.42+ (4-bit quantization)

**Frontend:**
- Node.js 18+
- npm or yarn or pnpm

## � Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/royxlead/autollmforge-python.git
cd autollmforge-python
```

### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install PyTorch with CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env
# Add other settings as needed

# Start backend
python main.py
```

✅ Backend running at: **http://localhost:8000**  
📚 API docs at: **http://localhost:8000/docs**

### 3️⃣ Frontend Setup

Open a **new terminal** (keep backend running):

```bash
cd frontend

# Install dependencies
npm install
# or: yarn install
# or: pnpm install

# Start frontend
npm run dev
```

✅ Frontend running at: **http://localhost:3000**

### 4️⃣ First Training Job

1. Open http://localhost:3000
2. Click **"Start Your Journey"**
3. Search for a model (e.g., `"gpt2"`)
4. Upload a dataset (JSON format with `"text"` field)
5. Get AI recommendations
6. Start training!
7. Watch real-time progress
8. Download your fine-tuned model

### 🔑 Hugging Face Token Setup

For gated models (Llama, Gemma, etc.):

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **read** access
3. Add to `backend/.env`:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```
4. Restart backend

### 🐳 Docker (Alternative)

```bash
# Coming soon
docker-compose up
```

## 📖 User Guide

### 🔍 Step 1: Model Analysis

1. **Search** for any Hugging Face model:
   - `"gpt2"` - Small model for testing
   - `"google/gemma-2b"` - Efficient 2B model
   - `"meta-llama/Llama-2-7b-hf"` - Popular 7B model
   
2. Click **"Analyze Model"** to fetch:
   - Architecture details
   - Parameter count
   - VRAM requirements (inference & training)
   - Supported tasks

3. Review metrics and click **"Select Model"**

### 📊 Step 2: Dataset Upload

**Dataset Format Required:**
```json
[
  {"text": "Your first training example..."},
  {"text": "Your second training example..."},
  {"text": "Your third training example..."}
]
```

1. **Drag & drop** your JSON file (or browse)
2. System validates:
   - ✅ JSON format
   - ✅ `"text"` field exists
   - ✅ Sample count
   - ✅ Token statistics

3. View dataset preview and stats
4. Click **"Continue"**

### ⚙️ Step 3: Hyperparameter Tuning

1. Click **"Get AI Recommendations"**
2. System analyzes:
   - Model size
   - Dataset size
   - Available VRAM
   - Compute tier

3. Review **8-tier recommendations**:
   - Learning rate
   - Batch size
   - LoRA rank (r)
   - Epochs
   - Gradient accumulation

4. Adjust manually (optional)
5. Click **"Start Training"**

### 🔥 Step 4: Training Monitor

**Real-time updates every 2 seconds:**

- 📥 **Model Download**: "Downloading model with 4-bit quantization..."
- 🔧 **Quantization**: "Preparing model for QLoRA training..."
- 📊 **Dataset**: "Tokenizing training dataset..."
- 🚀 **Training**: "Step 45/156 (29%) | Loss: 0.3456"
- ✅ **Complete**: "Training completed successfully!"

**Live Metrics:**
- Current step / Total steps
- Loss value (updates every 10 steps)
- Learning rate
- Samples per second
- GPU memory usage
- ETA remaining

### 📦 Step 5: Code Export

**4 types of production code generated:**

1. **Inference Script** (`inference.py`)
   - Load fine-tuned model with 4-bit quantization
   - Generate text with customizable parameters
   - Error handling and device management

2. **Gradio App** (`gradio_app.py`)
   - Interactive web UI
   - 4 parameter controls (temperature, top_p, max_length, repetition_penalty)
   - Example prompts included

3. **FastAPI Server** (`api_server.py`)
   - REST API with CORS
   - Pydantic validation
   - Health checks and OpenAPI docs
   - Production-ready deployment

4. **README** (`README.md`)
   - Installation instructions
   - Quick start guide
   - API documentation
   - Troubleshooting tips

**Actions:**
- 📋 **Copy** code to clipboard
- 💾 **Download** individual files
- 📦 **Export All** - ZIP with model + all code

## 📁 Project Structure

```
autollmforge-python/
├── backend/                          # FastAPI Backend (Python)
│   ├── main.py                       # API server & routes
│   ├── config.py                     # Settings & environment
│   ├── requirements.txt              # Python dependencies
│   ├── .env                          # Environment variables (create this)
│   │
│   ├── models/
│   │   └── schemas.py                # Pydantic data models
│   │
│   ├── services/                     # Business logic
│   │   ├── model_analyzer.py         # HF model analysis
│   │   ├── hyperparameter_optimizer.py  # 8-tier AI recommendations
│   │   ├── dataset_processor.py      # Dataset validation
│   │   ├── training_service.py       # QLoRA training pipeline
│   │   ├── quantization_service.py   # Post-training quantization
│   │   └── code_generator.py         # Production code templates
│   │
│   ├── utils/
│   │   ├── hf_utils.py               # Hugging Face helpers
│   │   ├── compute_estimator.py      # VRAM/time estimation
│   │   └── logger.py                 # Logging configuration
│   │
│   └── storage/                      # Auto-created directories
│       ├── datasets/                 # Uploaded datasets
│       ├── outputs/                  # Fine-tuned models
│       └── cache/                    # Model cache
│
├── frontend/                         # Next.js Frontend (TypeScript)
│   ├── app/
│   │   ├── layout.tsx                # Root layout
│   │   ├── page.tsx                  # Welcome screen
│   │   └── globals.css               # Global styles
│   │
│   ├── components/                   # React components
│   │   ├── ModelAnalysis.tsx         # Step 1: Model selection
│   │   ├── DatasetUpload.tsx         # Step 2: Dataset upload
│   │   ├── HyperparameterTuning.tsx  # Step 3: Hyperparameters
│   │   ├── Training.tsx              # Step 4: Training monitor
│   │   └── CodeGeneration.tsx        # Step 5: Code export
│   │
│   ├── store/
│   │   └── pipelineStore.ts          # Zustand state management
│   │
│   ├── types/
│   │   └── index.ts                  # TypeScript definitions
│   │
│   ├── package.json                  # Node dependencies
│   └── .env.local                    # Frontend config (create this)
│
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze-model` | Analyze Hugging Face model |
| `GET` | `/api/models/popular` | Get popular model list |
| `POST` | `/api/upload-dataset` | Upload & validate dataset |
| `POST` | `/api/recommend-hyperparameters` | Get AI recommendations |
| `POST` | `/api/start-training` | Start QLoRA training job |
| `GET` | `/api/training-progress/{job_id}` | Get training progress |
| `POST` | `/api/cancel-training/{job_id}` | Cancel training job |
| `GET` | `/api/training-jobs` | List all training jobs |
| `WS` | `/ws/training/{job_id}` | Real-time training updates |
| `POST` | `/api/generate-code` | Generate deployment code |
| `GET` | `/api/download-model/{job_id}` | Download fine-tuned model |
| `GET` | `/api/download-package/{job_id}` | Download complete ZIP |

### Example: Start Training

```bash
curl -X POST http://localhost:8000/api/start-training \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model_id": "gpt2",
      "dataset_id": "your-dataset.json",
      "num_epochs": 3,
      "learning_rate": 0.0002,
      "batch_size": 4,
      "use_lora": true,
      "lora_config": {
        "r": 16,
        "lora_alpha": 32
      }
    },
    "job_name": "my-first-training"
  }'
```

### Example: Get Progress

```bash
curl http://localhost:8000/api/training-progress/{job_id}
```

**Response:**
```json
{
  "job_id": "089b4602-1275-4a...",
  "status": "running",
  "current_step": 45,
  "total_steps": 156,
  "current_epoch": 1,
  "train_loss": 0.3456,
  "learning_rate": 0.0002,
  "samples_per_second": 15.34,
  "progress_message": "🔥 Step 45/156 (29%) | Loss: 0.3456 | LR: 2.00e-04"
}
```

📚 **Full API Documentation**: http://localhost:8000/docs (when backend is running)

## ⚙️ Configuration

### Backend Environment (`backend/.env`)

```env
# Required: Hugging Face Token for gated models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
ENVIRONMENT=development

# CORS (for frontend)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Storage Paths
HF_CACHE_DIR=./cache/huggingface
MODELS_DIR=./storage/models
DATASETS_DIR=./storage/datasets
OUTPUTS_DIR=./storage/outputs
TEMP_DIR=./storage/temp

# Training Configuration
MAX_CONCURRENT_TRAININGS=2
DEFAULT_DEVICE=cuda
MIXED_PRECISION=fp16
USE_QLORA_BY_DEFAULT=true
DEFAULT_QUANTIZATION=4bit

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Frontend Environment (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Hyperparameter Optimization Tiers

The system uses **8 intelligence tiers** for recommendations:

| Tier | Model Size | Batch Size | LoRA Rank | Learning Rate | Use Case |
|------|-----------|------------|-----------|---------------|----------|
| **1** | <1B | 16 | 8 | 3e-4 | Testing/Experimentation |
| **2** | 1-3B | 8 | 16 | 2e-4 | Small models |
| **3** | 3-7B | 4 | 16 | 2e-4 | Standard fine-tuning |
| **4** | 7-13B | 2 | 32 | 1.5e-4 | Larger models |
| **5** | 13-30B | 1 | 64 | 1e-4 | Very large models |
| **6** | 30-65B | 1 | 64 | 8e-5 | Huge models |
| **7** | 65-100B | 1 | 128 | 5e-5 | Massive models |
| **8** | >100B | 1 | 256 | 3e-5 | Extreme scale |

*Automatically adjusted based on available VRAM and dataset size*

## 🧪 Advanced Features

### 8-Tier Hyperparameter Intelligence

Automatically optimizes based on:
- Model architecture and size
- Dataset characteristics
- Available compute resources
- Task complexity

### Real-Time Progress Tracking

**Backend sends updates every training step:**
- 📥 Model download progress
- 🔧 Quantization status
- 📊 Dataset processing
- 🔥 Training metrics (loss, LR, speed)
- 💾 Model saving

**Frontend polls every 2 seconds** for smooth UI updates.

### Production-Ready Code Generation

**Inference Script Features:**
- 4-bit quantization loading
- Device auto-detection (CUDA/CPU)
- Configurable generation parameters
- Error handling
- Memory optimization

**Gradio App Features:**
- Interactive web interface
- Real-time text generation
- Parameter sliders
- Example prompts
- One-command launch

**FastAPI Server Features:**
- RESTful API
- CORS configuration
- Pydantic validation
- Async request handling
- OpenAPI documentation
- Health check endpoint

### Memory Optimization Techniques

1. **4-bit Quantization**: NF4 (NormalFloat 4-bit)
2. **Double Quantization**: Quantize quantization constants
3. **Paged Optimizers**: Prevent OOM with automatic offloading
4. **Gradient Checkpointing**: Trade compute for memory
5. **Mixed Precision**: FP16/BF16 training
6. **LoRA**: Train only 0.1% of parameters

## 🧪 Testing & Validation

### Quick Test with GPT-2

Perfect for testing the complete pipeline:

```bash
# 1. Start both servers (backend + frontend)

# 2. Go to http://localhost:3000

# 3. Model Analysis
Search: "gpt2"
Click: "Analyze Model"

# 4. Dataset Upload
# Create test dataset:
echo '[{"text":"Hello world"},{"text":"Test example"}]' > test.json
# Upload test.json

# 5. Get Recommendations
Click: "Get AI Recommendations"
# Should show: Tier 1, batch_size=16, r=8

# 6. Start Training
Click: "Start Training"
# Watch real-time progress updates

# 7. Export Code
Click through: Inference → Gradio → API → README
Download all
```

**Expected Duration**: ~2-3 minutes on GPU, ~10-15 minutes on CPU

### Verify Installation

```bash
# Check backend
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Check PyTorch GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check bitsandbytes
python -c "import bitsandbytes; print('✅ bitsandbytes OK')"
```

## 🐛 Troubleshooting

### ❌ Backend Won't Start

**Problem**: `ModuleNotFoundError` or import errors

```bash
# Solution 1: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Solution 2: Install PyTorch separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Solution 3: Check Python version
python --version  # Should be 3.10+
```

**Problem**: Port 8000 already in use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### ❌ Training Fails Immediately

**Problem**: CUDA out of memory

```
Solutions:
1. Reduce batch_size (try 1 or 2)
2. Enable gradient_checkpointing
3. Increase gradient_accumulation_steps
4. Use smaller model (e.g., 2B instead of 7B)
```

**Problem**: "Cannot access gated repo"

```bash
# Add HF token to backend/.env
HF_TOKEN=hf_xxxxxxxxxxxxx

# Restart backend
```

**Problem**: bitsandbytes not working (Windows)

```bash
# Use pre-built wheels
pip uninstall bitsandbytes
pip install bitsandbytes --prefer-binary

# Or use Windows-specific build
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### ❌ Frontend Issues

**Problem**: "Failed to fetch" errors

```
Solutions:
1. Check backend is running (http://localhost:8000/health)
2. Verify CORS settings in backend/.env
3. Check browser console for errors
4. Disable browser extensions (ad blockers)
```

**Problem**: Progress not updating

```
Solutions:
1. Check browser console for WebSocket errors
2. Verify job_id is valid
3. Check backend logs for training progress
4. Refresh page and restart training
```

### ❌ No GPU Detected

**Problem**: `torch.cuda.is_available()` returns `False`

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 💬 Still Having Issues?

1. Check [GitHub Issues](https://github.com/royxlead/autollmforge-python/issues)
2. Review logs: `backend/logs/app.log`
3. Check browser console (F12)
4. Open a [new issue](https://github.com/royxlead/autollmforge-python/issues/new) with:
   - Error message
   - System specs (OS, GPU, RAM)
   - Python/Node versions
   - Steps to reproduce

## 🚀 Deployment

### Production Checklist

- [ ] Change `ENVIRONMENT=production` in `backend/.env`
- [ ] Use strong `HF_TOKEN` with appropriate permissions
- [ ] Configure proper `ALLOWED_ORIGINS` for CORS
- [ ] Set up HTTPS (SSL certificates)
- [ ] Configure persistent storage for models
- [ ] Set up monitoring and alerts
- [ ] Configure log rotation
- [ ] Test all endpoints with production data
- [ ] Set up backup strategy for fine-tuned models
- [ ] Configure rate limiting (if public)
- [ ] Set up error tracking (Sentry, etc.)

### Deployment Options

#### Option 1: Cloud VM (Recommended for GPU)

**AWS EC2 / GCP Compute Engine / Azure VM**

```bash
# Instance requirements:
- GPU instance (g4dn.xlarge, n1-standard-4-k80, etc.)
- 50GB+ storage
- Ubuntu 20.04+

# Setup:
1. SSH into instance
2. Install NVIDIA drivers + CUDA
3. Clone repo and setup (same as Quick Start)
4. Use systemd for auto-restart
5. Configure nginx as reverse proxy
6. Setup SSL with Let's Encrypt
```

#### Option 2: Railway / Render (CPU-only)

```bash
# For CPU-only training (slower but cheaper)
1. Connect GitHub repo
2. Set environment variables
3. Deploy backend and frontend separately
4. Use persistent volumes for storage
```

#### Option 3: Docker (Coming Soon)

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./storage:/app/storage"]
    
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

### Scaling Considerations

| Users | Setup | Hardware |
|-------|-------|----------|
| **1-10** | Single server | 1x GPU instance |
| **10-100** | Load balancer + 2 servers | 2x GPU instances |
| **100-1000** | Kubernetes cluster | GPU node pool |
| **1000+** | Multi-region + queue system | Dedicated infrastructure |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for Transformers and PEFT
- FastAPI and Next.js teams
- Open source community

## 📞 Contact & Support

- **GitHub**: [@royxlead](https://github.com/royxlead)
- **Email**: royxlead@proton.me
- **Issues**: [Report a bug](https://github.com/royxlead/autollmforge-python/issues)
- **Repository**: https://github.com/royxlead/autollmforge-python

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/royxlead/autollmforge-python?style=social)
![GitHub forks](https://img.shields.io/github/forks/royxlead/autollmforge-python?style=social)
![GitHub issues](https://img.shields.io/github/issues/royxlead/autollmforge-python)
![GitHub last commit](https://img.shields.io/github/last-commit/royxlead/autollmforge-python)

## 🗺️ Roadmap

- [ ] Docker containerization with docker-compose
- [ ] Distributed training across multiple GPUs
- [ ] Additional quantization methods (AWQ, GGUF)
- [ ] Model merging and ensemble capabilities
- [ ] Automated evaluation benchmarks
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Custom model architecture support
- [ ] Multi-modal fine-tuning (vision + language)
- [ ] Dataset preprocessing pipelines
- [ ] Integration with Weights & Biases / MLflow

---

<div align="center">

**⚒️ Built with passion for the AI community**

⭐ **Star this repo if you find it useful!** ⭐

[🐛 Report Bug](https://github.com/royxlead/autollmforge-python/issues) • [✨ Request Feature](https://github.com/royxlead/autollmforge-python/issues) • [📖 Documentation](https://github.com/royxlead/autollmforge-python)

**Made with ❤️ using FastAPI, Next.js, and QLoRA**

</div>
