# 🚀 LLM Fine-Tuning Automation Pipeline (QLoRA Optimized)

**Production-ready full-stack application for automated LLM fine-tuning with QLoRA (Quantized LoRA), intelligent hyperparameter optimization, real-time monitoring, and one-click deployment.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Next.js](https://img.shields.io/badge/next.js-14-black.svg)
![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)
![QLoRA](https://img.shields.io/badge/QLoRA-4bit-green.svg)

## ✨ Key Features

### 🎯 Core Capabilities

- **⚡ QLoRA by Default**: 4-bit NormalFloat quantization for 75% memory reduction with minimal accuracy loss
- **🤖 Smart Model Analysis**: Comprehensive analysis of 1000+ Hugging Face models
- **🧠 AI-Powered Hyperparameters**: Intelligent recommendations optimized for QLoRA fine-tuning
- **📊 Real-Time Monitoring**: WebSocket-powered live training updates with interactive charts
- **💾 Ultra Memory-Efficient**: Train 13B models on 16GB VRAM, 70B models on 48GB VRAM
- **💻 Auto Code Generation**: Production-ready inference scripts with QLoRA optimizations
- **📦 One-Click Export**: Download complete packages with models, code, and documentation
- **🎨 Beautiful UI**: Modern, responsive interface with glassmorphism design

### 🏗️ Architecture Highlights

- **Backend**: FastAPI with async support, QLoRA-optimized training pipeline
- **Frontend**: Next.js 14 with App Router, TypeScript, Tailwind CSS
- **Fine-Tuning**: QLoRA (4-bit NF4 + double quantization) + PEFT
- **Optimizer**: Paged AdamW 8-bit for memory efficiency
- **Quantization**: bitsandbytes integration for 4-bit/8-bit inference
- **Real-Time**: WebSocket connections for training updates
- **Type-Safe**: Full TypeScript coverage, Pydantic models
- **Production-Ready**: Error handling, logging, monitoring, and security

## 📸 Screenshots

```
Landing Page → Model Selection → Dataset Upload → Hyperparameters → 
Training Monitor → Quantization → Export & Deploy
```

## 🎯 Use Cases

- **ML Engineers**: Fine-tune 7B-70B models on consumer GPUs with QLoRA
- **Researchers**: Memory-efficient experimentation with different configurations
- **Startups**: Deploy custom models quickly without expensive infrastructure
- **Educators**: Teaching LLM fine-tuning concepts with practical QLoRA implementation
- **Enterprises**: Standardized QLoRA fine-tuning pipelines for production

## 💡 QLoRA Benefits

**QLoRA (Quantized Low-Rank Adaptation)** enables efficient fine-tuning of large language models:

- **75% Memory Reduction**: 4-bit quantization vs FP16 (e.g., 7B model: 28GB → 7GB)
- **Minimal Accuracy Loss**: NF4 (NormalFloat 4-bit) preserves model quality
- **Faster Training**: Reduced memory allows larger batch sizes
- **Consumer Hardware**: Train 13B models on RTX 4090 (24GB), 70B on A100 (80GB)
- **Production Quality**: Matches full fine-tuning performance on most tasks

## 📋 Prerequisites

### System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 16GB+ recommended
- **GPU**: CUDA-capable GPU (optional but recommended)
- **Storage**: 10GB+ free space

### Software Requirements

**Backend:**
- Python 3.10 or higher
- pip or conda
- CUDA 11.8+ (for GPU training with QLoRA)
- PyTorch 2.0+ with CUDA support
- bitsandbytes for 4-bit quantization

**Frontend:**
- Node.js 18 or higher
- npm or yarn

**Recommended Hardware for QLoRA:**
- **RTX 3090/4090 (24GB)**: Fine-tune models up to 13B parameters
- **A100 (40GB)**: Fine-tune models up to 33B parameters  
- **A100 (80GB)**: Fine-tune models up to 70B parameters
- **CPU Training**: Possible but very slow (not recommended)

## 🛠️ Quick Start

### Option 1: Full Setup (Recommended)

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline
```

#### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings (HF token, etc.)

# Run backend
python main.py
```

Backend will be available at: http://localhost:8000

#### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# Edit .env.local if needed

# Run frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

#### 4. Open Application

Navigate to http://localhost:3000 and start fine-tuning!

### Option 2: Docker Setup (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up
```

## 📖 Documentation

### Quick Links

- [Backend README](./backend/README.md) - API documentation and backend setup
- [Frontend README](./frontend/README.md) - UI components and frontend guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when backend is running)

### User Guide

#### Step 1: Select Model

1. Search for a Hugging Face model (e.g., "gpt2", "llama-2-7b")
2. Click "Analyze Model" to fetch details
3. Review architecture, parameters, and VRAM requirements
4. Click "Continue" to proceed

#### Step 2: Upload Dataset

1. Drag & drop your dataset file (JSON, CSV, or JSONL)
2. Or enter a Hugging Face dataset ID
3. Review dataset statistics and validation warnings
4. Click "Continue"

#### Step 3: Configure Hyperparameters

1. Click "Get AI Recommendations"
2. Review recommended configuration
3. Adjust parameters with interactive sliders (optional)
4. Review memory and time estimates
5. Click "Start Training"

#### Step 4: Monitor Training

1. Watch real-time training progress
2. View loss curves and metrics
3. Check GPU utilization
4. See live logs
5. Wait for training to complete

#### Step 5: Quantize Model (Optional)

1. Select quantization method (4-bit, 8-bit, GPTQ, GGUF)
2. Review compression comparison
3. Click "Quantize Model"
4. Wait for quantization to complete

#### Step 6: Export & Deploy

1. Review training summary
2. View generated code (inference, Gradio, API)
3. Copy code to clipboard or download files
4. Download complete package (ZIP)
5. Deploy to your infrastructure!

## 🏗️ Project Structure

```
llm-finetuning-pipeline/
├── backend/                    # FastAPI Backend
│   ├── main.py                 # API entry point
│   ├── config.py               # Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   ├── model_analyzer.py
│   │   ├── hyperparameter_optimizer.py
│   │   ├── dataset_processor.py
│   │   ├── training_service.py
│   │   ├── quantization_service.py
│   │   └── code_generator.py
│   └── utils/
│       ├── hf_utils.py
│       ├── compute_estimator.py
│       └── logger.py
├── frontend/                   # Next.js Frontend
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # Landing page
│   │   └── pipeline/
│   │       └── page.tsx       # Pipeline interface
│   ├── components/            # React components
│   ├── lib/
│   │   ├── api.ts            # API client
│   │   ├── websocket.ts      # WebSocket manager
│   │   └── utils.ts          # Utilities
│   ├── store/
│   │   └── pipelineStore.ts  # State management
│   └── types/
│       └── index.ts          # TypeScript types
└── README.md                 # This file
```

## 🔌 API Endpoints

### Model Endpoints

```http
POST   /api/analyze-model             # Analyze HF model
GET    /api/models/popular            # Get popular models
```

### Dataset Endpoints

```http
POST   /api/upload-dataset            # Upload dataset file
POST   /api/validate-dataset          # Validate dataset
```

### Hyperparameter Endpoints

```http
POST   /api/recommend-hyperparameters # Get AI recommendations
```

### Training Endpoints

```http
POST   /api/start-training            # Start training job
GET    /api/training-progress/{job_id} # Get progress
POST   /api/cancel-training/{job_id}  # Cancel training
GET    /api/training-jobs             # List all jobs
WS     /ws/training/{job_id}          # Real-time updates
```

### Quantization Endpoints

```http
POST   /api/quantize                  # Quantize model
GET    /api/quantization-comparison/{job_id} # Compare methods
```

### Code Generation Endpoints

```http
POST   /api/generate-code             # Generate deployment code
```

### Export Endpoints

```http
GET    /api/download-package/{job_id} # Download ZIP package
GET    /api/download-file/{job_id}/{filename} # Download specific file
```

## 🔧 Configuration

### Backend (.env)

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Hugging Face
HF_TOKEN=your_huggingface_token_here

# Storage
MODELS_DIR=./storage/models
DATASETS_DIR=./storage/datasets
OUTPUTS_DIR=./storage/outputs

# QLoRA Training Configuration
MAX_CONCURRENT_TRAININGS=2
DEFAULT_DEVICE=cuda
MIXED_PRECISION=fp16
USE_QLORA_BY_DEFAULT=true
DEFAULT_QUANTIZATION=4bit

# QLoRA Settings
QLORA_COMPUTE_DTYPE=float16
QLORA_QUANT_TYPE=nf4
QLORA_DOUBLE_QUANT=true
QLORA_OPTIMIZER=paged_adamw_8bit
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## 📊 Tech Stack

### Backend

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API framework |
| Transformers | HuggingFace models |
| PEFT | QLoRA (LoRA with quantization) |
| bitsandbytes | 4-bit/8-bit quantization |
| PyTorch | Deep learning framework |
| Uvicorn | ASGI server |
| Pydantic | Data validation |
| Loguru | Logging |

**QLoRA Stack:**
- 4-bit NF4 quantization (NormalFloat)
- Double quantization for nested efficiency
- Paged AdamW 8-bit optimizer
- Gradient checkpointing
- CUDA kernels via bitsandbytes

### Frontend

| Technology | Purpose |
|------------|---------|
| Next.js 14 | React framework |
| TypeScript | Type safety |
| Tailwind CSS | Styling |
| Radix UI | UI components |
| Zustand | State management |
| React Query | Data fetching |
| Axios | HTTP client |
| Recharts | Data visualization |
| Framer Motion | Animations |

## 🧪 Testing

### Backend Testing

```bash
cd backend
pytest tests/
```

### Frontend Testing

```bash
cd frontend
npm run test
```

### Manual Testing

1. Start backend and frontend
2. Navigate to http://localhost:3000
3. Test each pipeline step
4. Verify real-time updates
5. Check downloads

## 🐛 Troubleshooting

### Common Issues

**Backend won't start:**
- Check Python version (`python --version`)
- Verify all dependencies installed (`pip list`)
- Check port 8000 not in use
- Review logs in `backend/logs/app.log`

**Frontend won't start:**
- Check Node version (`node --version`)
- Delete `node_modules` and reinstall
- Clear Next.js cache (`.next` folder)
- Check port 3000 not in use

**Training fails:**
- Verify GPU is available (`torch.cuda.is_available()`)
- Check VRAM sufficient for model
- Review training job logs
- Try with smaller batch size

**WebSocket not connecting:**
- Verify backend is running
- Check CORS settings
- Review browser console for errors
- Ensure no proxy blocking WebSocket

### Getting Help

1. Check documentation
2. Review GitHub issues
3. Check logs (backend and browser console)
4. Open a new issue with details

## 🚀 Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production` in backend
- [ ] Configure production database (if using)
- [ ] Setup Redis for job queue
- [ ] Configure CORS properly
- [ ] Use HTTPS for frontend
- [ ] Setup monitoring and logging
- [ ] Configure backups
- [ ] Test all functionality

### Deployment Options

**Backend:**
- AWS EC2 / Azure VM / GCP Compute Engine
- Docker containers
- Kubernetes cluster
- Railway / Render / fly.io

**Frontend:**
- Vercel (recommended)
- Netlify
- AWS Amplify
- Docker + Nginx

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

## 📞 Contact

- **GitHub**: [Your GitHub]
- **Email**: your.email@example.com
- **Twitter**: [@yourusername]

---

**Built with ❤️ by [Your Name]**

⭐ Star this repo if you find it useful!
