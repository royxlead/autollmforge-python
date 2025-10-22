# 🚀 LLM Fine-Tuning Pipeline - Backend API

Production-ready FastAPI backend for automated LLM fine-tuning with intelligent hyperparameter optimization, real-time progress tracking, and model quantization.

## ✨ Features

- **🤖 Model Analysis**: Comprehensive analysis of Hugging Face models
- **📊 Smart Hyperparameters**: AI-powered hyperparameter recommendations
- **📁 Dataset Processing**: Validation and preprocessing for JSON/CSV/JSONL
- **🔥 Training Orchestration**: Background training with progress tracking
- **⚡ Quantization**: Support for 4-bit, 8-bit, GPTQ, and GGUF
- **💻 Code Generation**: Auto-generate inference scripts, Gradio apps, and APIs
- **🔌 WebSocket Updates**: Real-time training progress
- **📦 Export & Deploy**: Download complete packages with models and code

## 🏗️ Architecture

```
backend/
├── main.py                          # FastAPI app with all endpoints
├── config.py                        # Configuration management
├── requirements.txt                 # Python dependencies
├── models/
│   └── schemas.py                   # Pydantic models
├── services/
│   ├── model_analyzer.py            # Model analysis
│   ├── hyperparameter_optimizer.py  # Hyperparameter recommendations
│   ├── dataset_processor.py         # Dataset validation
│   ├── training_service.py          # Training orchestration
│   ├── quantization_service.py      # Model quantization
│   └── code_generator.py            # Code generation
└── utils/
    ├── hf_utils.py                  # Hugging Face utilities
    ├── compute_estimator.py         # VRAM/compute calculations
    └── logger.py                    # Logging utilities
```

## 📋 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- Git

## 🛠️ Installation

### 1. Clone the Repository

```bash
cd backend
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example env file
copy .env.example .env

# Edit .env with your settings
# Add your Hugging Face token if needed
```

### 5. Create Required Directories

```bash
mkdir -p storage\models storage\datasets storage\outputs storage\temp logs cache\huggingface
```

## 🚀 Running the Server

### Development Mode

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Model Endpoints

```http
POST /api/analyze-model
```
Analyze a Hugging Face model and get comprehensive information.

**Request:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf"
}
```

**Response:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "architecture": "llama",
  "num_parameters": 6738415616,
  "parameter_size": "6.7B",
  "supported_tasks": ["text-generation"],
  "context_length": 4096,
  "vram_requirements": {
    "inference_fp16": 13.5,
    "training_lora_fp16": 18.2
  }
}
```

```http
GET /api/models/popular
```
Get list of popular pre-analyzed models.

### Dataset Endpoints

```http
POST /api/upload-dataset
```
Upload and validate a dataset file.

**Form Data:**
- `file`: Dataset file (JSON/CSV/JSONL)
- `format`: File format (json, csv, jsonl)

### Hyperparameter Endpoints

```http
POST /api/recommend-hyperparameters
```
Get AI-recommended hyperparameters.

**Request:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "dataset_id": "my_dataset",
  "compute_tier": "free",
  "task_type": "text-generation"
}
```

**Response:**
```json
{
  "config": {
    "learning_rate": 0.0002,
    "batch_size": 4,
    "num_epochs": 3,
    "lora_config": {
      "r": 8,
      "lora_alpha": 16
    }
  },
  "explanations": {...},
  "estimated_vram_gb": 18.2,
  "estimated_training_time_hours": 2.5,
  "confidence_score": 0.95
}
```

### Training Endpoints

```http
POST /api/start-training
```
Start a new training job.

```http
GET /api/training-progress/{job_id}
```
Get current training progress.

```http
WS /ws/training/{job_id}
```
WebSocket for real-time training updates.

### Quantization Endpoints

```http
POST /api/quantize
```
Quantize a fine-tuned model.

**Request:**
```json
{
  "model_path": "./storage/outputs/job_id/final_model",
  "method": "4bit",
  "bits": 4
}
```

### Code Generation Endpoints

```http
POST /api/generate-code
```
Generate deployment code.

**Request:**
```json
{
  "model_info": {...},
  "config": {...},
  "code_type": "inference"
}
```

Code types: `inference`, `gradio`, `api`, `readme`

### Export Endpoints

```http
GET /api/download-package/{job_id}
```
Download complete ZIP package with model and code.

## 🔧 Configuration

Edit `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
ENVIRONMENT=development

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Hugging Face
HF_TOKEN=your_token_here

# Storage
MODELS_DIR=./storage/models
DATASETS_DIR=./storage/datasets
OUTPUTS_DIR=./storage/outputs

# Training
MAX_CONCURRENT_TRAININGS=2
DEFAULT_DEVICE=cuda
MIXED_PRECISION=fp16

# Logging
LOG_LEVEL=INFO
```

## 📊 Usage Example

### Python Client Example

```python
import requests

# 1. Analyze model
response = requests.post(
    "http://localhost:8000/api/analyze-model",
    json={"model_id": "gpt2"}
)
model_info = response.json()
print(f"Model: {model_info['parameter_size']} parameters")

# 2. Get hyperparameter recommendations
response = requests.post(
    "http://localhost:8000/api/recommend-hyperparameters",
    json={
        "model_id": "gpt2",
        "dataset_id": "my_dataset",
        "compute_tier": "free",
        "task_type": "text-generation"
    }
)
recommendations = response.json()
print(f"Recommended LR: {recommendations['config']['learning_rate']}")

# 3. Start training
response = requests.post(
    "http://localhost:8000/api/start-training",
    json={
        "config": recommendations['config'],
        "job_name": "my_fine_tune"
    }
)
job_id = response.json()['job_id']
print(f"Job started: {job_id}")

# 4. Monitor progress
response = requests.get(
    f"http://localhost:8000/api/training-progress/{job_id}"
)
progress = response.json()
print(f"Progress: {progress['current_step']}/{progress['total_steps']}")
```

### WebSocket Example

```python
import websockets
import asyncio
import json

async def monitor_training(job_id):
    uri = f"ws://localhost:8000/ws/training/{job_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data['type'] == 'progress':
                progress = data['data']
                print(f"Epoch: {progress['current_epoch']}, Loss: {progress['train_loss']}")
            
            elif data['type'] == 'complete':
                print(f"Training complete! Status: {data['status']}")
                break

# Run
asyncio.run(monitor_training("your-job-id"))
```

## 🧪 Testing

### Manual Testing

Visit the interactive API docs at http://localhost:8000/docs and try out the endpoints.

### Curl Examples

```bash
# Analyze a model
curl -X POST "http://localhost:8000/api/analyze-model" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'

# Check health
curl "http://localhost:8000/health"

# List popular models
curl "http://localhost:8000/api/models/popular"
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Port Already in Use

```bash
# Change port in .env or command line
uvicorn main:app --port 8001
```

### Hugging Face Token

```bash
# Set token in .env
HF_TOKEN=hf_...

# Or login via CLI
huggingface-cli login
```

## 📚 Project Structure Details

### Services

- **model_analyzer.py**: Fetches and analyzes HF models, extracts architecture, counts parameters
- **hyperparameter_optimizer.py**: Intelligent hyperparameter selection based on model/data/compute
- **dataset_processor.py**: Loads, validates, and preprocesses datasets
- **training_service.py**: Manages training jobs with background execution
- **quantization_service.py**: Quantizes models using various methods
- **code_generator.py**: Generates deployment code using Jinja2 templates

### Utils

- **hf_utils.py**: Hugging Face Hub API wrapper
- **compute_estimator.py**: VRAM and training time calculations
- **logger.py**: Centralized logging with Loguru

## 🚀 Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables

Set these in production:
- `ENVIRONMENT=production`
- `API_RELOAD=false`
- `LOG_LEVEL=WARNING`
- `ALLOWED_ORIGINS=https://yourdomain.com`

## 📝 License

This project is part of the LLM Fine-Tuning Pipeline.

## 🤝 Contributing

Contributions welcome! Please follow the code style and add tests for new features.

## 📞 Support

For issues and questions, please open a GitHub issue or contact the maintainers.

---

**Built with ❤️ using FastAPI, Transformers, and PEFT**
