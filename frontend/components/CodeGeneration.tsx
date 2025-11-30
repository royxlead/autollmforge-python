"use client";

import { useState, useEffect } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import { Download, Code, Rocket, FileText, CheckCircle, PackageOpen, AlertCircle, Sparkles, BarChart3, FileCheck, Clock, Tag, User, Activity } from 'lucide-react';

export default function CodeGeneration() {
  const { modelInfo, trainingConfig, trainingJobId, evalMetrics, modelCard, experimentMetadata, setEvalMetrics, setModelCard, setExperimentMetadata } = usePipelineStore();
  const [selectedType, setSelectedType] = useState<'inference' | 'gradio' | 'api' | 'readme'>('inference');
  const [generatedCode, setGeneratedCode] = useState('');
  const [isDownloadingModel, setIsDownloadingModel] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [isLoadingEval, setIsLoadingEval] = useState(false);

  // Fetch evaluation metrics and model card when training job is available
  useEffect(() => {
    const fetchEvalData = async () => {
      if (!trainingJobId) return;
      
      setIsLoadingEval(true);
      try {
        // Fetch evaluation metrics
        const evalResponse = await fetch(`http://localhost:8000/api/experiment/${trainingJobId}/eval`);
        if (evalResponse.ok) {
          const evalData = await evalResponse.json();
          setEvalMetrics(evalData.metrics);
          if (evalData.model_card) {
            setModelCard(evalData.model_card);
          }
        }
        
        // Fetch experiment metadata
        const metaResponse = await fetch(`http://localhost:8000/api/experiment/${trainingJobId}/metadata`);
        if (metaResponse.ok) {
          const metaData = await metaResponse.json();
          setExperimentMetadata(metaData);
        }
      } catch (err) {
        console.log('Could not fetch evaluation data:', err);
      } finally {
        setIsLoadingEval(false);
      }
    };
    
    fetchEvalData();
  }, [trainingJobId, setEvalMetrics, setModelCard, setExperimentMetadata]);

  const codeTypes = [
    { id: 'inference' as const, label: 'Inference Script', icon: Code, description: 'Load and use your model' },
    { id: 'gradio' as const, label: 'Gradio App', icon: Rocket, description: 'Interactive web interface' },
    { id: 'api' as const, label: 'FastAPI Server', icon: Code, description: 'REST API endpoint' },
    { id: 'readme' as const, label: 'README', icon: FileText, description: 'Documentation' },
  ];

  const generateSampleCode = (type: string) => {
    const modelId = modelInfo?.model_id || 'your-model-id';
    const checkpointPath = './checkpoint-final';
    
    switch (type) {
      case 'inference':
        return `"""
QLoRA Fine-Tuned Model Inference Script
Load and generate text with your fine-tuned model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model with quantization
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "${modelId}",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter weights
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "${checkpointPath}")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("${modelId}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate text function
def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9):
    """Generate text from a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
    print(f"\\nPrompt: {prompt}")
    print("\\nGenerating...")
    
    result = generate_text(prompt, max_length=150, temperature=0.7)
    print(f"\\nGenerated:\\n{result}")
`;

      case 'gradio':
        return `"""
Gradio Web Interface for Fine-Tuned Model
Interactive demo with adjustable parameters
Requirements: pip install gradio torch transformers peft bitsandbytes accelerate
"""
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model... This may take a minute.")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "${modelId}",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "${checkpointPath}")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("${modelId}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    """Generate text with the fine-tuned model"""
    if not prompt.strip():
        return "Please enter a prompt."
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_length),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=temperature > 0,
                repetition_penalty=float(repetition_penalty),
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Fine-Tuned Model Demo")
    gr.Markdown("Generate text using your QLoRA fine-tuned model")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5
            )
            
            with gr.Accordion("Generation Parameters", open=False):
                max_length = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    label="Repetition Penalty"
                )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text",
                lines=15,
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Once upon a time"],
            ["The future of AI is"],
            ["In a world where technology"]
        ],
        inputs=prompt_input
    )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_p, repetition_penalty],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
`;

      case 'api':
        return `"""
FastAPI Server for Fine-Tuned Model
REST API endpoint for text generation
Requirements: pip install fastapi uvicorn torch transformers peft bitsandbytes accelerate
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional

app = FastAPI(
    title="Fine-Tuned Model API",
    description="API for QLoRA fine-tuned language model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_length: int = Field(200, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    parameters: dict

# Global model variables
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer
    
    print("Loading model... This may take a minute.")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "${modelId}",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "${checkpointPath}")
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("${modelId}", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "${modelId}",
        "message": "Fine-tuned model API is running"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "device": str(model.device) if model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;

      case 'readme':
        return `# QLoRA Fine-Tuned Model 🚀

## 📋 Model Details
- **Base Model:** \`${modelId}\`
- **Fine-Tuning Method:** QLoRA (4-bit Quantization)
- **Task:** Text Generation
- **Framework:** Hugging Face Transformers + PEFT

## 🔧 Training Configuration
- **Quantization:** 4-bit NormalFloat (NF4)
- **Double Quantization:** ✅ Enabled
- **Optimizer:** paged_adamw_8bit
- **Learning Rate:** ${trainingConfig?.learning_rate || '2e-4'}
- **Batch Size:** ${trainingConfig?.batch_size || 4}
- **Gradient Accumulation:** ${trainingConfig?.gradient_accumulation_steps || 4}
- **Epochs:** ${trainingConfig?.num_epochs || 3}
- **Max Sequence Length:** ${trainingConfig?.max_seq_length || 512}

## 💾 Memory Requirements
- **Training VRAM:** ~${Math.ceil((modelInfo?.num_parameters || 7e9) * 0.55 / 1e9)}GB
- **Inference VRAM:** ~${Math.ceil((modelInfo?.num_parameters || 7e9) * 0.55 / 1e9)}GB
- **Model Parameters:** ${modelInfo?.num_parameters ? (modelInfo.num_parameters / 1e9).toFixed(2) + 'B' : 'N/A'}

## 📦 Installation

\`\`\`bash
pip install torch transformers peft bitsandbytes accelerate
\`\`\`

## 🚀 Quick Start

### Basic Inference

\`\`\`python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "${modelId}",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./checkpoint-final")
tokenizer = AutoTokenizer.from_pretrained("${modelId}")

# Generate
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\`\`\`

## 📁 Model Files

The checkpoint includes:
- \`adapter_model.bin\` - LoRA adapter weights (~16-32MB)
- \`adapter_config.json\` - LoRA configuration
- \`config.json\` - Model configuration
- \`tokenizer.json\` - Tokenizer files
- \`training_args.json\` - Training parameters

## 🎯 Use Cases

This fine-tuned model can be used for:
- Text generation
- Creative writing
- Domain-specific text completion
- Chatbot applications
- Custom text classification (with modifications)

## ⚡ Performance Tips

1. **Batch Processing:** Use batch generation for multiple prompts
2. **Temperature:** Lower (0.3-0.7) for focused text, higher (0.8-1.5) for creative
3. **Top-p Sampling:** Use 0.9 for balanced diversity
4. **Memory:** Monitor GPU memory usage with \`nvidia-smi\`

## 🐛 Troubleshooting

**Out of Memory Error:**
- Reduce batch size
- Lower max_seq_length
- Enable gradient checkpointing

**Slow Generation:**
- Use smaller max_new_tokens
- Enable KV cache
- Consider using flash attention

## 📚 Citation

If you use this model, please cite:

\`\`\`bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
\`\`\`

## 📄 License

This model inherits the license from the base model [\`${modelId}\`](https://huggingface.co/${modelId}).

## 🤝 Contributing

For issues or improvements, please open an issue or pull request.

## 📞 Contact

For questions about this fine-tuned model, please contact the model creator.

---

**Generated by AutoLLM Forge** ⚒️
`;

      default:
        return '// Select a code type to generate';
    }
  };

  const handleGenerate = () => {
    setGeneratedCode(generateSampleCode(selectedType));
  };

  const handleDownload = () => {
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedType}.${selectedType === 'readme' ? 'md' : 'py'}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadModel = async () => {
    if (!trainingJobId) {
      setDownloadError('No training job ID found');
      return;
    }

    setIsDownloadingModel(true);
    setDownloadError(null);

    try {
      console.log('Downloading model for job:', trainingJobId);
      
      const response = await fetch(`http://localhost:8000/api/download-model/${trainingJobId}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to download model: ${response.status} - ${errorText}`);
      }

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `model-${trainingJobId}.zip`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) filename = filenameMatch[1];
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);

      console.log('Model downloaded successfully');
    } catch (err: any) {
      console.error('Error downloading model:', err);
      setDownloadError(err.message);
    } finally {
      setIsDownloadingModel(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="text-center">
        <div className="inline-block mb-3">
          <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold tracking-wider">
            STEP 5 OF 5
          </div>
        </div>
        <div className="flex items-center justify-center gap-4 mb-3">
          <div className="w-14 h-14 bg-green-500/20 rounded-xl flex items-center justify-center shadow-lg shadow-green-500/30">
            <CheckCircle className="w-8 h-8 text-green-400 animate-pulse" />
          </div>
          <h2 className="text-4xl font-bold gradient-text">Training Complete!</h2>
        </div>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Export code to use your fine-tuned model.
        </p>
      </div>

      {/* Download Fine-Tuned Model */}
      <div className="relative bg-gradient-to-br from-green-500/20 to-green-500/5 border border-green-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm hover:border-green-500/40 transition-all duration-300">
        <div className="absolute top-0 right-0 w-40 h-40 bg-green-500/10 rounded-full blur-3xl"></div>
        <div className="relative flex items-start gap-6">
          <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center flex-shrink-0 hover:scale-110 transition-transform duration-300">
            <PackageOpen className="w-8 h-8 text-green-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-2xl font-bold text-white mb-3">
              Download Your Fine-Tuned Model
            </h3>
            <p className="text-sm text-gray-300 mb-5 leading-relaxed">
              Download the complete QLoRA fine-tuned model including adapter weights, configuration files, and tokenizer. 
              This ZIP file contains everything you need to use your model locally.
            </p>
            <div className="flex items-center gap-4 flex-wrap">
              <button
                onClick={handleDownloadModel}
                disabled={isDownloadingModel || !trainingJobId}
                className="px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed font-bold text-lg transition-all duration-300 hover:glow-strong flex items-center gap-3 hover:scale-105 shadow-lg shadow-green-500/20"
              >
                {isDownloadingModel ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Downloading Model...
                  </>
                ) : (
                  <>
                    <Download className="w-5 h-5" />
                    Download Model Package
                  </>
                )}
              </button>
              {trainingJobId && (
                <span className="text-xs text-gray-500 font-mono bg-black/30 px-3 py-2 rounded-lg">
                  Job ID: {trainingJobId.slice(0, 16)}...
                </span>
              )}
            </div>
            {downloadError && (
              <div className="mt-4 flex items-start gap-2 text-sm text-red-400 bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{downloadError}</span>
              </div>
            )}
            {!trainingJobId && (
              <p className="mt-4 text-sm text-yellow-400 bg-yellow-500/10 p-3 rounded-lg border border-yellow-500/20 inline-block">
                ⚠️ No training job found. Please complete training first.
              </p>
            )}
            <div className="mt-6 pt-6 border-t border-green-500/20">
              <p className="text-xs font-bold text-gray-400 mb-3 tracking-wider">📦 PACKAGE INCLUDES:</p>
              <ul className="text-xs text-gray-300 space-y-2">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                  <code className="bg-black/40 px-2 py-1 rounded text-green-400 font-mono">adapter_model.bin</code>
                  <span className="text-gray-500">- LoRA adapter weights (~16MB)</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                  <code className="bg-black/40 px-2 py-1 rounded text-green-400 font-mono">adapter_config.json</code>
                  <span className="text-gray-500">- LoRA configuration</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                  <code className="bg-black/40 px-2 py-1 rounded text-green-400 font-mono">config.json</code>
                  <span className="text-gray-500">- Training configuration</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                  <code className="bg-black/40 px-2 py-1 rounded text-green-400 font-mono">tokenizer files</code>
                  <span className="text-gray-500">- Model tokenizer</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                  <code className="bg-black/40 px-2 py-1 rounded text-green-400 font-mono">training_metrics.json</code>
                  <span className="text-gray-500">- Training statistics</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Metrics Section */}
      {(evalMetrics || isLoadingEval) && (
        <div className="relative bg-gradient-to-br from-purple-500/20 to-purple-500/5 border border-purple-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm hover:border-purple-500/40 transition-all duration-300">
          <div className="absolute top-0 right-0 w-40 h-40 bg-purple-500/10 rounded-full blur-3xl"></div>
          <div className="relative">
            <h3 className="font-bold text-white mb-6 flex items-center gap-3 text-2xl">
              <BarChart3 className="w-7 h-7 text-purple-400" />
              Evaluation Metrics
            </h3>
            
            {isLoadingEval ? (
              <div className="flex items-center gap-3 text-gray-400">
                <div className="w-5 h-5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                Loading evaluation results...
              </div>
            ) : evalMetrics ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-black/30 rounded-xl p-4 border border-purple-500/20">
                  <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Perplexity</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {evalMetrics.perplexity?.toFixed(2) || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Lower is better</div>
                </div>
                <div className="bg-black/30 rounded-xl p-4 border border-purple-500/20">
                  <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Final Loss</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {evalMetrics.final_loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 rounded-xl p-4 border border-purple-500/20">
                  <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Training Time</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {evalMetrics.training_time_seconds 
                      ? `${Math.floor(evalMetrics.training_time_seconds / 60)}m ${Math.floor(evalMetrics.training_time_seconds % 60)}s`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 rounded-xl p-4 border border-purple-500/20">
                  <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Peak Memory</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {evalMetrics.peak_memory_mb 
                      ? `${(evalMetrics.peak_memory_mb / 1024).toFixed(2)} GB`
                      : 'N/A'}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Model Card Section */}
      {modelCard && (
        <div className="relative bg-gradient-to-br from-blue-500/20 to-blue-500/5 border border-blue-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm hover:border-blue-500/40 transition-all duration-300">
          <div className="absolute top-0 right-0 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl"></div>
          <div className="relative">
            <h3 className="font-bold text-white mb-6 flex items-center gap-3 text-2xl">
              <FileCheck className="w-7 h-7 text-blue-400" />
              Model Card
            </h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center gap-3 text-sm">
                  <Tag className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-400">Model Name:</span>
                  <span className="text-white font-mono">{modelCard.model_name || 'N/A'}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-400">Base Model:</span>
                  <span className="text-white font-mono">{modelCard.base_model || 'N/A'}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Clock className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-400">Created:</span>
                  <span className="text-white">{modelCard.created_at 
                    ? new Date(modelCard.created_at).toLocaleString()
                    : 'N/A'}</span>
                </div>
                {modelCard.language && (
                  <div className="flex items-center gap-3 text-sm">
                    <User className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">Language:</span>
                    <span className="text-white">{modelCard.language}</span>
                  </div>
                )}
              </div>
              
              <div className="space-y-4">
                {modelCard.tags && modelCard.tags.length > 0 && (
                  <div>
                    <div className="text-xs text-gray-400 mb-2 uppercase tracking-wider">Tags</div>
                    <div className="flex flex-wrap gap-2">
                      {modelCard.tags.map((tag: string, idx: number) => (
                        <span key={idx} className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-lg border border-blue-500/30">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {modelCard.library_name && (
                  <div className="flex items-center gap-3 text-sm">
                    <Code className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">Library:</span>
                    <span className="text-white">{modelCard.library_name}</span>
                  </div>
                )}
              </div>
            </div>
            
            {modelCard.description && (
              <div className="mt-6 pt-6 border-t border-blue-500/20">
                <div className="text-xs text-gray-400 mb-2 uppercase tracking-wider">Description</div>
                <p className="text-gray-300 text-sm leading-relaxed">{modelCard.description}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Experiment Metadata Section */}
      {experimentMetadata && (
        <div className="relative bg-gradient-to-br from-orange-500/20 to-orange-500/5 border border-orange-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm hover:border-orange-500/40 transition-all duration-300">
          <div className="absolute top-0 right-0 w-40 h-40 bg-orange-500/10 rounded-full blur-3xl"></div>
          <div className="relative">
            <h3 className="font-bold text-white mb-6 flex items-center gap-3 text-2xl">
              <FileText className="w-7 h-7 text-orange-400" />
              Experiment Metadata
            </h3>
            
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-black/30 rounded-xl p-4 border border-orange-500/20">
                <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Experiment ID</div>
                <div className="text-sm font-mono text-orange-400 break-all">
                  {experimentMetadata.experiment_id || trainingJobId}
                </div>
              </div>
              <div className="bg-black/30 rounded-xl p-4 border border-orange-500/20">
                <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Seed</div>
                <div className="text-2xl font-bold text-orange-400">
                  {experimentMetadata.seed ?? trainingConfig?.seed ?? 42}
                </div>
              </div>
              <div className="bg-black/30 rounded-xl p-4 border border-orange-500/20">
                <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Status</div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                  <span className="text-green-400 font-bold">Completed</span>
                </div>
              </div>
            </div>

            {experimentMetadata.artifacts && Object.keys(experimentMetadata.artifacts).length > 0 && (
              <div className="mt-6 pt-6 border-t border-orange-500/20">
                <div className="text-xs text-gray-400 mb-3 uppercase tracking-wider">Experiment Artifacts</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(experimentMetadata.artifacts).map(([name, path]) => (
                    <a
                      key={name}
                      href={`http://localhost:8000/api/experiment/${trainingJobId}/artifact/${name}`}
                      download
                      className="px-3 py-2 bg-orange-500/20 text-orange-400 text-xs rounded-lg border border-orange-500/30 hover:bg-orange-500/30 transition-colors flex items-center gap-2"
                    >
                      <Download className="w-3 h-3" />
                      {name}
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Code Type Selection */}
      <div>
        <h3 className="text-sm font-bold text-green-400 mb-5 tracking-wider">SELECT CODE TYPE</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {codeTypes.map((type) => {
            const Icon = type.icon;
            return (
              <button
                key={type.id}
                onClick={() => setSelectedType(type.id)}
                className={`group relative p-6 rounded-xl border-2 text-left transition-all duration-300 hover:scale-105 backdrop-blur-sm ${
                  selectedType === type.id
                    ? 'border-green-500 bg-green-500/10 shadow-lg shadow-green-500/20'
                    : 'border-white/10 hover:border-green-500/30 bg-gradient-to-br from-white/5 to-white/[0.02]'
                }`}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 transition-all duration-300 ${
                    selectedType === type.id ? 'bg-green-500/20' : 'bg-white/5 group-hover:bg-green-500/10'
                  }`}>
                    <Icon className={`w-6 h-6 transition-colors ${
                      selectedType === type.id ? 'text-green-400' : 'text-gray-400 group-hover:text-green-400'
                    }`} />
                  </div>
                  <div className="font-bold text-white mb-2 text-lg group-hover:text-green-50 transition-colors">{type.label}</div>
                  <div className="text-xs text-gray-400">{type.description}</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        className="w-full py-5 bg-green-600 text-white rounded-xl hover:bg-green-500 font-bold text-xl transition-all duration-300 flex items-center justify-center gap-3 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20"
      >
        <Code className="w-6 h-6" />
        Generate {codeTypes.find(t => t.id === selectedType)?.label}
      </button>

      {/* Generated Code */}
      {generatedCode && (
        <div className="bg-black border-2 border-white/20 rounded-2xl overflow-hidden backdrop-blur-sm hover:border-green-500/30 transition-all duration-300 animate-fadeIn shadow-xl">
          <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-white/5 to-white/[0.02] border-b border-white/10">
            <span className="text-sm text-gray-300 font-mono font-bold flex items-center gap-2">
              <Code className="w-4 h-4 text-green-400" />
              {selectedType}.{selectedType === 'readme' ? 'md' : 'py'}
            </span>
            <button
              onClick={handleDownload}
              className="px-5 py-2.5 bg-green-600 text-white rounded-lg text-sm hover:bg-green-500 flex items-center gap-2 transition-all duration-300 hover:glow font-bold hover:scale-105"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
          <pre className="p-6 overflow-x-auto text-sm max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-green-500/20 scrollbar-track-white/5">
            <code className="text-gray-200 leading-relaxed">{generatedCode}</code>
          </pre>
        </div>
      )}

      {/* Summary */}
      <div className="relative bg-gradient-to-br from-green-500/20 to-green-500/5 border border-green-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm">
        <div className="absolute top-0 right-0 w-40 h-40 bg-green-500/10 rounded-full blur-3xl"></div>
        <div className="relative">
          <h3 className="font-bold text-white mb-5 flex items-center gap-3 text-2xl">
            <Sparkles className="w-7 h-7 text-green-400 animate-pulse" />
            🎉 Congratulations!
          </h3>
          <div className="space-y-3 text-sm text-gray-300 mb-6 leading-relaxed">
            <p className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
              <span>Model fine-tuned with QLoRA (4-bit quantization)</span>
            </p>
            <p className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
              <span>75% memory savings achieved</span>
            </p>
            <p className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
              <span>Ready for inference and deployment</span>
            </p>
          </div>
          <div className="pt-5 border-t border-green-500/20">
            <p className="font-bold text-white mb-3 flex items-center gap-2">
              <Rocket className="w-5 h-5 text-green-400" />
              Next Steps:
            </p>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-400 flex-shrink-0">1.</span>
                <span>Download your inference code</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 flex-shrink-0">2.</span>
                <span>Test the model with your use cases</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 flex-shrink-0">3.</span>
                <span>Deploy using Gradio or FastAPI</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 flex-shrink-0">4.</span>
                <span>Share your model on Hugging Face Hub</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
