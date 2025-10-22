"use client";

import { useEffect, useState } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import ModelAnalysis from '@/components/ModelAnalysis';
import DatasetUpload from '@/components/DatasetUpload';
import HyperparameterTuning from '@/components/HyperparameterTuning';
import Training from '@/components/Training';
import CodeGeneration from '@/components/CodeGeneration';
import { CheckCircle2, ArrowRight, Brain, Database, Sliders, Cpu, Code2, Sparkles, Zap, Shield } from 'lucide-react';

export default function Home() {
  const { currentStep, setCurrentStep, reset } = usePipelineStore();
  const [showWelcome, setShowWelcome] = useState(true);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);

  useEffect(() => {
    reset();
    console.log('🔄 Pipeline state reset - Fresh session started');
  }, [reset]);

  const steps = [
    { id: 'model' as const, label: 'Model Analysis', number: 1 },
    { id: 'dataset' as const, label: 'Dataset Upload', number: 2 },
    { id: 'hyperparameters' as const, label: 'Hyperparameters', number: 3 },
    { id: 'training' as const, label: 'Training', number: 4 },
    { id: 'export' as const, label: 'Export Code', number: 5 },
  ];

  const pipelineSteps = [
    {
      icon: Brain,
      number: 1,
      title: "Model Analysis",
      description: "Choose your base model from popular LLMs like GPT-2, Llama, or Mistral. Our analyzer automatically detects architecture details and calculates optimal memory requirements.",
      highlight: "Automatic VRAM calculation with QLoRA optimization"
    },
    {
      icon: Database,
      number: 2,
      title: "Dataset Upload",
      description: "Upload your training data in JSONL format. We validate the structure, calculate statistics, and prepare your dataset for efficient fine-tuning.",
      highlight: "Smart validation with token length analysis"
    },
    {
      icon: Sliders,
      number: 3,
      title: "Hyperparameter Tuning",
      description: "Get AI-powered hyperparameter recommendations tailored to your model size. Our 8-tier optimization system adapts learning rates, batch sizes, and more.",
      highlight: "Intelligent defaults based on model parameters"
    },
    {
      icon: Cpu,
      number: 4,
      title: "Training Execution",
      description: "Monitor real-time training progress with live metrics. Track loss curves, learning rates, GPU memory usage, and estimated completion time.",
      highlight: "Real-time monitoring with automatic checkpointing"
    },
    {
      icon: Code2,
      number: 5,
      title: "Code Generation",
      description: "Export production-ready inference scripts, Gradio demos, API endpoints, and comprehensive documentation for your fine-tuned model.",
      highlight: "Deploy-ready code in multiple formats"
    }
  ];

  const features = [
    {
      icon: Zap,
      title: "QLoRA Optimization",
      description: "4-bit quantization reduces memory by 75%"
    },
    {
      icon: Shield,
      title: "Production Grade",
      description: "Enterprise-ready with full error handling"
    },
    {
      icon: Sparkles,
      title: "Auto-Configuration",
      description: "AI-powered hyperparameter optimization"
    }
  ];

  const currentStepIndex = steps.findIndex(s => s.id === currentStep);

  if (showWelcome) {
    return (
      <div className="min-h-screen bg-black text-white overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-green-900/20 via-black to-black"></div>
        <div className="absolute inset-0">
          <div className="absolute top-20 left-1/4 w-96 h-96 bg-green-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 right-1/4 w-96 h-96 bg-green-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        </div>
        
        <div className="relative container mx-auto px-4 py-12 max-w-7xl">
          {/* Header */}
          <div className="text-center mb-16 animate-fadeIn">
            <div className="inline-block mb-6">
              <div className="flex items-center justify-center space-x-3 px-6 py-3 rounded-full bg-green-500/10 border border-green-500/20 backdrop-blur-sm hover:bg-green-500/15 transition-all duration-300">
                <div className="text-3xl animate-pulse">⚒️</div>
                <span className="text-2xl font-bold gradient-text">AutoLLM Forge</span>
              </div>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="gradient-text drop-shadow-[0_0_30px_rgba(34,197,94,0.3)]">Forge Your</span>
              <br />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-white via-green-100 to-white">Perfect Model</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-400 mb-4 max-w-3xl mx-auto leading-relaxed">
              Fine-tune any large language model with intelligent QLoRA optimization
            </p>
            
            <div className="flex items-center justify-center gap-2 text-lg text-gray-500 max-w-2xl mx-auto">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <p>No PhD required. No complex setup. Just powerful AI customization.</p>
            </div>
          </div>

          {/* How It Works Section */}
          <div className="mb-16">
            <div className="text-center mb-12">
              <div className="inline-block mb-3">
                <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-sm font-semibold">
                  PIPELINE OVERVIEW
                </div>
              </div>
              <h2 className="text-4xl font-bold mb-3 gradient-text">How It Works</h2>
              <p className="text-gray-400 text-lg">Five simple steps to your custom AI model</p>
            </div>

            <div className="grid gap-6 max-w-5xl mx-auto">
              {pipelineSteps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <div
                    key={index}
                    onMouseEnter={() => setHoveredStep(index)}
                    onMouseLeave={() => setHoveredStep(null)}
                    className="relative group"
                  >
                    <div className={`flex gap-6 p-8 rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 transition-all duration-300 backdrop-blur-sm ${
                      hoveredStep === index ? 'border-green-500/50 shadow-xl shadow-green-500/10 scale-[1.02]' : 'hover:border-green-500/30'
                    }`}>
                      {/* Step Number Circle */}
                      <div className="flex-shrink-0">
                        <div className={`relative w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-all duration-300 ${
                          hoveredStep === index 
                            ? 'bg-green-600 border-green-500 shadow-lg shadow-green-500/50 scale-110' 
                            : 'bg-white/5 border-white/20'
                        }`}>
                          <Icon className={`w-8 h-8 transition-all duration-300 ${
                            hoveredStep === index ? 'text-white scale-110' : 'text-gray-400'
                          }`} />
                          {hoveredStep === index && (
                            <div className="absolute inset-0 rounded-xl bg-green-500 animate-ping opacity-20"></div>
                          )}
                        </div>
                      </div>

                      {/* Content */}
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <span className="text-xs font-bold text-green-400 tracking-wider">STEP {step.number}</span>
                          <div className="h-px flex-1 bg-gradient-to-r from-green-500/50 to-transparent"></div>
                        </div>
                        
                        <h3 className="text-2xl font-bold mb-3 group-hover:text-green-50 transition-colors">{step.title}</h3>
                        <p className="text-gray-400 leading-relaxed mb-4">{step.description}</p>
                        
                        <div className="flex items-center gap-2 text-sm p-3 rounded-lg bg-green-500/5 border border-green-500/10">
                          <Sparkles className="w-4 h-4 text-green-400 flex-shrink-0" />
                          <span className="text-green-400 font-medium">{step.highlight}</span>
                        </div>
                      </div>
                    </div>

                    {/* Connector Line */}
                    {index < pipelineSteps.length - 1 && (
                      <div className="flex justify-center py-2">
                        <div className="w-0.5 h-8 bg-gradient-to-b from-green-500/50 via-green-500/30 to-transparent rounded-full"></div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Features Grid */}
          <div className="mb-16">
            <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={index}
                    className="group relative p-8 rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-center hover:border-green-500/30 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-green-500/10 backdrop-blur-sm"
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    <div className="relative">
                      <div className="inline-flex p-4 rounded-xl bg-green-500/10 mb-4 group-hover:bg-green-500/20 transition-all duration-300 group-hover:scale-110">
                        <Icon className="w-7 h-7 text-green-400" />
                      </div>
                      <h3 className="text-xl font-bold mb-2 group-hover:text-green-50 transition-colors">{feature.title}</h3>
                      <p className="text-gray-400 text-sm leading-relaxed">{feature.description}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* CTA Section */}
          <div className="max-w-4xl mx-auto">
            <div className="relative bg-gradient-to-br from-green-500/10 to-transparent border border-green-500/20 rounded-3xl p-12 text-center overflow-hidden backdrop-blur-sm">
              {/* Animated Background Elements */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/10 rounded-full blur-3xl"></div>
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-green-500/5 rounded-full blur-3xl"></div>
              
              <div className="relative">
                <div className="inline-block mb-4">
                  <div className="px-4 py-1.5 rounded-full bg-green-500/20 border border-green-500/30 text-green-400 text-xs font-bold tracking-wider">
                    🚀 START YOUR JOURNEY
                  </div>
                </div>
                
                <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">Ready to Start Fine-Tuning?</h2>
                <p className="text-gray-400 text-xl mb-10 max-w-2xl mx-auto leading-relaxed">
                  Transform any LLM into your specialized AI assistant in minutes
                </p>
                
                <button
                  onClick={() => setShowWelcome(false)}
                  className="group relative inline-flex items-center gap-3 px-10 py-5 bg-green-600 hover:bg-green-500 rounded-xl font-bold text-xl transition-all duration-300 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20"
                >
                  <span>Get Started</span>
                  <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform duration-300" />
                  <div className="absolute inset-0 rounded-xl bg-green-400 opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-300"></div>
                </button>

                <div className="mt-8 flex flex-wrap items-center justify-center gap-6 text-sm">
                  <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-gray-400">7B models on <span className="text-white font-semibold">12GB VRAM</span></span>
                  </div>
                  <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-gray-400">13B models on <span className="text-white font-semibold">24GB VRAM</span></span>
                  </div>
                  <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-gray-400">70B models on <span className="text-white font-semibold">80GB VRAM</span></span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-20 pt-8 border-t border-white/10 text-center">
            <div className="flex items-center justify-center gap-2 text-gray-500 text-sm mb-2">
              <Zap className="w-4 h-4 text-green-500" />
              <span>Powered by QLoRA, FastAPI, and Next.js</span>
            </div>
            <p className="text-gray-600 text-xs">© 2025 AutoLLM Forge | Sourav Roy | royxlead</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-green-900/10 via-black to-black"></div>
      <div className="absolute inset-0">
        <div className="absolute top-0 left-1/3 w-96 h-96 bg-green-500/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="relative container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="text-2xl">⚒️</div>
              <h1 className="text-3xl font-bold gradient-text">AutoLLM Forge</h1>
            </div>
            <div className="text-sm text-gray-500">
              Enterprise LLM Fine-Tuning Platform
            </div>
          </div>
          <p className="text-gray-400 text-lg">
            Fine-tune large language models with 4-bit quantization
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8 bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            {steps.map((item, index) => (
              <div key={item.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <button
                    onClick={() => setCurrentStep(item.id)}
                    className={`w-14 h-14 rounded-xl flex items-center justify-center border-2 transition-all duration-300 font-semibold relative group ${
                      currentStepIndex >= index
                        ? 'bg-green-600 border-green-500 text-white hover:bg-green-500 shadow-lg shadow-green-500/30'
                        : 'bg-white/5 border-white/20 text-gray-500 hover:border-white/30 hover:bg-white/10'
                    }`}
                  >
                    {currentStepIndex > index ? (
                      <CheckCircle2 className="w-7 h-7" />
                    ) : (
                      <span className="text-lg">{item.number}</span>
                    )}
                    {currentStepIndex === index && (
                      <div className="absolute inset-0 rounded-xl bg-green-500 animate-ping opacity-20"></div>
                    )}
                  </button>
                  <span
                    className={`mt-3 text-sm font-semibold transition-colors ${
                      currentStepIndex >= index ? 'text-white' : 'text-gray-500'
                    }`}
                  >
                    {item.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className="flex-1 h-1 -mt-12 mx-4 rounded-full overflow-hidden bg-white/10">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        currentStepIndex > index ? 'bg-gradient-to-r from-green-600 to-green-500' : 'bg-transparent'
                      }`}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8 min-h-[600px] backdrop-blur-sm shadow-xl">
          {currentStep === 'model' && <ModelAnalysis />}
          {currentStep === 'dataset' && <DatasetUpload />}
          {currentStep === 'hyperparameters' && <HyperparameterTuning />}
          {currentStep === 'training' && <Training />}
          {currentStep === 'export' && <CodeGeneration />}
        </div>

        {/* QLoRA Info Banner */}
        <div className="mt-6 bg-gradient-to-r from-green-500/10 to-transparent border border-green-500/20 rounded-xl p-6 backdrop-blur-sm hover:border-green-500/30 transition-all duration-300">
          <div className="flex items-start gap-4">
            <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
              <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse"></div>
            </div>
            <div>
              <h3 className="text-sm font-bold text-green-400 mb-1.5 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                QLoRA Optimization Enabled
              </h3>
              <p className="text-sm text-gray-400 leading-relaxed">
                This pipeline uses QLoRA (4-bit quantization) by default, reducing memory usage by 75%. 
                Fine-tune 7B models on 12GB VRAM, 13B on 24GB, and 70B on 80GB GPUs.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
