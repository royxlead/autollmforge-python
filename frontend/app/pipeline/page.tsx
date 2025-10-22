"use client";

import { useEffect } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import ModelAnalysis from '@/components/ModelAnalysis';
import DatasetUpload from '@/components/DatasetUpload';
import HyperparameterTuning from '@/components/HyperparameterTuning';
import Training from '@/components/Training';
import CodeGeneration from '@/components/CodeGeneration';
import { CheckCircle2, Circle } from 'lucide-react';

export default function PipelinePage() {
  const { currentStep, setCurrentStep, reset } = usePipelineStore();

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

  const currentStepIndex = steps.findIndex(s => s.id === currentStep);

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-green-900/10 via-black to-black"></div>
      
      <div className="relative container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold gradient-text mb-2">
            QLoRA Fine-Tuning Pipeline
          </h1>
          <p className="text-gray-400 text-lg">
            Fine-tune large language models with 4-bit quantization
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8 bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8">
          <div className="flex items-center justify-between">
            {steps.map((item, index) => (
              <div key={item.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <button
                    onClick={() => setCurrentStep(item.id)}
                    className={`w-12 h-12 rounded-xl flex items-center justify-center border-2 transition-all font-semibold relative ${
                      currentStepIndex >= index
                        ? 'bg-green-600 border-green-500 text-white hover:bg-green-500'
                        : 'bg-white/5 border-white/20 text-gray-500 hover:border-white/30'
                    }`}
                  >
                    {currentStepIndex > index ? (
                      <CheckCircle2 className="w-6 h-6" />
                    ) : (
                      <span>{item.number}</span>
                    )}
                    {currentStepIndex === index && (
                      <div className="absolute inset-0 rounded-xl bg-green-500 animate-ping opacity-20"></div>
                    )}
                  </button>
                  <span
                    className={`mt-3 text-sm font-medium transition-colors ${
                      currentStepIndex >= index ? 'text-white' : 'text-gray-500'
                    }`}
                  >
                    {item.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className="flex-1 h-0.5 -mt-12 mx-4">
                    <div
                      className={`h-full transition-all rounded ${
                        currentStepIndex > index ? 'bg-green-600' : 'bg-white/20'
                      }`}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8 min-h-[600px]">
          {currentStep === 'model' && <ModelAnalysis />}
          {currentStep === 'dataset' && <DatasetUpload />}
          {currentStep === 'hyperparameters' && <HyperparameterTuning />}
          {currentStep === 'training' && <Training />}
          {currentStep === 'export' && <CodeGeneration />}
        </div>

        {/* QLoRA Info Banner */}
        <div className="mt-6 bg-gradient-to-r from-green-500/10 to-transparent border border-green-500/20 rounded-xl p-5">
          <div className="flex items-start gap-3">
            <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-green-400 mb-1">QLoRA Optimization Enabled</h3>
              <p className="text-sm text-gray-400">
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
