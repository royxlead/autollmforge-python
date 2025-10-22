"use client";

import { useState, useEffect } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import { Loader2, Sparkles, Info, AlertCircle } from 'lucide-react';

export default function HyperparameterTuning() {
  const { modelInfo, datasetInfo, setRecommendations, setTrainingConfig, setCurrentStep, recommendations } = usePipelineStore();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState({
    learning_rate: 0.0002,
    batch_size: 4,
    gradient_accumulation_steps: 4,
    num_epochs: 3,
    max_seq_length: 512,
    warmup_steps: 100,
    save_steps: 500,
    eval_steps: 500,
    logging_steps: 10,
  });

  const parseNumber = (value: string, fallback: number): number => {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? fallback : parsed;
  };

  const parseIntSafe = (value: string, fallback: number): number => {
    const parsed = Number.parseInt(value);
    return isNaN(parsed) ? fallback : parsed;
  };

  const handleGetRecommendations = async () => {
    if (!modelInfo || !datasetInfo) return;

    setIsLoading(true);
    setError(null);

    try {
      console.log('Requesting recommendations for:', modelInfo.model_id);
      
      const response = await fetch('http://localhost:8000/api/recommend-hyperparameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelInfo.model_id,
          dataset_id: datasetInfo.dataset_id,
          compute_tier: 'basic',
          task_type: 'text-generation',
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', errorText);
        throw new Error(`Failed to get recommendations: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received recommendations:', data);
      
      setRecommendations(data);
      
      setConfig({
        learning_rate: data.config.learning_rate ?? 0.0002,
        batch_size: data.config.batch_size ?? 4,
        gradient_accumulation_steps: data.config.gradient_accumulation_steps ?? 4,
        num_epochs: data.config.num_epochs ?? 3,
        max_seq_length: data.config.max_seq_length ?? 512,
        warmup_steps: data.config.warmup_steps ?? 100,
        save_steps: data.config.save_steps ?? 500,
        eval_steps: data.config.eval_steps ?? 500,
        logging_steps: data.config.logging_steps ?? 10,
      });
      
      console.log('Config updated successfully');
    } catch (err: any) {
      console.error('Error getting recommendations:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveConfig = () => {
    if (!modelInfo || !datasetInfo) return;

    const trainingConfig = {
      model_id: modelInfo.model_id,
      dataset_id: datasetInfo.dataset_id,
      task_type: 'text-generation',
      ...config,
      use_lora: true,
      lora_config: {
        r: 8,
        lora_alpha: 16,
        lora_dropout: 0.05,
        target_modules: ['q_proj', 'v_proj'],
        bias: 'none',
        task_type: 'CAUSAL_LM',
      },
      load_in_4bit: true,
      quantization: '4bit',
      bnb_4bit_quant_type: 'nf4',
      bnb_4bit_use_double_quant: true,
      bnb_4bit_compute_dtype: 'float16',
      optimizer: 'paged_adamw_8bit',
    };

    setTrainingConfig(trainingConfig as any);
    setCurrentStep('training');
  };

  const handleBack = () => {
    setCurrentStep('dataset');
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="text-center">
        <div className="inline-block mb-3">
          <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold tracking-wider">
            STEP 3 OF 5
          </div>
        </div>
        <h2 className="text-4xl font-bold gradient-text mb-3">Hyperparameter Tuning</h2>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Get AI-powered recommendations or customize parameters for your training.
        </p>
      </div>

      {/* Get Recommendations Button */}
      {!recommendations && (
        <div className="relative bg-gradient-to-br from-green-500/20 to-green-500/5 border border-green-500/30 rounded-2xl p-8 overflow-hidden backdrop-blur-sm hover:border-green-500/40 transition-all duration-300">
          <div className="absolute top-0 right-0 w-40 h-40 bg-green-500/10 rounded-full blur-3xl"></div>
          <div className="relative flex items-start gap-6">
            <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
              <Sparkles className="w-8 h-8 text-green-400 animate-pulse" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-white mb-3 text-2xl">
                Get AI-Powered Recommendations
              </h3>
              <p className="text-sm text-gray-300 mb-6 leading-relaxed">
                Our AI will analyze your model and dataset to recommend optimal hyperparameters for QLoRA fine-tuning.
              </p>
              <button
                onClick={handleGetRecommendations}
                disabled={isLoading || !modelInfo || !datasetInfo}
                className="px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center gap-3 transition-all duration-300 hover:glow-strong font-bold text-lg hover:scale-105 shadow-lg shadow-green-500/20"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing model and dataset...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Get Recommendations
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Success Message */}
      {recommendations && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-5 flex items-start gap-4 backdrop-blur-sm animate-fadeIn">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <Sparkles className="w-6 h-6 text-green-400 animate-pulse" />
          </div>
          <div>
            <h4 className="font-bold text-green-400 text-lg mb-1">Recommendations Received!</h4>
            <p className="text-sm text-green-300">
              Configuration has been optimized for your model and dataset. Review and adjust below.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 flex items-start gap-4 backdrop-blur-sm animate-fadeIn">
          <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h4 className="font-bold text-red-400 text-lg mb-1">Failed to Get Recommendations</h4>
            <p className="text-sm text-red-300">{error}</p>
          </div>
        </div>
      )}

      <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8 space-y-8 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-2xl font-bold text-white">Training Configuration</h3>
          {recommendations && (
            <span className="text-sm text-green-400 flex items-center gap-2 font-bold px-4 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
              <Sparkles className="w-4 h-4 animate-pulse" />
              AI Recommended
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              LEARNING RATE
            </label>
            <input
              type="number"
              step="0.00001"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseNumber(e.target.value, config.learning_rate) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">Typical: 0.0001 - 0.0003</p>
          </div>

          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              BATCH SIZE
            </label>
            <input
              type="number"
              min="1"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseIntSafe(e.target.value, config.batch_size) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">Per device batch size</p>
          </div>

          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              GRADIENT ACCUMULATION STEPS
            </label>
            <input
              type="number"
              min="1"
              value={config.gradient_accumulation_steps}
              onChange={(e) => setConfig({ ...config, gradient_accumulation_steps: parseIntSafe(e.target.value, config.gradient_accumulation_steps) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">
              Effective batch = <span className="text-green-400 font-bold">{config.batch_size * config.gradient_accumulation_steps}</span>
            </p>
          </div>

          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              NUMBER OF EPOCHS
            </label>
            <input
              type="number"
              min="1"
              value={config.num_epochs}
              onChange={(e) => setConfig({ ...config, num_epochs: parseIntSafe(e.target.value, config.num_epochs) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">Full passes through dataset</p>
          </div>

          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              MAX SEQUENCE LENGTH
            </label>
            <input
              type="number"
              min="128"
              step="128"
              value={config.max_seq_length}
              onChange={(e) => setConfig({ ...config, max_seq_length: parseIntSafe(e.target.value, config.max_seq_length) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">Longer = more memory</p>
          </div>

          <div className="group">
            <label className="block text-sm font-bold text-green-400 mb-3 tracking-wide">
              WARMUP STEPS
            </label>
            <input
              type="number"
              min="0"
              value={config.warmup_steps}
              onChange={(e) => setConfig({ ...config, warmup_steps: parseIntSafe(e.target.value, config.warmup_steps) })}
              className="w-full rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg group-hover:border-white/30"
            />
            <p className="text-xs text-gray-500 mt-2 ml-1">Learning rate warmup</p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-500/10 to-transparent border border-green-500/30 rounded-xl p-6 backdrop-blur-sm">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Info className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <h4 className="font-bold text-green-400 mb-3 text-lg">QLoRA Configuration</h4>
              <div className="text-sm text-gray-300 space-y-2 leading-relaxed">
                <p>• <strong className="text-white">Quantization:</strong> 4-bit NormalFloat (NF4)</p>
                <p>• <strong className="text-white">Double Quantization:</strong> Enabled</p>
                <p>• <strong className="text-white">Compute Dtype:</strong> float16</p>
                <p>• <strong className="text-white">Optimizer:</strong> paged_adamw_8bit</p>
                <p>• <strong className="text-white">Memory Savings:</strong> <span className="text-green-400 font-bold">~75% reduction</span></p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={handleBack}
          className="px-8 py-4 border-2 border-white/20 text-gray-300 rounded-xl hover:bg-white/5 hover:border-white/30 font-bold text-lg transition-all duration-300"
        >
          ← Back to Dataset
        </button>
        <button
          onClick={handleSaveConfig}
          className="group px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 font-bold text-lg transition-all duration-300 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20 flex items-center gap-2"
        >
          Start Training
          <span className="group-hover:translate-x-1 transition-transform">→</span>
        </button>
      </div>
    </div>
  );
}
