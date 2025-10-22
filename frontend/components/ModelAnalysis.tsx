"use client";

import { useState } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import { useModelAnalysis } from '@/hooks/useModelAnalysis';
import { Loader2, Search, CheckCircle, AlertCircle, Info, Sparkles } from 'lucide-react';

export default function ModelAnalysis() {
  const [modelId, setModelId] = useState('');
  const { setModelInfo, setCurrentStep, modelInfo } = usePipelineStore();
  const { analyzeModel, isLoading, error } = useModelAnalysis();

  const handleAnalyze = async () => {
    if (!modelId.trim()) return;
    
    const result = await analyzeModel(modelId);
    if (result) {
      setModelInfo(result);
    }
  };

  const handleNext = () => {
    if (modelInfo) {
      setCurrentStep('dataset');
    }
  };

  const popularModels = [
    { id: 'meta-llama/Llama-2-7b-hf', name: 'Llama 2 7B', size: '7B', vram: '~7GB (QLoRA)' },
    { id: 'meta-llama/Llama-2-13b-hf', name: 'Llama 2 13B', size: '13B', vram: '~13GB (QLoRA)' },
    { id: 'mistralai/Mistral-7B-v0.1', name: 'Mistral 7B', size: '7B', vram: '~7GB (QLoRA)' },
    { id: 'tiiuae/falcon-7b', name: 'Falcon 7B', size: '7B', vram: '~7GB (QLoRA)' },
    { id: 'gpt2', name: 'GPT-2', size: '124M', vram: '~1GB' },
    { id: 'gpt2-medium', name: 'GPT-2 Medium', size: '355M', vram: '~2GB' },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="text-center">
        <div className="inline-block mb-3">
          <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold tracking-wider">
            STEP 1 OF 5
          </div>
        </div>
        <h2 className="text-4xl font-bold gradient-text mb-3">Model Analysis</h2>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Enter a Hugging Face model ID to analyze its architecture and compute requirements.
        </p>
      </div>

      {/* Model Input */}
      <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-8 backdrop-blur-sm hover:border-green-500/20 transition-all duration-300">
        <label htmlFor="modelId" className="block text-sm font-bold text-green-400 mb-4 tracking-wide">
          HUGGING FACE MODEL ID
        </label>
        <div className="flex gap-3">
          <input
            id="modelId"
            type="text"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="e.g., meta-llama/Llama-2-7b-hf"
            className="flex-1 rounded-xl bg-black/50 border border-white/20 px-5 py-4 text-white placeholder-gray-500 focus:border-green-500 focus:ring-2 focus:ring-green-500/20 outline-none transition-all duration-300 text-lg"
            disabled={isLoading}
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
          />
          <button
            onClick={handleAnalyze}
            disabled={isLoading || !modelId.trim()}
            className="px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center gap-3 transition-all duration-300 font-bold text-lg hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>

      {/* Popular Models */}
      <div>
        <h3 className="text-sm font-bold text-green-400 mb-5 tracking-wider">POPULAR MODELS</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {popularModels.map((model) => (
            <button
              key={model.id}
              onClick={() => setModelId(model.id)}
              className="group relative text-left p-6 bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-xl hover:border-green-500/50 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-green-500/10 backdrop-blur-sm"
              disabled={isLoading}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="font-bold text-white group-hover:text-green-400 transition-colors mb-2 text-lg">{model.name}</div>
                <div className="text-sm text-gray-400 mb-3 flex items-center gap-2">
                  <span className="px-2 py-1 rounded-md bg-green-500/10 text-green-400 font-semibold">{model.size}</span>
                  <span>•</span>
                  <span>{model.vram}</span>
                </div>
                <div className="text-xs text-gray-500 font-mono truncate bg-black/30 px-3 py-2 rounded-lg">{model.id}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 flex items-start gap-4 backdrop-blur-sm animate-fadeIn">
          <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h4 className="font-bold text-red-400 mb-1 text-lg">Analysis Failed</h4>
            <p className="text-sm text-red-300">{error}</p>
          </div>
        </div>
      )}

      {/* Model Info */}
      {modelInfo && (
        <div className="bg-gradient-to-br from-green-500/20 to-green-500/5 border border-green-500/30 rounded-2xl p-8 space-y-6 glow-strong backdrop-blur-sm animate-fadeIn shadow-xl shadow-green-500/20">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 bg-green-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-white text-2xl mb-2 flex items-center gap-2">
                Model Analysis Complete
                <Sparkles className="w-6 h-6 text-green-400 animate-pulse" />
              </h3>
              <p className="text-sm text-green-300 font-mono bg-black/30 inline-block px-3 py-1.5 rounded-lg">
                {modelInfo.model_id} • {modelInfo.architecture}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-green-500/20">
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">PARAMETERS</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {(modelInfo.num_parameters / 1e9).toFixed(1)}B
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">HIDDEN SIZE</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {modelInfo.hidden_size}
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">LAYERS</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {modelInfo.num_layers}
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">QLORA VRAM</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                ~{Math.ceil(modelInfo.num_parameters * 0.55 / 1e9)}GB
              </div>
            </div>
          </div>

          {/* Compute Requirements */}
          {modelInfo.compute_requirements && (
            <div className="pt-6 border-t border-green-500/20">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2 text-lg">
                <Info className="w-5 h-5 text-green-400" />
                QLoRA Memory Requirements
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-black/40 rounded-xl p-5 border border-white/10">
                  <div className="text-xs text-gray-400 mb-2 font-bold tracking-wider">MINIMUM (BATCH 1)</div>
                  <div className="font-bold text-white text-xl">
                    {modelInfo.compute_requirements.min_vram_gb}GB VRAM
                  </div>
                </div>
                <div className="bg-black/40 rounded-xl p-5 border border-white/10">
                  <div className="text-xs text-gray-400 mb-2 font-bold tracking-wider">RECOMMENDED (BATCH 4)</div>
                  <div className="font-bold text-white text-xl">
                    {modelInfo.compute_requirements.recommended_vram_gb}GB VRAM
                  </div>
                </div>
                <div className="bg-black/40 rounded-xl p-5 border border-white/10">
                  <div className="text-xs text-gray-400 mb-2 font-bold tracking-wider">MAX BATCH SIZE</div>
                  <div className="font-bold text-white text-xl">
                    {modelInfo.compute_requirements.max_batch_size}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Next Button */}
      {modelInfo && (
        <div className="flex justify-end pt-4">
          <button
            onClick={handleNext}
            className="group px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 font-bold text-lg transition-all duration-300 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20 flex items-center gap-2"
          >
            Continue to Dataset Upload
            <span className="group-hover:translate-x-1 transition-transform">→</span>
          </button>
        </div>
      )}
    </div>
  );
}
