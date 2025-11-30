"use client";

import { useState, useEffect } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import type { TrainingProgress } from '@/types';
import { Play, CheckCircle, AlertCircle, Zap, Activity, Cpu, BarChart3, Download } from 'lucide-react';

export default function Training() {
  const { trainingConfig, setTrainingJobId, setTrainingProgress, setCurrentStep, trainingJobId, trainingProgress } = usePipelineStore();
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (!trainingJobId) return;

    const pollProgress = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/training-progress/${trainingJobId}`);
        if (!response.ok) return;
        
        const progress: TrainingProgress = await response.json();
        setTrainingProgress(progress);
        
        console.log('Training progress:', progress);
      } catch (err) {
        console.error('Failed to fetch progress:', err);
      }
    };

    pollProgress();
    
    const interval = setInterval(pollProgress, 2000);
    
    return () => clearInterval(interval);
  }, [trainingJobId, setTrainingProgress]);

  const handleStartTraining = async () => {
    if (!trainingConfig) return;

    setIsStarting(true);
    setError(null);

    try {
      console.log('Starting training with config:', trainingConfig);
      
      const response = await fetch('http://localhost:8000/api/start-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          config: trainingConfig,
          job_name: `Training-${Date.now()}`
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Training start failed:', errorText);
        throw new Error(`Failed to start training: ${response.status}`);
      }

      const data = await response.json();
      console.log('Training started, job ID:', data.job_id);
      setTrainingJobId(data.job_id);
    } catch (err: any) {
      console.error('Error starting training:', err);
      setError(err.message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleBack = () => {
    setCurrentStep('hyperparameters');
  };

  const handleNext = () => {
    setCurrentStep('export');
  };

  const isTraining = trainingProgress?.status === 'running';
  const isCompleted = trainingProgress?.status === 'completed';
  const isFailed = trainingProgress?.status === 'failed';

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="text-center">
        <div className="inline-block mb-3">
          <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold tracking-wider">
            STEP 4 OF 5
          </div>
        </div>
        <h2 className="text-4xl font-bold gradient-text mb-3">Training</h2>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Monitor your QLoRA fine-tuning progress in real-time
        </p>
      </div>

      {!trainingJobId && (
        <div className="relative bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-12 text-center overflow-hidden backdrop-blur-sm">
          <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-green-500/5 rounded-full blur-3xl"></div>
          
          <div className="relative">
            <div className="mb-10">
              <div className="w-24 h-24 bg-gradient-to-br from-green-600 to-green-500 rounded-2xl flex items-center justify-center mx-auto mb-6 glow-strong shadow-xl shadow-green-500/30 hover:scale-110 transition-transform duration-300">
                <Play className="w-12 h-12 text-white" />
              </div>
              <h3 className="text-3xl font-bold text-white mb-4">
                Ready to Start Training
              </h3>
            </div>
            
            <p className="text-gray-300 mb-4 text-xl">
              Your model will be fine-tuned using QLoRA with 4-bit quantization for optimal memory efficiency.
            </p>
            
            <div className="inline-block text-left mb-8 bg-white/5 rounded-xl p-6 border border-white/10">
              <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span className="text-gray-400">Model:</span>
                  <span className="text-white font-semibold">{trainingConfig?.model_id?.split('/').pop() || 'Not configured'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span className="text-gray-400">Epochs:</span>
                  <span className="text-white font-semibold">{trainingConfig?.num_epochs || 3}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span className="text-gray-400">Dataset:</span>
                  <span className="text-white font-semibold">{trainingConfig?.dataset_id?.split('-')[0] || 'Not configured'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span className="text-gray-400">Learning Rate:</span>
                  <span className="text-white font-semibold">{trainingConfig?.learning_rate || 0.0002}</span>
                </div>
              </div>
            </div>
            
            <div className="mb-10 p-6 bg-green-500/10 border border-green-500/30 rounded-xl text-left max-w-2xl mx-auto backdrop-blur-sm">
              <p className="text-sm font-bold text-green-400 mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                📋 Dataset Format Requirements:
              </p>
              <p className="text-xs text-gray-300 mb-3">Your JSON dataset file must have a "text" field for each entry:</p>
              <pre className="text-xs bg-black/50 p-4 rounded-lg border border-white/20 text-gray-300 overflow-x-auto">
{`[
  {"text": "Your first training text..."},
  {"text": "Your second training text..."},
  {"text": "Your third training text..."}
]`}
              </pre>
            </div>
            
            <button
              onClick={handleStartTraining}
              disabled={isStarting || !trainingConfig}
              className="px-12 py-5 bg-green-600 text-white rounded-xl hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center gap-3 mx-auto font-bold text-xl transition-all duration-300 hover:glow-strong hover:scale-105 shadow-xl shadow-green-500/30"
            >
              {isStarting ? (
                <>
                  <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Starting Training...
                </>
              ) : (
                <>
                  <Play className="w-6 h-6" />
                  Start Training Now
                </>
              )}
            </button>
            {!trainingConfig && (
              <p className="text-sm text-red-400 mt-5 font-semibold">
                ⚠️ Please configure hyperparameters first
              </p>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 flex items-start gap-4 backdrop-blur-sm animate-fadeIn">
          <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h4 className="font-bold text-red-400 text-lg mb-1">Training Failed to Start</h4>
            <p className="text-sm text-red-300">{error}</p>
          </div>
        </div>
      )}

      {trainingJobId && trainingProgress && (
        <div className="space-y-6 animate-fadeIn">
          <div className={`border-2 rounded-2xl p-8 backdrop-blur-sm transition-all duration-300 ${
            isCompleted ? 'bg-gradient-to-br from-green-500/20 to-green-500/5 border-green-500/40 glow-strong shadow-xl shadow-green-500/30' :
            isFailed ? 'bg-red-500/10 border-red-500/30' :
            'bg-gradient-to-br from-white/5 to-white/[0.02] border-white/10'
          }`}>
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-5">
                {isCompleted ? (
                  <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center shadow-lg shadow-green-500/30">
                    <CheckCircle className="w-9 h-9 text-green-400" />
                  </div>
                ) : isFailed ? (
                  <div className="w-16 h-16 bg-red-500/20 rounded-xl flex items-center justify-center">
                    <AlertCircle className="w-9 h-9 text-red-400" />
                  </div>
                ) : (
                  <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center animate-pulse">
                    <Zap className="w-9 h-9 text-green-400" />
                  </div>
                )}
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    {isCompleted ? '🎉 Training Completed' :
                     isFailed ? '❌ Training Failed' :
                     '⚡ Training in Progress'}
                  </h3>
                  <p className="text-sm text-gray-400 font-mono bg-black/30 inline-block px-3 py-1 rounded-lg">
                    Job ID: {trainingJobId.slice(0, 16)}...
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-4xl font-bold text-white mb-1">
                  {trainingProgress.current_step} / {trainingProgress.total_steps}
                </div>
                <div className="text-sm text-gray-400 font-semibold">steps</div>
              </div>
            </div>

            {trainingProgress.progress_message && (
              <div className={`mb-6 p-5 rounded-xl border transition-all duration-300 ${
                isFailed ? 'bg-red-500/10 border-red-500/30' : 'bg-green-500/10 border-green-500/30'
              }`}>
                <p className={`text-sm font-bold flex items-center gap-3 ${
                  isFailed ? 'text-red-400' : 'text-green-400'
                }`}>
                  <div className={`w-2.5 h-2.5 rounded-full ${
                    isFailed ? 'bg-red-400' : 'bg-green-400 animate-pulse'
                  }`} />
                  {trainingProgress.progress_message}
                </p>
              </div>
            )}

            {isFailed && trainingProgress.error_message && (
              <div className="mb-6 p-5 bg-red-500/10 border border-red-500/30 rounded-xl">
                <p className="text-sm font-bold text-red-400 mb-2">Error Details:</p>
                <p className="text-sm text-red-300 font-mono whitespace-pre-wrap bg-black/30 p-3 rounded-lg">
                  {trainingProgress.error_message}
                </p>
              </div>
            )}

            <div className="mb-8">
              <div className="flex justify-between text-sm text-gray-300 mb-4 font-bold">
                <span>Overall Progress</span>
                <span className="text-lg text-white">{Math.round((trainingProgress.current_step / trainingProgress.total_steps) * 100)}%</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-5 overflow-hidden shadow-inner">
                <div
                  className={`h-5 rounded-full transition-all duration-500 relative overflow-hidden ${
                    isCompleted ? 'bg-gradient-to-r from-green-600 via-green-500 to-green-400' :
                    isFailed ? 'bg-gradient-to-r from-red-600 to-red-400' :
                    'bg-gradient-to-r from-green-600 via-green-500 to-green-400'
                  }`}
                  style={{ width: `${(trainingProgress.current_step / trainingProgress.total_steps) * 100}%` }}
                >
                  {!isCompleted && !isFailed && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                  )}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">EPOCH</div>
                <div className="text-2xl font-bold text-white group-hover:text-green-50 transition-colors">
                  {trainingProgress.current_epoch}
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">LOSS</div>
                <div className="text-2xl font-bold text-white group-hover:text-green-50 transition-colors">
                  {trainingProgress.train_loss?.toFixed(4) || 'N/A'}
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">LEARNING RATE</div>
                <div className="text-2xl font-bold text-white group-hover:text-green-50 transition-colors">
                  {trainingProgress.learning_rate?.toExponential(2) || 'N/A'}
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">GRAD NORM</div>
                <div className="text-2xl font-bold text-white group-hover:text-green-50 transition-colors">
                  {trainingProgress.latest_metrics?.grad_norm?.toFixed(4) || 'N/A'}
                </div>
              </div>
            </div>

            {/* Memory Profiling Section */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-purple-400 mb-2 tracking-wider flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> GPU MEMORY
                </div>
                <div className="text-2xl font-bold text-white group-hover:text-purple-50 transition-colors">
                  {trainingProgress.gpu_memory_usage ? `${trainingProgress.gpu_memory_usage.toFixed(1)}GB` : 'N/A'}
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-purple-400 mb-2 tracking-wider flex items-center gap-1">
                  <Activity className="w-3 h-3" /> GPU UTIL
                </div>
                <div className="text-2xl font-bold text-white group-hover:text-purple-50 transition-colors">
                  {trainingProgress.gpu_utilization ? `${trainingProgress.gpu_utilization.toFixed(0)}%` : 'N/A'}
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-purple-400 mb-2 tracking-wider flex items-center gap-1">
                  <BarChart3 className="w-3 h-3" /> THROUGHPUT
                </div>
                <div className="text-2xl font-bold text-white group-hover:text-purple-50 transition-colors">
                  {trainingProgress.samples_per_second?.toFixed(1) || 'N/A'}
                  <span className="text-sm text-gray-400 ml-1">s/s</span>
                </div>
              </div>
              <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-purple-500/30 transition-all duration-300 hover:scale-105">
                <div className="text-xs font-bold text-purple-400 mb-2 tracking-wider">VAL LOSS</div>
                <div className="text-2xl font-bold text-white group-hover:text-purple-50 transition-colors">
                  {trainingProgress.val_loss?.toFixed(4) || 'N/A'}
                </div>
              </div>
            </div>

            {trainingProgress.eta_seconds !== undefined && (
              <div className="mt-6 pt-6 border-t border-white/10">
                <div className="flex justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-green-400" />
                    <span className="text-gray-400">Speed:</span>
                    <span className="text-white font-bold">{trainingProgress.samples_per_second.toFixed(2)} samples/sec</span>
                  </div>
                  {trainingProgress.eta_seconds > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">ETA:</span>
                      <span className="text-white font-bold">{Math.floor(trainingProgress.eta_seconds / 60)} minutes</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Experiment Data Download - Show when complete */}
            {isCompleted && (
              <div className="mt-6 pt-6 border-t border-white/10">
                <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-green-400" />
                  Experiment Artifacts
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <a
                    href={`http://localhost:8000/api/download-file/${trainingJobId}/training_metrics.json`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-3 bg-black/40 rounded-lg border border-white/10 hover:border-green-500/30 transition-all text-sm text-gray-300 hover:text-white"
                  >
                    <Download className="w-4 h-4" />
                    Metrics JSON
                  </a>
                  <a
                    href={`http://localhost:8000/storage/experiments/${trainingJobId}/loss.png`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-3 bg-black/40 rounded-lg border border-white/10 hover:border-green-500/30 transition-all text-sm text-gray-300 hover:text-white"
                  >
                    <BarChart3 className="w-4 h-4" />
                    Loss Graph
                  </a>
                  <a
                    href={`http://localhost:8000/storage/experiments/${trainingJobId}/metadata.json`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-3 bg-black/40 rounded-lg border border-white/10 hover:border-purple-500/30 transition-all text-sm text-gray-300 hover:text-white"
                  >
                    <Activity className="w-4 h-4" />
                    Experiment Config
                  </a>
                  <a
                    href={`http://localhost:8000/api/download-model/${trainingJobId}`}
                    className="flex items-center gap-2 px-4 py-3 bg-green-600/20 rounded-lg border border-green-500/30 hover:bg-green-600/30 transition-all text-sm text-green-300 hover:text-green-200 font-semibold"
                  >
                    <Download className="w-4 h-4" />
                    Download Model
                  </a>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button
          onClick={() => setCurrentStep('hyperparameters')}
          disabled={isTraining}
          className="px-8 py-4 border-2 border-white/20 text-gray-300 rounded-xl hover:bg-white/5 hover:border-white/30 font-bold text-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← Back to Hyperparameters
        </button>
        {isCompleted && (
          <button
            onClick={handleNext}
            className="group px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 font-bold text-lg transition-all duration-300 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20 flex items-center gap-2"
          >
            Export Code
            <span className="group-hover:translate-x-1 transition-transform">→</span>
          </button>
        )}
      </div>
    </div>
  );
}
