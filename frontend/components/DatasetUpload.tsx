"use client";

import { useState, useCallback } from 'react';
import { usePipelineStore } from '@/store/pipelineStore';
import { useDropzone } from 'react-dropzone';
import { Upload, File, CheckCircle, AlertCircle, Loader2, Info, Sparkles } from 'lucide-react';

export default function DatasetUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setDatasetInfo, setCurrentStep, datasetInfo, modelInfo } = usePipelineStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/json': ['.json', '.jsonl'],
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
  });

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('format', file.name.endsWith('.csv') ? 'csv' : 'json');

      const response = await fetch('http://localhost:8000/api/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setDatasetInfo(data);
    } catch (err: any) {
      setError(err.message || 'Failed to upload dataset');
    } finally {
      setIsUploading(false);
    }
  };

  const handleNext = () => {
    if (datasetInfo) {
      setCurrentStep('hyperparameters');
    }
  };

  const handleBack = () => {
    setCurrentStep('model');
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="text-center">
        <div className="inline-block mb-3">
          <div className="px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold tracking-wider">
            STEP 2 OF 5
          </div>
        </div>
        <h2 className="text-4xl font-bold gradient-text mb-3">Dataset Upload</h2>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Upload your training dataset in JSON, JSONL, or CSV format.
        </p>
      </div>

      {/* Model Info Banner */}
      {modelInfo && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-5 backdrop-blur-sm hover:border-green-500/40 transition-all duration-300">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Info className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <span className="font-bold text-green-300 text-sm tracking-wide">SELECTED MODEL: </span>
              <span className="text-green-400 font-mono font-semibold">{modelInfo.model_id}</span>
            </div>
          </div>
        </div>
      )}

      {/* Upload Area */}
      {!datasetInfo && (
        <div
          {...getRootProps()}
          className={`group border-2 border-dashed rounded-2xl p-20 text-center cursor-pointer transition-all duration-300 backdrop-blur-sm ${
            isDragActive
              ? 'border-green-500 bg-green-500/20 scale-105 shadow-xl shadow-green-500/20'
              : 'border-white/20 hover:border-green-500/50 bg-gradient-to-br from-white/5 to-white/[0.02] hover:bg-white/10'
          }`}
        >
          <input {...getInputProps()} />
          <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center transition-all duration-300 ${
            isDragActive ? 'bg-green-500/30 scale-110' : 'bg-green-500/10 group-hover:bg-green-500/20'
          }`}>
            <Upload className={`w-10 h-10 transition-all duration-300 ${
              isDragActive ? 'text-green-400 scale-110' : 'text-gray-400 group-hover:text-green-400'
            }`} />
          </div>
          <p className="text-2xl font-bold text-white mb-3">
            {isDragActive ? 'Drop your dataset here' : 'Drag & drop your dataset'}
          </p>
          <p className="text-sm text-gray-400 mb-5">
            or click to browse your files
          </p>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <p className="text-xs text-gray-400">
              Supported: <span className="text-green-400 font-semibold">JSON, JSONL, CSV</span> (max 100MB)
            </p>
          </div>
        </div>
      )}

      {/* Selected File */}
      {file && !datasetInfo && (
        <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-6 backdrop-blur-sm hover:border-green-500/30 transition-all duration-300 animate-fadeIn">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-5">
              <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <File className="w-8 h-8 text-green-400" />
              </div>
              <div>
                <p className="font-bold text-white text-lg mb-1">{file.name}</p>
                <p className="text-sm text-gray-400 flex items-center gap-2">
                  <span className="px-2 py-1 rounded-md bg-green-500/10 text-green-400 font-semibold text-xs">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </p>
              </div>
            </div>
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center gap-3 transition-all duration-300 hover:glow-strong font-bold text-lg hover:scale-105 shadow-lg shadow-green-500/20"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                'Upload & Analyze'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 flex items-start gap-4 backdrop-blur-sm animate-fadeIn">
          <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h4 className="font-bold text-red-400 mb-1 text-lg">Upload Failed</h4>
            <p className="text-sm text-red-300">{error}</p>
          </div>
        </div>
      )}

      {/* Dataset Info */}
      {datasetInfo && (
        <div className="bg-gradient-to-br from-green-500/20 to-green-500/5 border border-green-500/30 rounded-2xl p-8 space-y-6 glow-strong backdrop-blur-sm animate-fadeIn shadow-xl shadow-green-500/20">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 bg-green-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-white text-2xl mb-2 flex items-center gap-2">
                Dataset Uploaded Successfully
                <Sparkles className="w-6 h-6 text-green-400 animate-pulse" />
              </h3>
              <p className="text-sm text-green-300 font-mono bg-black/30 inline-block px-3 py-1.5 rounded-lg">
                {datasetInfo.dataset_id}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-green-500/20">
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">TOTAL SAMPLES</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {datasetInfo.num_samples.toLocaleString()}
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">TRAINING SAMPLES</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {datasetInfo.num_train_samples.toLocaleString()}
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">AVG TOKENS</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {datasetInfo.avg_tokens}
              </div>
            </div>
            <div className="group bg-black/40 rounded-xl p-5 border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
              <div className="text-xs font-bold text-green-400 mb-2 tracking-wider">MAX TOKENS</div>
              <div className="text-3xl font-bold text-white group-hover:text-green-50 transition-colors">
                {datasetInfo.max_tokens}
              </div>
            </div>
          </div>

          {/* Validation Warnings */}
          {datasetInfo.validation_warnings && datasetInfo.validation_warnings.length > 0 && (
            <div className="pt-6 border-t border-green-500/20">
              <h4 className="font-bold text-yellow-400 mb-3 flex items-center gap-2 text-lg">
                <AlertCircle className="w-5 h-5" />
                Warnings
              </h4>
              <ul className="space-y-2">
                {datasetInfo.validation_warnings.map((warning, idx) => (
                  <li key={idx} className="text-sm text-yellow-300 flex items-start gap-2 bg-yellow-500/5 p-3 rounded-lg border border-yellow-500/20">
                    <span className="text-yellow-400 flex-shrink-0">•</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex justify-between pt-4">
        <button
          onClick={handleBack}
          className="px-8 py-4 border-2 border-white/20 text-gray-300 rounded-xl hover:bg-white/5 hover:border-white/30 font-bold text-lg transition-all duration-300"
        >
          ← Back to Model
        </button>
        {datasetInfo && (
          <button
            onClick={handleNext}
            className="group px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-500 font-bold text-lg transition-all duration-300 hover:glow-strong hover:scale-105 shadow-lg shadow-green-500/20 flex items-center gap-2"
          >
            Continue to Hyperparameters
            <span className="group-hover:translate-x-1 transition-transform">→</span>
          </button>
        )}
      </div>
    </div>
  );
}
