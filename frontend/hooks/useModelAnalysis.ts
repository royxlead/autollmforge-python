import { useState } from 'react';
import { apiClient } from '@/lib/api';
import type { ModelInfo } from '@/types';

export function useModelAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeModel = async (modelId: string): Promise<ModelInfo | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.analyzeModel(modelId);
      return response;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to analyze model';
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { analyzeModel, isLoading, error };
}
