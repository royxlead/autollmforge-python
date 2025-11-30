import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  PipelineState,
  PipelineStep,
  ModelInfo,
  DatasetInfo,
  HyperparameterRecommendation,
  TrainingConfig,
  TrainingProgress,
  QuantizationResult,
  EvalMetrics,
  ModelCard,
  ExperimentMetadata,
} from '@/types';

interface PipelineStore extends PipelineState {
  setCurrentStep: (step: PipelineStep) => void;
  completeStep: (step: PipelineStep) => void;
  setModelInfo: (info: ModelInfo | null) => void;
  setDatasetInfo: (info: DatasetInfo | null) => void;
  setRecommendations: (rec: HyperparameterRecommendation | null) => void;
  setTrainingConfig: (config: TrainingConfig | null) => void;
  setTrainingJobId: (jobId: string | null) => void;
  setTrainingProgress: (progress: TrainingProgress | null) => void;
  setQuantizationResult: (result: QuantizationResult | null) => void;
  setEvalMetrics: (metrics: EvalMetrics | null) => void;
  setModelCard: (card: ModelCard | null) => void;
  setExperimentMetadata: (metadata: ExperimentMetadata | null) => void;
  reset: () => void;
  canProceedToStep: (step: PipelineStep) => boolean;
}

const initialState: PipelineState = {
  currentStep: 'model',
  completedSteps: [],
  modelInfo: null,
  datasetInfo: null,
  recommendations: null,
  trainingConfig: null,
  trainingJobId: null,
  trainingProgress: null,
  quantizationResult: null,
  evalMetrics: null,
  modelCard: null,
  experimentMetadata: null,
};

export const usePipelineStore = create<PipelineStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setCurrentStep: (step) => set({ currentStep: step }),

      completeStep: (step) => {
        const { completedSteps } = get();
        if (!completedSteps.includes(step)) {
          set({ completedSteps: [...completedSteps, step] });
        }
      },

      setModelInfo: (info) => {
        set({ modelInfo: info });
        if (info) {
          get().completeStep('model');
        }
      },

      setDatasetInfo: (info) => {
        set({ datasetInfo: info });
        if (info) {
          get().completeStep('dataset');
        }
      },

      setRecommendations: (rec) => {
        set({ recommendations: rec });
        if (rec) {
          set({ trainingConfig: rec.config });
          get().completeStep('hyperparameters');
        }
      },

      setTrainingConfig: (config) => set({ trainingConfig: config }),

      setTrainingJobId: (jobId) => set({ trainingJobId: jobId }),

      setTrainingProgress: (progress) => {
        set({ trainingProgress: progress });
        if (progress && progress.status === 'completed') {
          get().completeStep('training');
        }
      },

      setQuantizationResult: (result) => {
        set({ quantizationResult: result });
        if (result) {
          get().completeStep('quantization');
        }
      },

      setEvalMetrics: (metrics) => set({ evalMetrics: metrics }),

      setModelCard: (card) => set({ modelCard: card }),

      setExperimentMetadata: (metadata) => set({ experimentMetadata: metadata }),

      reset: () => set(initialState),

      canProceedToStep: (step) => {
        const { completedSteps } = get();
        const steps: PipelineStep[] = [
          'model',
          'dataset',
          'hyperparameters',
          'training',
          'quantization',
          'export',
        ];
        const stepIndex = steps.indexOf(step);
        const previousStep = steps[stepIndex - 1];

        if (stepIndex === 0) return true;

        return previousStep ? completedSteps.includes(previousStep) : false;
      },
    })
  )
);

export const selectCurrentStep = (state: PipelineStore) => state.currentStep;
export const selectModelInfo = (state: PipelineStore) => state.modelInfo;
export const selectDatasetInfo = (state: PipelineStore) => state.datasetInfo;
export const selectTrainingConfig = (state: PipelineStore) => state.trainingConfig;
export const selectTrainingProgress = (state: PipelineStore) => state.trainingProgress;
