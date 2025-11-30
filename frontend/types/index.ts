export interface ModelInfo {
  model_id: string;
  architecture: string;
  num_parameters: number;
  parameter_size: string;
  supported_tasks: string[];
  tokenizer_type: string;
  context_length: number;
  vram_requirements: {
    inference_fp16: number;
    inference_int8: number;
    inference_int4: number;
    training_full_fp16: number;
    training_lora_fp16: number;
    training_qlora_4bit: number;
  };
  compute_requirements?: {
    min_vram_gb: number;
    recommended_vram_gb: number;
    max_batch_size: number;
  };
  license: string;
  model_type: string;
  hidden_size: number;
  num_layers: number;
  num_attention_heads: number;
  vocab_size: number;
  has_bias: boolean;
  activation_function: string;
}

export interface DatasetInfo {
  dataset_id: string;
  num_samples: number;
  num_train_samples: number;
  num_validation_samples: number | null;
  avg_tokens: number;
  max_tokens: number;
  min_tokens: number;
  data_preview: Array<Record<string, any>>;
  columns: string[];
  validation_warnings: string[];
  format: string;
  size_mb: number;
}

export interface LoRAConfig {
  r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  bias: string;
  task_type: string;
}

export interface TrainingConfig {
  model_id: string;
  dataset_id: string;
  task_type: string;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  num_epochs: number;
  max_steps: number | null;
  warmup_steps: number;
  optimizer: string;
  scheduler: string;
  weight_decay: number;
  max_grad_norm: number;
  use_lora: boolean;
  lora_config: LoRAConfig | null;
  quantization: string | null;
  load_in_4bit: boolean;
  load_in_8bit: boolean;
  fp16: boolean;
  bf16: boolean;
  gradient_checkpointing: boolean;
  logging_steps: number;
  save_steps: number;
  eval_steps: number;
  save_total_limit: number;
  max_seq_length: number;
  validation_split: number;
  seed: number;
  group_by_length: boolean;
  report_to: string[];
  // Ablation toggles
  qlora: boolean;
  use_paged_optimizers: boolean;
  bnb_4bit_use_double_quant: boolean;
}

export interface HyperparameterRecommendation {
  config: TrainingConfig;
  explanations: Record<string, string>;
  estimated_vram_gb: number;
  estimated_training_time_hours: number;
  estimated_cost_usd: number | null;
  confidence_score: number;
  warnings: string[];
}

export interface TrainingMetrics {
  step: number;
  epoch: number;
  train_loss: number;
  learning_rate: number;
  grad_norm: number | null;
  samples_per_second: number;
  steps_per_second: number;
  gpu_mem_allocated?: number;
  gpu_mem_reserved?: number;
}

export interface TrainingProgress {
  job_id: string;
  status: 'queued' | 'initializing' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_step: number;
  total_steps: number;
  current_epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  best_val_loss: number | null;
  learning_rate: number;
  samples_per_second: number;
  eta_seconds: number;
  gpu_memory_usage: number;
  gpu_utilization: number;
  latest_metrics: TrainingMetrics | null;
  checkpoints: string[];
  started_at: string | null;
  progress_message?: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface ExperimentMetadata {
  experiment_id: string;
  seed: number | null;
  config: TrainingConfig | null;
  artifacts: Record<string, string>;
}

export interface EvalMetrics {
  perplexity: number | null;
  final_loss: number | null;
  training_time_seconds: number | null;
  peak_memory_mb: number | null;
  total_steps: number | null;
  eval_samples?: number;
}

export interface ModelCard {
  model_name?: string;
  base_model?: string;
  created_at?: string;
  language?: string;
  tags?: string[];
  library_name?: string;
  description?: string;
  model_id?: string;
  job_id?: string;
  date?: string;
  training_config?: TrainingConfig;
  dataset_stats?: DatasetInfo;
  evaluation?: EvalMetrics;
  ablations?: Record<string, boolean>;
  environment?: Record<string, any>;
  baseline_comparison?: {
    qlora: boolean;
    improvement_over_baseline: string;
  };
}

export interface JobResponse {
  job_id: string;
  status: string;
  message: string;
  estimated_duration_minutes: number | null;
}

export interface QuantizationResult {
  original_size_mb: number;
  quantized_size_mb: number;
  compression_ratio: number;
  method: string;
  bits: number;
  output_path: string;
  estimated_speedup: number;
  estimated_memory_reduction: number;
}

export interface GeneratedCode {
  code: string;
  filename: string;
  language: string;
  description: string;
}

export type ComputeTier = 'free' | 'basic' | 'pro' | 'enterprise';

export type TaskType = 
  | 'text-generation'
  | 'text-classification'
  | 'token-classification'
  | 'question-answering'
  | 'summarization'
  | 'translation';

export type PipelineStep = 
  | 'model'
  | 'dataset'
  | 'hyperparameters'
  | 'training'
  | 'quantization'
  | 'export';

export interface PipelineState {
  currentStep: PipelineStep;
  completedSteps: PipelineStep[];
  modelInfo: ModelInfo | null;
  datasetInfo: DatasetInfo | null;
  recommendations: HyperparameterRecommendation | null;
  trainingConfig: TrainingConfig | null;
  trainingJobId: string | null;
  trainingProgress: TrainingProgress | null;
  quantizationResult: QuantizationResult | null;
  evalMetrics: EvalMetrics | null;
  modelCard: ModelCard | null;
  experimentMetadata: ExperimentMetadata | null;
}

export interface ApiError {
  error: string;
  detail?: string;
  status_code: number;
}

export interface WebSocketMessage {
  type: 'progress' | 'log' | 'complete' | 'error';
  data?: any;
  message?: string;
  status?: string;
}
