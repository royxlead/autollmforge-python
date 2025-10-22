"""Hyperparameter optimization service."""

from typing import Dict, Any
from models.schemas import (
    ModelInfo, DatasetInfo, TrainingConfig, LoRAConfig,
    HyperparameterRecommendation, ComputeTier, TaskType
)
from utils.compute_estimator import ComputeEstimator
from utils.logger import get_logger
from config import COMPUTE_TIERS
import math

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """Intelligent hyperparameter recommendation engine."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.compute_estimator = ComputeEstimator()
    
    def recommend_hyperparameters(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        compute_tier: ComputeTier = ComputeTier.FREE,
        task_type: TaskType = TaskType.TEXT_GENERATION
    ) -> HyperparameterRecommendation:
        """Generate optimal hyperparameters based on model, data, and compute.
        
        Args:
            model_info: Model information
            dataset_info: Dataset information
            compute_tier: Available compute tier
            task_type: Task type
            
        Returns:
            HyperparameterRecommendation with config and explanations
        """
        logger.info(f"="*80)
        logger.info(f"🎯 Generating recommendations for {model_info.model_id}")
        logger.info(f"📊 Model Parameters: {model_info.num_parameters:,}")
        logger.info(f"📚 Dataset Samples: {dataset_info.num_train_samples:,}")
        logger.info(f"🏗️ Model Type: {model_info.model_type}")
        logger.info(f"="*80)
        
        compute_constraints = COMPUTE_TIERS[compute_tier.value]
        available_vram = compute_constraints["vram_gb"]
        
        batch_size = self.calculate_optimal_batch_size(
            model_info.num_parameters,
            available_vram,
            dataset_info.avg_tokens
        )
        
        target_effective_batch = self.get_target_batch_size(dataset_info.num_samples)
        gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
        
        learning_rate = self.recommend_learning_rate(
            model_info.num_parameters,
            task_type
        )
        logger.info(f"✅ Recommended learning rate: {learning_rate:.2e} (for {model_info.num_parameters:,} params)")
        
        num_epochs = self.recommend_epochs(dataset_info.num_samples)
        
        warmup_steps = self.recommend_warmup_steps(
            dataset_info.num_samples,
            batch_size,
            gradient_accumulation_steps
        )
        
        lora_config = self.suggest_lora_config(
            model_info.model_type,
            model_info.num_parameters,
            task_type
        )
        logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, modules={lora_config.target_modules}")
        
        quantization, load_in_4bit, load_in_8bit = self.recommend_quantization(
            model_info.num_parameters,
            available_vram
        )
        
        max_seq_length = min(
            max(128, int(dataset_info.avg_tokens * 1.3)),
            model_info.context_length,
            2048 if model_info.num_parameters > 1_000_000_000 else 1024
        )
        
        if max_seq_length < 256:
            max_seq_length = 256
        else:
            max_seq_length = ((max_seq_length + 63) // 64) * 64
        
        # Calculate save/eval steps with safety check
        steps_per_epoch = max(1, dataset_info.num_samples // (batch_size * gradient_accumulation_steps))
        save_eval_steps = max(50, min(500, steps_per_epoch // 10))
        
        config = TrainingConfig(
            model_id=model_info.model_id,
            dataset_id=dataset_info.dataset_id,
            task_type=task_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer="paged_adamw_8bit",
            scheduler="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            use_lora=True,
            lora_config=lora_config,
            quantization=quantization,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            fp16=True,
            bf16=False,
            gradient_checkpointing=True,
            max_seq_length=max_seq_length,
            validation_split=0.1,
            logging_steps=10,
            save_steps=save_eval_steps,
            eval_steps=save_eval_steps,
        )
        
        explanations = self.generate_explanations(
            config, model_info, dataset_info, compute_tier
        )
        
        memory_estimate = self.compute_estimator.estimate_training_memory(
            num_params=model_info.num_parameters,
            batch_size=batch_size,
            seq_length=max_seq_length,
            use_lora=True,
            gradient_checkpointing=True
        )
        
        throughput = self.compute_estimator.calculate_throughput(
            num_params=model_info.num_parameters,
            batch_size=batch_size,
            seq_length=max_seq_length,
            gpu_type="A100" if compute_tier.value == "enterprise" else "V100"
        )
        
        training_time_seconds, time_str = self.compute_estimator.estimate_training_time(
            num_samples=dataset_info.num_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            samples_per_second=throughput
        )
        
        training_time_hours = training_time_seconds / 3600
        
        cost = self.compute_estimator.estimate_cost(
            training_time_hours=training_time_hours,
            gpu_type="A100" if compute_tier.value == "enterprise" else "V100"
        )
        
        confidence = self.calculate_confidence_score(
            model_info, dataset_info, memory_estimate["total_memory_gb"], available_vram
        )
        
        warnings = self.generate_warnings(
            memory_estimate["total_memory_gb"],
            available_vram,
            dataset_info.num_samples,
            dataset_info.validation_warnings
        )
        
        return HyperparameterRecommendation(
            config=config,
            explanations=explanations,
            estimated_vram_gb=memory_estimate["total_memory_gb"],
            estimated_training_time_hours=round(training_time_hours, 2),
            estimated_cost_usd=cost,
            confidence_score=confidence,
            warnings=warnings
        )
    
    def calculate_optimal_batch_size(
        self,
        model_size: int,
        gpu_vram: float,
        avg_seq_length: float
    ) -> int:
        """Calculate optimal batch size for available VRAM.
        
        Args:
            model_size: Number of parameters
            gpu_vram: Available VRAM in GB
            avg_seq_length: Average sequence length
            
        Returns:
            Recommended batch size
        """
        return self.compute_estimator.recommend_batch_size(
            num_params=model_size,
            available_vram_gb=gpu_vram,
            seq_length=int(avg_seq_length),
            use_lora=True,
            gradient_checkpointing=True
        )
    
    def get_target_batch_size(self, num_samples: int) -> int:
        if num_samples < 100:
            return 2
        elif num_samples < 500:
            return 4
        elif num_samples < 1000:
            return 6
        elif num_samples < 5000:
            return 8
        elif num_samples < 10000:
            return 12
        elif num_samples < 50000:
            return 16
        elif num_samples < 100000:
            return 24
        else:
            return 32
    
    def recommend_learning_rate(self, num_params: int, task_type: TaskType) -> float:
        if num_params < 100_000_000:  # < 100M (e.g., DistilGPT2: 82M)
            base_lr = 8e-4
        elif num_params < 500_000_000:  # < 500M (e.g., GPT2-medium: 355M)
            base_lr = 5e-4
        elif num_params < 1_000_000_000:  # < 1B (e.g., GPT2-large: 774M)
            base_lr = 3e-4
        elif num_params < 3_000_000_000:  # < 3B
            base_lr = 2e-4
        elif num_params < 7_000_000_000:  # < 7B (e.g., Llama2-7B, Mistral-7B)
            base_lr = 1.5e-4
        elif num_params < 13_000_000_000:  # < 13B (e.g., Llama2-13B)
            base_lr = 1e-4
        elif num_params < 30_000_000_000:  # < 30B
            base_lr = 7e-5
        else:  # >= 30B (e.g., Llama2-70B)
            base_lr = 5e-5
        
        if task_type == TaskType.TEXT_CLASSIFICATION:
            base_lr *= 2
        elif task_type == TaskType.QUESTION_ANSWERING:
            base_lr *= 1.5
        
        return base_lr
    
    def recommend_epochs(self, num_samples: int) -> int:
        if num_samples < 50:
            return 15
        elif num_samples < 100:
            return 10
        elif num_samples < 500:
            return 7
        elif num_samples < 1000:
            return 5
        elif num_samples < 5000:
            return 4
        elif num_samples < 10000:
            return 3
        elif num_samples < 50000:
            return 2
        else:
            return 1
    
    def recommend_warmup_steps(
        self,
        num_samples: int,
        batch_size: int,
        gradient_accumulation: int
    ) -> int:
        """Recommend warmup steps.
        
        Args:
            num_samples: Number of training samples
            batch_size: Batch size
            gradient_accumulation: Gradient accumulation steps
            
        Returns:
            Recommended warmup steps
        """
        steps_per_epoch = math.ceil(num_samples / (batch_size * gradient_accumulation))
        # 10% of first epoch
        warmup = max(50, int(steps_per_epoch * 0.1))
        return min(warmup, 500)  # Cap at 500
    
    def suggest_lora_config(
        self,
        model_type: str,
        num_params: int,
        task_type: TaskType
    ) -> LoRAConfig:
        if num_params < 100_000_000:  # < 100M (tiny models like DistilGPT2)
            r = 32
            lora_dropout = 0.1
        elif num_params < 500_000_000:  # < 500M
            r = 16
            lora_dropout = 0.08
        elif num_params < 1_000_000_000:  # < 1B
            r = 16
            lora_dropout = 0.05
        elif num_params < 3_000_000_000:  # < 3B
            r = 12
            lora_dropout = 0.05
        elif num_params < 7_000_000_000:  # < 7B
            r = 8
            lora_dropout = 0.05
        elif num_params < 13_000_000_000:  # < 13B
            r = 8
            lora_dropout = 0.05
        else:  # >= 13B
            r = 4
            lora_dropout = 0.05
        
        lora_alpha = r * 2
        
        model_type_lower = model_type.lower()
        
        if 'llama' in model_type_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'mistral' in model_type_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'falcon' in model_type_lower:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 'gpt2' in model_type_lower or 'distilgpt2' in model_type_lower:
            target_modules = ["c_attn", "c_proj", "c_fc"]
        elif 'gptj' in model_type_lower or 'gpt-j' in model_type_lower:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"]
        elif 'bloom' in model_type_lower:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 't5' in model_type_lower:
            target_modules = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        elif 'bert' in model_type_lower or 'roberta' in model_type_lower:
            target_modules = ["query", "key", "value", "dense"]
        else:
            target_modules = ["q_proj", "v_proj"]
        
        return LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM" if task_type == TaskType.TEXT_GENERATION else "SEQ_CLS"
        )
    
    def recommend_quantization(
        self,
        num_params: int,
        available_vram: float
    ) -> tuple[str | None, bool, bool]:
        """Recommend quantization settings for QLoRA (default to 4-bit).
        
        QLoRA (Quantized LoRA) uses 4-bit quantization by default for efficient fine-tuning.
        This reduces memory usage by ~75% while maintaining performance.
        
        Args:
            num_params: Number of parameters
            available_vram: Available VRAM in GB
            
        Returns:
            Tuple of (quantization_type, load_in_4bit, load_in_8bit)
        """
        # QLoRA: Default to 4-bit NormalFloat quantization
        # This is the recommended approach for efficient fine-tuning
        # Benefits: 4x memory reduction, minimal accuracy loss, faster training
        
        # Always use 4-bit for models > 1B parameters (QLoRA standard)
        if num_params > 1_000_000_000:
            return "4bit", True, False
        
        # For smaller models, still prefer 4-bit unless we have lots of VRAM
        fp16_memory = num_params * 2 / (1024 ** 3)
        
        if fp16_memory < 4 and available_vram > 16:
            # Very small models with plenty of VRAM can use FP16
            # But still prefer 4-bit for consistency and best practices
            return "4bit", True, False
        else:
            # Default to 4-bit QLoRA for optimal efficiency
            return "4bit", True, False
    
    def generate_explanations(
        self,
        config: TrainingConfig,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        compute_tier: ComputeTier
    ) -> Dict[str, str]:
        """Generate explanations for each hyperparameter choice.
        
        Args:
            config: Training configuration
            model_info: Model information
            dataset_info: Dataset information
            compute_tier: Compute tier
            
        Returns:
            Dictionary of explanations
        """
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        
        return {
            "learning_rate": f"Set to {config.learning_rate:.2e} optimized for {model_info.parameter_size} ({model_info.num_parameters:,} params). Smaller models use higher LR, larger models need lower LR for stability.",
            "batch_size": f"Batch size {config.batch_size} optimized for {model_info.parameter_size} on {compute_tier.value} tier ({COMPUTE_TIERS[compute_tier.value]['vram_gb']}GB VRAM). Accounts for model size, sequence length ({config.max_seq_length}), and 4-bit quantization.",
            "gradient_accumulation_steps": f"Accumulating {config.gradient_accumulation_steps} steps → effective batch of {effective_batch}. Optimized for {dataset_info.num_samples} samples to balance convergence speed and memory.",
            "num_epochs": f"{config.num_epochs} epochs for {dataset_info.num_samples} samples. Smaller datasets need more epochs, larger datasets fewer to prevent overfitting.",
            "lora_config": f"LoRA rank {config.lora_config.r} with alpha {config.lora_config.lora_alpha}, targeting {len(config.lora_config.target_modules)} modules: {', '.join(config.lora_config.target_modules[:3])}{'...' if len(config.lora_config.target_modules) > 3 else ''}. Rank scales with model size - smaller models use higher rank.",
            "quantization": f"QLoRA 4-bit NF4 quantization for {model_info.parameter_size} model. Reduces {model_info.num_parameters:,} params from ~{model_info.num_parameters*2/1e9:.1f}GB to ~{model_info.num_parameters*0.5/1e9:.1f}GB (75% memory savings).",
            "max_seq_length": f"{config.max_seq_length} tokens (dataset avg: {dataset_info.avg_tokens:.0f}). Aligned to 64-token boundary for optimal GPU performance. Max context: {model_info.context_length}.",
            "warmup_steps": f"{config.warmup_steps} warmup steps for gradual learning rate ramp-up. Prevents initial training instability, especially important for {model_info.parameter_size} models."
        }
    
    def calculate_confidence_score(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        estimated_vram: float,
        available_vram: float
    ) -> float:
        """Calculate confidence score for recommendations.
        
        Args:
            model_info: Model information
            dataset_info: Dataset information
            estimated_vram: Estimated VRAM usage
            available_vram: Available VRAM
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 1.0
        
        # Reduce confidence if VRAM is tight
        vram_ratio = estimated_vram / available_vram
        if vram_ratio > 0.9:
            confidence *= 0.7
        elif vram_ratio > 0.8:
            confidence *= 0.85
        
        # Reduce confidence for very small datasets
        if dataset_info.num_samples < 100:
            confidence *= 0.8
        elif dataset_info.num_samples < 500:
            confidence *= 0.9
        
        # Reduce confidence if dataset has warnings
        if dataset_info.validation_warnings:
            confidence *= 0.85
        
        return round(confidence, 2)
    
    def generate_warnings(
        self,
        estimated_vram: float,
        available_vram: float,
        num_samples: int,
        dataset_warnings: list[str]
    ) -> list[str]:
        """Generate warnings about potential issues.
        
        Args:
            estimated_vram: Estimated VRAM usage
            available_vram: Available VRAM
            num_samples: Number of samples
            dataset_warnings: Dataset validation warnings
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # VRAM warnings
        if estimated_vram > available_vram * 0.95:
            warnings.append("⚠️ VRAM usage is very high - training may fail with OOM errors")
        elif estimated_vram > available_vram * 0.85:
            warnings.append("⚠️ VRAM usage is high - consider reducing batch size if you encounter OOM")
        
        # Dataset warnings
        if num_samples < 100:
            warnings.append("⚠️ Very small dataset - model may overfit quickly")
        elif num_samples < 500:
            warnings.append("⚠️ Small dataset - monitor validation loss closely for overfitting")
        
        # Add dataset validation warnings
        warnings.extend(dataset_warnings)
        
        return warnings
