"""Compute resource estimation utilities."""

from typing import Dict, Tuple
import math
from utils.logger import get_logger

logger = get_logger(__name__)


class ComputeEstimator:
    """Estimate compute requirements for training."""
    
    # Memory overhead factors
    GRADIENT_MEMORY_FACTOR = 2  # Gradients take same space as weights
    OPTIMIZER_MEMORY_FACTOR = 2  # Adam optimizer states
    ACTIVATION_MEMORY_FACTOR = 1.5  # Activation memory
    
    # Bytes per parameter by precision
    BYTES_PER_PARAM = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    @staticmethod
    def estimate_model_memory(
        num_params: int,
        precision: str = "fp16",
        quantization: str = "4bit"  # QLoRA default
    ) -> float:
        """Estimate model memory requirements in GB (QLoRA optimized).
        
        Args:
            num_params: Number of model parameters
            precision: Training precision (fp32, fp16, bf16)
            quantization: Quantization type (4bit [QLoRA default], 8bit, None)
            
        Returns:
            Estimated memory in GB
        """
        if quantization == "4bit":
            bytes_per_param = 0.55
        elif quantization == "8bit":
            bytes_per_param = ComputeEstimator.BYTES_PER_PARAM["int8"]
        else:
            bytes_per_param = ComputeEstimator.BYTES_PER_PARAM.get(precision, 2)
        
        memory_bytes = num_params * bytes_per_param
        memory_gb = memory_bytes / (1024 ** 3)
        
        logger.debug(f"QLoRA memory estimate: {memory_gb:.2f} GB for {num_params:,} params with {quantization or precision}")
        return memory_gb
    
    @staticmethod
    def estimate_training_memory(
        num_params: int,
        batch_size: int,
        seq_length: int,
        precision: str = "fp16",
        use_lora: bool = True,
        gradient_checkpointing: bool = True,
        quantization: str = "4bit"  # QLoRA default
    ) -> Dict[str, float]:
        """Estimate total QLoRA training memory requirements.
        
        Args:
            num_params: Number of model parameters
            batch_size: Training batch size
            seq_length: Sequence length
            precision: Training precision
            use_lora: Whether using LoRA (QLoRA always uses LoRA)
            gradient_checkpointing: Whether using gradient checkpointing
            quantization: Quantization type (4bit [QLoRA], 8bit, None)
            
        Returns:
            Dictionary with memory breakdown in GB
        """
        if quantization == "4bit":
            model_memory = ComputeEstimator.estimate_model_memory(
                num_params, precision, "4bit"
            )
            bytes_per_param = 0.55
        elif quantization == "8bit":
            model_memory = ComputeEstimator.estimate_model_memory(
                num_params, precision, "8bit"
            )
            bytes_per_param = 1
        else:
            model_memory = num_params * 2 / (1024 ** 3)
            bytes_per_param = 2
        
        lora_params_ratio = 0.01 if use_lora else 1.0
        trainable_params = num_params * lora_params_ratio
        
        adapter_bytes_per_param = 2
        optimizer_memory = trainable_params * adapter_bytes_per_param * ComputeEstimator.OPTIMIZER_MEMORY_FACTOR / (1024 ** 3)
        
        gradient_memory = trainable_params * adapter_bytes_per_param / (1024 ** 3)
        
        hidden_size = math.sqrt(num_params / 12)
        num_layers = max(12, int(num_params / (hidden_size ** 2) / 12))
        
        activation_memory = (hidden_size * seq_length * batch_size * num_layers * 2) / (1024 ** 3)
        
        if gradient_checkpointing:
            activation_memory *= 0.1
        
        overhead_memory = (model_memory + optimizer_memory + gradient_memory + activation_memory) * 0.1
        
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory + overhead_memory
        
        return {
            "model_memory_gb": round(model_memory, 2),
            "optimizer_memory_gb": round(optimizer_memory, 2),
            "gradient_memory_gb": round(gradient_memory, 2),
            "activation_memory_gb": round(activation_memory, 2),
            "overhead_memory_gb": round(overhead_memory, 2),
            "total_memory_gb": round(total_memory, 2),
            "quantization": quantization or "none"
        }
    
    @staticmethod
    def estimate_training_time(
        num_samples: int,
        num_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        samples_per_second: float = 1.0
    ) -> Tuple[int, str]:
        """Estimate training time.
        
        Args:
            num_samples: Number of training samples
            num_epochs: Number of epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            samples_per_second: Throughput estimate
            
        Returns:
            Tuple of (seconds, formatted_string)
        """
        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = math.ceil(num_samples / effective_batch_size)
        total_steps = steps_per_epoch * num_epochs
        total_samples = num_samples * num_epochs
        
        estimated_seconds = int(total_samples / samples_per_second)
        
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"
        
        return estimated_seconds, time_str
    
    @staticmethod
    def estimate_cost(
        training_time_hours: float,
        gpu_type: str = "A100"
    ) -> float:
        """Estimate training cost.
        
        Args:
            training_time_hours: Training duration in hours
            gpu_type: GPU type
            
        Returns:
            Estimated cost in USD
        """
        gpu_costs = {
            "T4": 0.35,
            "V100": 1.50,
            "A100": 3.00,
            "A100-80GB": 4.00,
            "H100": 8.00
        }
        
        cost_per_hour = gpu_costs.get(gpu_type, 3.00)
        estimated_cost = training_time_hours * cost_per_hour
        
        return round(estimated_cost, 2)
    
    @staticmethod
    def recommend_batch_size(
        num_params: int,
        available_vram_gb: float,
        seq_length: int,
        use_lora: bool = True,
        gradient_checkpointing: bool = True,
        quantization: str = "4bit"  # QLoRA default
    ) -> int:
        """Recommend optimal batch size for available VRAM with QLoRA.
        
        Args:
            num_params: Number of model parameters
            available_vram_gb: Available VRAM in GB
            seq_length: Sequence length
            use_lora: Whether using LoRA (QLoRA always True)
            gradient_checkpointing: Whether using gradient checkpointing
            quantization: Quantization type (4bit for QLoRA)
            
        Returns:
            Recommended batch size
        """
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            memory_req = ComputeEstimator.estimate_training_memory(
                num_params=num_params,
                batch_size=batch_size,
                seq_length=seq_length,
                use_lora=use_lora,
                gradient_checkpointing=gradient_checkpointing,
                quantization=quantization
            )
            
            if memory_req["total_memory_gb"] > available_vram_gb * 0.85:
                return max(1, batch_size // 2)
        
        return 64
    
    @staticmethod
    def calculate_throughput(
        num_params: int,
        batch_size: int,
        seq_length: int,
        gpu_type: str = "A100"
    ) -> float:
        """Estimate training throughput (samples per second).
        
        Args:
            num_params: Number of model parameters
            batch_size: Batch size
            seq_length: Sequence length
            gpu_type: GPU type
            
        Returns:
            Estimated samples per second
        """
        reference_throughput = 2.0
        
        size_factor = (7_000_000_000 / num_params) ** 0.5
        
        batch_factor = math.sqrt(batch_size / 4)
        
        seq_factor = 512 / seq_length
        
        gpu_factors = {
            "T4": 0.2,
            "V100": 0.5,
            "A100": 1.0,
            "A100-80GB": 1.0,
            "H100": 2.0
        }
        gpu_factor = gpu_factors.get(gpu_type, 1.0)
        
        estimated_throughput = reference_throughput * size_factor * batch_factor * seq_factor * gpu_factor
        
        return round(estimated_throughput, 2)
