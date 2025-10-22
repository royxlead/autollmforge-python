"""
Additional QLoRA utility functions and helpers.
"""

from typing import Dict, Optional, List
import torch


def validate_qlora_environment() -> Dict[str, bool]:
    """Validate QLoRA environment and dependencies.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'cuda_available': False,
        'bitsandbytes_available': False,
        'peft_available': False,
        'transformers_available': False,
        'gpu_memory_sufficient': False
    }
    
    try:
        results['cuda_available'] = torch.cuda.is_available()
        if results['cuda_available']:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results['gpu_memory_sufficient'] = gpu_memory >= 8
    except Exception:
        pass
    
    try:
        import bitsandbytes
        results['bitsandbytes_available'] = True
    except ImportError:
        pass
    
    try:
        import peft
        results['peft_available'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        results['transformers_available'] = True
    except ImportError:
        pass
    
    return results


def get_optimal_qlora_config(
    model_size_b: float,
    available_vram_gb: float,
    task_complexity: str = "medium"
) -> Dict[str, any]:
    """Get optimal QLoRA configuration for given constraints.
    
    Args:
        model_size_b: Model size in billions of parameters
        available_vram_gb: Available VRAM in GB
        task_complexity: Task complexity ('simple', 'medium', 'complex')
        
    Returns:
        Optimal QLoRA configuration
    """
    # Base LoRA rank selection
    if task_complexity == "simple":
        base_rank = 4
    elif task_complexity == "complex":
        base_rank = 16
    else:
        base_rank = 8
    
    # Adjust rank based on model size
    if model_size_b > 30:
        rank = min(base_rank, 8)  # Cap at 8 for very large models
    elif model_size_b > 13:
        rank = base_rank
    else:
        rank = min(base_rank * 2, 16)  # Can afford higher rank for smaller models
    
    # Batch size based on VRAM
    if available_vram_gb < 12:
        batch_size = 1
        grad_accum = 8
    elif available_vram_gb < 24:
        batch_size = 2
        grad_accum = 4
    elif available_vram_gb < 40:
        batch_size = 4
        grad_accum = 4
    else:
        batch_size = 8
        grad_accum = 2
    
    # Learning rate scaling
    if model_size_b < 1:
        lr = 5e-4
    elif model_size_b < 7:
        lr = 2e-4
    elif model_size_b < 13:
        lr = 1e-4
    else:
        lr = 5e-5
    
    return {
        'load_in_4bit': True,
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True,
        'lora_r': rank,
        'lora_alpha': rank * 2,
        'lora_dropout': 0.05,
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_accum,
        'learning_rate': lr,
        'optimizer': 'paged_adamw_8bit',
        'gradient_checkpointing': True,
        'max_seq_length': 512 if available_vram_gb < 24 else 1024
    }


def estimate_qlora_memory_usage(
    model_size_b: float,
    batch_size: int,
    seq_length: int,
    lora_rank: int = 8
) -> Dict[str, float]:
    """Quick estimation of QLoRA memory usage.
    
    Args:
        model_size_b: Model size in billions
        batch_size: Training batch size
        seq_length: Sequence length
        lora_rank: LoRA rank
        
    Returns:
        Memory usage breakdown in GB
    """
    params = model_size_b * 1e9
    
    # Base model (4-bit with double quant)
    model_memory = params * 0.55 / (1024**3)
    
    # LoRA adapters (1% of params in FP16)
    adapter_params = params * 0.01
    adapter_memory = adapter_params * 2 / (1024**3)
    
    # Optimizer states (only for adapters)
    optimizer_memory = adapter_params * 2 * 2 / (1024**3)  # AdamW = 2x params
    
    # Activations (with gradient checkpointing)
    hidden_size = (params / 12) ** 0.5
    activation_memory = (hidden_size * seq_length * batch_size * 12 * 2) / (1024**3) * 0.1
    
    # Overhead
    overhead = (model_memory + adapter_memory + optimizer_memory + activation_memory) * 0.1
    
    total = model_memory + adapter_memory + optimizer_memory + activation_memory + overhead
    
    return {
        'model_memory_gb': round(model_memory, 2),
        'adapter_memory_gb': round(adapter_memory, 2),
        'optimizer_memory_gb': round(optimizer_memory, 2),
        'activation_memory_gb': round(activation_memory, 2),
        'overhead_memory_gb': round(overhead, 2),
        'total_memory_gb': round(total, 2)
    }


def get_qlora_warnings(
    model_size_b: float,
    available_vram_gb: float,
    dataset_size: int
) -> List[str]:
    """Get warnings about potential QLoRA training issues.
    
    Args:
        model_size_b: Model size in billions
        available_vram_gb: Available VRAM
        dataset_size: Number of training samples
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Estimate minimum VRAM needed
    min_vram = model_size_b * 1.5  # Rough estimate
    
    if available_vram_gb < min_vram:
        warnings.append(
            f"⚠️ VRAM may be insufficient. Model needs ~{min_vram:.0f}GB, "
            f"you have {available_vram_gb:.0f}GB. Consider a smaller model."
        )
    elif available_vram_gb < min_vram * 1.2:
        warnings.append(
            f"⚠️ VRAM is tight. Reduce batch_size to 1 if you encounter OOM errors."
        )
    
    # Dataset size warnings
    if dataset_size < 50:
        warnings.append(
            "⚠️ Very small dataset (<50 samples). Model will likely overfit. "
            "Use more data or strong regularization."
        )
    elif dataset_size < 200:
        warnings.append(
            "⚠️ Small dataset (<200 samples). Monitor validation loss closely "
            "for overfitting."
        )
    
    # Model size warnings
    if model_size_b > 70 and available_vram_gb < 80:
        warnings.append(
            f"⚠️ {model_size_b:.0f}B model is very large. Consider using a "
            "smaller model or multi-GPU setup."
        )
    
    return warnings


def format_qlora_summary(config: Dict) -> str:
    """Format QLoRA configuration as human-readable summary.
    
    Args:
        config: QLoRA configuration dictionary
        
    Returns:
        Formatted summary string
    """
    summary = "🎯 QLoRA Configuration Summary\n\n"
    summary += "Quantization:\n"
    summary += f"  • 4-bit: {config.get('load_in_4bit', False)}\n"
    summary += f"  • Quant Type: {config.get('bnb_4bit_quant_type', 'nf4')}\n"
    summary += f"  • Double Quant: {config.get('bnb_4bit_use_double_quant', True)}\n\n"
    
    summary += "LoRA:\n"
    summary += f"  • Rank: {config.get('lora_r', 8)}\n"
    summary += f"  • Alpha: {config.get('lora_alpha', 16)}\n"
    summary += f"  • Dropout: {config.get('lora_dropout', 0.05)}\n\n"
    
    summary += "Training:\n"
    summary += f"  • Batch Size: {config.get('batch_size', 4)}\n"
    summary += f"  • Grad Accum: {config.get('gradient_accumulation_steps', 4)}\n"
    summary += f"  • Learning Rate: {config.get('learning_rate', 2e-4):.2e}\n"
    summary += f"  • Optimizer: {config.get('optimizer', 'paged_adamw_8bit')}\n"
    summary += f"  • Gradient Checkpointing: {config.get('gradient_checkpointing', True)}\n"
    
    return summary
