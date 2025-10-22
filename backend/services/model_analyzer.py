"""Model analysis service for extracting model information."""

from typing import Dict, Any
from models.schemas import ModelInfo
from utils.hf_utils import HuggingFaceClient
from utils.compute_estimator import ComputeEstimator
from utils.logger import get_logger
from config import settings, SUPPORTED_TASKS

logger = get_logger(__name__)


class ModelAnalyzer:
    """Service for analyzing Hugging Face models."""
    
    def __init__(self):
        """Initialize model analyzer."""
        self.hf_client = HuggingFaceClient(token=settings.hf_token)
        self.compute_estimator = ComputeEstimator()
    
    async def analyze_model(self, model_id: str) -> ModelInfo:
        """Comprehensively analyze a Hugging Face model.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            ModelInfo object with complete model details
            
        Raises:
            Exception: If model cannot be analyzed
        """
        try:
            logger.info(f"Starting analysis for model: {model_id}")
            
            hub_info = await self.hf_client.get_model_info(model_id)
            config = self.hf_client.get_model_config(model_id)
            tokenizer_info = self.hf_client.get_tokenizer_info(model_id)
            architecture_details = self.get_architecture_details(config)
            
            num_params = 0
            
            if hub_info.get("num_parameters"):
                num_params = hub_info["num_parameters"]
                logger.info(f"Got {num_params} parameters from HuggingFace Hub metadata")
            else:
                num_params = self.hf_client.count_parameters(config)
                logger.info(f"Estimated {num_params} parameters from config")
            
            if num_params == 0 and hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
                logger.info(f"Got {num_params} parameters from config.num_parameters")
            
            param_size = self.hf_client.format_parameter_size(num_params)
            vram_requirements = self.estimate_memory_requirements(num_params)
            supported_tasks = self.get_supported_tasks(hub_info, config)
            context_length = self.get_context_length(config, tokenizer_info)
            
            model_info = ModelInfo(
                model_id=model_id,
                architecture=architecture_details["model_type"],
                num_parameters=num_params,
                parameter_size=param_size,
                supported_tasks=supported_tasks,
                tokenizer_type=tokenizer_info.get("tokenizer_type", "unknown"),
                context_length=context_length,
                vram_requirements=vram_requirements,
                license=hub_info.get("license", "unknown"),
                model_type=architecture_details["model_type"],
                hidden_size=architecture_details["hidden_size"],
                num_layers=architecture_details["num_layers"],
                num_attention_heads=architecture_details["num_attention_heads"],
                vocab_size=architecture_details["vocab_size"],
                has_bias=architecture_details["has_bias"],
                activation_function=architecture_details["activation_function"]
            )
            
            logger.info(f"Successfully analyzed model: {model_id} - {param_size} parameters")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_id}: {e}")
            raise Exception(f"Model analysis failed: {str(e)}")
    
    def get_architecture_details(self, config: Any) -> Dict[str, Any]:
        """Extract detailed architecture information from config.
        
        Args:
            config: Model configuration object
            
        Returns:
            Dictionary with architecture details
        """
        return {
            "model_type": getattr(config, 'model_type', 'unknown'),
            "hidden_size": getattr(config, 'hidden_size', 0),
            "num_layers": getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0),
            "num_attention_heads": getattr(config, 'num_attention_heads', 0),
            "intermediate_size": getattr(config, 'intermediate_size', 0),
            "vocab_size": getattr(config, 'vocab_size', 0),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 0),
            "has_bias": getattr(config, 'bias', True),
            "activation_function": getattr(config, 'hidden_act', 'unknown'),
            "layer_norm_eps": getattr(config, 'layer_norm_eps', getattr(config, 'layer_norm_epsilon', 1e-5)),
            "initializer_range": getattr(config, 'initializer_range', 0.02),
        }
    
    def estimate_memory_requirements(self, num_params: int) -> Dict[str, float]:
        """Calculate VRAM requirements for different configurations.
        
        Args:
            num_params: Number of model parameters
            
        Returns:
            Dictionary with VRAM estimates in GB
        """
        return {
            "inference_fp16": round(self.compute_estimator.estimate_model_memory(num_params, "fp16"), 2),
            "inference_int8": round(self.compute_estimator.estimate_model_memory(num_params, "fp16", "8bit"), 2),
            "inference_int4": round(self.compute_estimator.estimate_model_memory(num_params, "fp16", "4bit"), 2),
            "training_full_fp16": round(self.compute_estimator.estimate_training_memory(
                num_params, batch_size=1, seq_length=512, use_lora=False
            )["total_memory_gb"], 2),
            "training_lora_fp16": round(self.compute_estimator.estimate_training_memory(
                num_params, batch_size=4, seq_length=512, use_lora=True
            )["total_memory_gb"], 2),
            "training_qlora_4bit": round(self.compute_estimator.estimate_training_memory(
                num_params, batch_size=4, seq_length=512, use_lora=True
            )["total_memory_gb"] * 0.5, 2),  # QLoRA roughly halves memory
        }
    
    def get_supported_tasks(self, hub_info: Dict[str, Any], config: Any) -> list[str]:
        """Determine supported tasks for the model.
        
        Args:
            hub_info: Model hub information
            config: Model configuration
            
        Returns:
            List of supported task types
        """
        tasks = []
        
        pipeline_tag = hub_info.get("pipeline_tag")
        if pipeline_tag and pipeline_tag in SUPPORTED_TASKS:
            tasks.append(pipeline_tag)
        
        tags = hub_info.get("tags", [])
        for tag in tags:
            if tag in SUPPORTED_TASKS:
                tasks.append(tag)
        
        model_type = getattr(config, 'model_type', '').lower()
        
        if any(x in model_type for x in ['gpt', 'llama', 'mistral', 'opt', 'bloom']):
            if "text-generation" not in tasks:
                tasks.append("text-generation")
        
        if any(x in model_type for x in ['bert', 'roberta', 'albert', 'electra']):
            if "text-classification" not in tasks:
                tasks.append("text-classification")
            if "token-classification" not in tasks:
                tasks.append("token-classification")
        
        if 't5' in model_type or 'bart' in model_type:
            if "summarization" not in tasks:
                tasks.append("summarization")
            if "translation" not in tasks:
                tasks.append("translation")
        
        if not tasks:
            tasks.append("text-generation")
        
        return tasks
    
    def get_context_length(self, config: Any, tokenizer_info: Dict[str, Any]) -> int:
        """Determine maximum context length.
        
        Args:
            config: Model configuration
            tokenizer_info: Tokenizer information
            
        Returns:
            Maximum context length
        """
        context_length = (
            getattr(config, 'max_position_embeddings', 0) or
            getattr(config, 'n_positions', 0) or
            getattr(config, 'max_sequence_length', 0) or
            tokenizer_info.get('model_max_length', 0)
        )
        
        if context_length > 1_000_000:
            context_length = 4096  # Use reasonable default
        
        return context_length if context_length > 0 else 2048  # Default fallback
