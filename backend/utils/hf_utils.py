"""Hugging Face API utilities."""

from typing import Dict, Any, Optional, List
from huggingface_hub import HfApi, model_info, list_models
from transformers import AutoConfig, AutoTokenizer
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceClient:
    """Client for interacting with Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize HF client.
        
        Args:
            token: Hugging Face API token
        """
        self.api = HfApi(token=token)
        self.token = token
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"Fetching model info for: {model_id}")
            info = model_info(model_id, token=self.token)
            
            safetensors_params = None
            if hasattr(info, 'safetensors') and info.safetensors:
                if hasattr(info.safetensors, 'parameters'):
                    params_data = info.safetensors.parameters
                    if isinstance(params_data, dict):
                        safetensors_params = sum(params_data.values()) if params_data else None
                    else:
                        safetensors_params = params_data
                    logger.info(f"Got parameter count from safetensors: {safetensors_params:,}")
            
            return {
                "model_id": model_id,
                "author": info.author if hasattr(info, 'author') else None,
                "downloads": info.downloads if hasattr(info, 'downloads') else 0,
                "likes": info.likes if hasattr(info, 'likes') else 0,
                "tags": info.tags if hasattr(info, 'tags') else [],
                "pipeline_tag": info.pipeline_tag if hasattr(info, 'pipeline_tag') else None,
                "library_name": info.library_name if hasattr(info, 'library_name') else None,
                "license": info.card_data.license if hasattr(info, 'card_data') and info.card_data else "unknown",
                "created_at": str(info.created_at) if hasattr(info, 'created_at') else None,
                "last_modified": str(info.last_modified) if hasattr(info, 'last_modified') else None,
                "num_parameters": safetensors_params,
            }
        except Exception as e:
            logger.error(f"Error fetching model info for {model_id}: {e}")
            raise
    
    def get_model_config(self, model_id: str) -> AutoConfig:
        """Load model configuration.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            Model configuration object
        """
        try:
            logger.info(f"Loading config for: {model_id}")
            config = AutoConfig.from_pretrained(
                model_id,
                token=self.token,
                trust_remote_code=True
            )
            return config
        except Exception as e:
            logger.error(f"Error loading config for {model_id}: {e}")
            raise
    
    def get_tokenizer_info(self, model_id: str) -> Dict[str, Any]:
        """Get tokenizer information.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            Dictionary with tokenizer details
        """
        try:
            logger.info(f"Loading tokenizer for: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.token,
                trust_remote_code=True
            )
            
            return {
                "tokenizer_type": tokenizer.__class__.__name__,
                "vocab_size": tokenizer.vocab_size,
                "model_max_length": tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else None,
                "padding_side": tokenizer.padding_side if hasattr(tokenizer, 'padding_side') else None,
                "truncation_side": tokenizer.truncation_side if hasattr(tokenizer, 'truncation_side') else None,
                "special_tokens": {
                    "bos_token": tokenizer.bos_token,
                    "eos_token": tokenizer.eos_token,
                    "unk_token": tokenizer.unk_token,
                    "sep_token": tokenizer.sep_token if hasattr(tokenizer, 'sep_token') else None,
                    "pad_token": tokenizer.pad_token,
                    "cls_token": tokenizer.cls_token if hasattr(tokenizer, 'cls_token') else None,
                    "mask_token": tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else None,
                }
            }
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_id}: {e}")
            # Return basic info if tokenizer fails
            return {
                "tokenizer_type": "unknown",
                "vocab_size": 0,
                "error": str(e)
            }
    
    async def search_models(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face Hub.
        
        Args:
            query: Search query
            task: Filter by task type
            limit: Maximum number of results
            
        Returns:
            List of model information dictionaries
        """
        try:
            logger.info(f"Searching models: query={query}, task={task}")
            models = list_models(
                search=query,
                task=task,
                sort="downloads",
                direction=-1,
                limit=limit,
                token=self.token
            )
            
            results = []
            for model in models:
                results.append({
                    "model_id": model.id if hasattr(model, 'id') else model.modelId,
                    "downloads": model.downloads if hasattr(model, 'downloads') else 0,
                    "likes": model.likes if hasattr(model, 'likes') else 0,
                    "tags": model.tags if hasattr(model, 'tags') else [],
                    "pipeline_tag": model.pipeline_tag if hasattr(model, 'pipeline_tag') else None,
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            raise
    
    async def download_model_card(self, model_id: str) -> str:
        """Download and parse model card README.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            Model card content as string
        """
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(f"Could not fetch README for {model_id}")
                    return ""
        except Exception as e:
            logger.error(f"Error downloading model card: {e}")
            return ""
    
    def count_parameters(self, config: AutoConfig) -> int:
        try:
            if hasattr(config, 'num_parameters'):
                logger.info(f"Got num_parameters from config: {config.num_parameters}")
                return config.num_parameters
            
            model_type = getattr(config, 'model_type', '').lower()
            hidden_size = getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0)
            num_layers = getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0) or getattr(config, 'n_layer', 0)
            vocab_size = getattr(config, 'vocab_size', 0)
            num_heads = getattr(config, 'num_attention_heads', 0) or getattr(config, 'n_head', 0)
            intermediate_size = getattr(config, 'intermediate_size', 0) or getattr(config, 'ffn_dim', 0)
            
            if not intermediate_size:
                intermediate_size = hidden_size * 4
            
            logger.info(f"Model type: {model_type}, hidden_size: {hidden_size}, num_layers: {num_layers}, vocab_size: {vocab_size}")
            
            if 'gpt2' in model_type or 'gpt-2' in model_type:
                embedding_params = vocab_size * hidden_size
                position_params = getattr(config, 'n_positions', 1024) * hidden_size
                ln_params = num_layers * 2 * hidden_size
                attn_params = num_layers * (hidden_size * hidden_size * 3 + hidden_size * 4)
                mlp_params = num_layers * (hidden_size * intermediate_size * 2 + intermediate_size + hidden_size)
                final_ln = 2 * hidden_size
                
                total = embedding_params + position_params + ln_params + attn_params + mlp_params + final_ln
                logger.info(f"GPT-2 parameter count: {total}")
                return int(total)
            
            elif 'llama' in model_type or 'mistral' in model_type:
                embedding_params = vocab_size * hidden_size
                norm_params = (num_layers + 1) * hidden_size
                attn_params = num_layers * (hidden_size * hidden_size * 4)
                mlp_params = num_layers * (hidden_size * intermediate_size * 3)
                
                total = embedding_params + norm_params + attn_params + mlp_params
                logger.info(f"Llama/Mistral parameter count: {total}")
                return int(total)
            
            elif 't5' in model_type:
                embedding_params = vocab_size * hidden_size
                encoder_params = num_layers * (hidden_size * hidden_size * 4 + hidden_size * intermediate_size * 2)
                decoder_params = num_layers * (hidden_size * hidden_size * 4 + hidden_size * intermediate_size * 2)
                
                total = embedding_params + encoder_params + decoder_params
                logger.info(f"T5 parameter count: {total}")
                return int(total)
            
            else:
                embedding_params = vocab_size * hidden_size
                attention_params = num_layers * (4 * hidden_size * hidden_size)
                ffn_params = num_layers * (2 * hidden_size * intermediate_size)
                total_params = embedding_params + attention_params + ffn_params
                
                logger.info(f"Generic transformer parameter count: {total_params}")
                return int(total_params)
            
        except Exception as e:
            logger.error(f"Error counting parameters: {e}")
            return 0
    
    def format_parameter_size(self, num_params: int) -> str:
        """Format parameter count as human-readable string.
        
        Args:
            num_params: Number of parameters
            
        Returns:
            Formatted string (e.g., "7B", "13B", "1.5B")
        """
        if num_params >= 1_000_000_000:
            return f"{num_params / 1_000_000_000:.1f}B"
        elif num_params >= 1_000_000:
            return f"{num_params / 1_000_000:.1f}M"
        elif num_params >= 1_000:
            return f"{num_params / 1_000:.1f}K"
        else:
            return str(num_params)
