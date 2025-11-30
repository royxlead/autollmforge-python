import json
import os

# Disable torch.compile before importing torch (not supported on Python 3.14+)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from utils.logger import get_logger
from config import settings
from datetime import datetime

logger = get_logger(__name__)

class EvalService:
    def __init__(self):
        self.output_dir = Path(settings.outputs_dir)
        
    async def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        config: Dict[str, Any],
        split: str = "test"
    ) -> Dict[str, float]:
        """Run evaluation pipeline on a trained model."""
        logger.info(f"Starting evaluation for {model_path} on {dataset_path}")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            model.eval()
            
            # Load dataset
            dataset = load_dataset("json", data_files=dataset_path, split="train")
            # If we have a split strategy, we should respect it. 
            # For now, assuming the dataset passed is the validation set or we split it.
            if config.get("validation_split", 0) > 0:
                dataset = dataset.train_test_split(
                    test_size=config["validation_split"],
                    seed=config.get("seed", 42)
                )["test"]
            
            # Calculate Perplexity
            encodings = tokenizer(
                "\n\n".join(dataset["text"][:100]), # Limit to 100 samples for speed
                return_tensors="pt"
            )
            
            # Get max length from config (different models use different attribute names)
            max_length = getattr(model.config, 'n_positions', None) or \
                         getattr(model.config, 'max_position_embeddings', None) or \
                         getattr(model.config, 'max_sequence_length', 2048)
            stride = 512
            seq_len = encodings.input_ids.size(1)
            
            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                    
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            
            ppl = torch.exp(torch.stack(nlls).mean())
            logger.info(f"Perplexity: {ppl.item()}")
            
            metrics = {
                "perplexity": float(ppl.item()),
                "eval_samples": len(dataset)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    async def generate_model_card(
        self,
        job_id: str,
        training_args: Dict[str, Any],
        dataset_stats: Dict[str, Any],
        eval_metrics: Dict[str, Any],
        ablations: Dict[str, Any],
        environment: Dict[str, Any]
    ):
        """Generate a model card for the trained model."""
        card_path = self.output_dir / job_id / "model_card.json"
        
        model_card = {
            "model_id": training_args.get("model_id"),
            "job_id": job_id,
            "date": datetime.now().isoformat(),
            "training_config": training_args,
            "dataset_stats": dataset_stats,
            "evaluation": eval_metrics,
            "ablations": ablations,
            "environment": environment,
            "baseline_comparison": {
                "qlora": ablations.get("qlora", True),
                "improvement_over_baseline": "N/A" # Placeholder
            }
        }
        
        with open(card_path, "w") as f:
            json.dump(model_card, f, indent=2)
            
        logger.info(f"Model card generated at {card_path}")
        return model_card

    async def validate_quantization(self, model_path: str, expected_dtype: str = "float16"):
        """Validate that the model layers match the expected quantization/dtype."""
        logger.info(f"Validating quantization for {model_path}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            
            mismatches = []
            for name, module in model.named_modules():
                if "Linear" in str(type(module)):
                    # Check if 4bit or 8bit
                    if hasattr(module, "weight"):
                        dtype = str(module.weight.dtype)
                        if expected_dtype not in dtype:
                            mismatches.append(f"{name}: {dtype} != {expected_dtype}")
            
            if mismatches:
                logger.warning(f"Quantization mismatches found: {mismatches[:5]}...")
                return {"valid": False, "mismatches": mismatches}
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Quantization validation failed: {e}")
            return {"valid": False, "error": str(e)}
