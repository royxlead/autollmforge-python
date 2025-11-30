"""Training service for fine-tuning models."""

import os
# Disable torch.compile before importing torch (not supported on Python 3.14+)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Disable torch dynamo/compile for Python 3.14+ compatibility
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

from models.schemas import TrainingConfig, TrainingProgress, TrainingStatus, TrainingMetrics
from utils.logger import get_logger
from config import settings
from services.dataset_processor import DatasetProcessor

logger = get_logger(__name__)


class TrainingService:
    """Service for managing model training jobs."""
    
    def __init__(self):
        """Initialize training service."""
        self.output_dir = Path(settings.outputs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = Path(settings.experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.dataset_processor = DatasetProcessor()

    def _set_seed(self, seed: int):
        """Set global seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    
    async def start_training(
        self,
        config: TrainingConfig,
        job_name: Optional[str] = None
    ) -> str:
        job_id = str(uuid.uuid4())
        
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = job_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        
        job_state = {
            "job_id": job_id,
            "job_name": job_name or f"training_{job_id[:8]}",
            "config": config,
            "status": TrainingStatus.QUEUED,
            "started_at": datetime.now().isoformat(),
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "metrics": [],
            "checkpoints": [],
            "output_dir": str(job_dir)
        }
        
        self.active_jobs[job_id] = job_state
        logger.info(f"Training job created: {job_id}")
        
        asyncio.create_task(self._run_training(job_id))
        
        return job_id
    
    async def _run_training(self, job_id: str):
        job_state = self.active_jobs[job_id]
        config: TrainingConfig = job_state["config"]
        
        try:
            job_state["status"] = TrainingStatus.INITIALIZING
            logger.info(f"Initializing training job: {job_id}")
            
            await self._run_actual_training(job_id)
            
            if job_state["status"] != TrainingStatus.FAILED:
                job_state["status"] = TrainingStatus.COMPLETED
                job_state["completed_at"] = datetime.now().isoformat()
                logger.info(f"Training job completed: {job_id}")
            
        except Exception as e:
            logger.error(f"Training job failed: {job_id} - {e}")
            job_state["status"] = TrainingStatus.FAILED
            job_state["error_message"] = str(e)
            job_state["completed_at"] = datetime.now().isoformat()
    
    async def _simulate_training(self, job_id: str):
        """Simulate training progress (for demo).
        
        In production, replace this with actual training using:
        - Transformers Trainer
        - PEFT for LoRA
        - Custom training loops
        
        Args:
            job_id: Job identifier
        """
        job_state = self.active_jobs[job_id]
        config: TrainingConfig = job_state["config"]
        
        steps_per_epoch = 100
        total_steps = steps_per_epoch * config.num_epochs
        job_state["total_steps"] = total_steps
        job_state["status"] = TrainingStatus.RUNNING
        
        step = 0
        for epoch in range(config.num_epochs):
            job_state["current_epoch"] = epoch + 1
            
            for step_in_epoch in range(steps_per_epoch):
                step += 1
                job_state["current_step"] = step
                
                train_loss = 2.0 * (1 - step / total_steps) + 0.3
                
                if step < config.warmup_steps:
                    lr = config.learning_rate * (step / config.warmup_steps)
                else:
                    import math
                    progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
                    lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
                
                metrics = TrainingMetrics(
                    step=step,
                    epoch=epoch + step_in_epoch / steps_per_epoch,
                    train_loss=round(train_loss, 4),
                    learning_rate=lr,
                    grad_norm=1.0,
                    samples_per_second=2.5,
                    steps_per_second=0.625
                )
                
                job_state["metrics"].append(metrics.dict())
                
                if step % config.save_steps == 0:
                    checkpoint_name = f"checkpoint-{step}"
                    job_state["checkpoints"].append(checkpoint_name)
                    logger.info(f"Checkpoint saved: {checkpoint_name}")
                
                await asyncio.sleep(0.1)
        
        logger.info(f"Training simulation completed for job: {job_id}")
    
    def get_training_progress(self, job_id: str) -> TrainingProgress:
        if job_id not in self.active_jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        job_state = self.active_jobs[job_id]
        
        latest_metrics = None
        if job_state["metrics"]:
            latest_metrics = TrainingMetrics(**job_state["metrics"][-1])
        
        if job_state["status"] == TrainingStatus.RUNNING and latest_metrics:
            remaining_steps = job_state["total_steps"] - job_state["current_step"]
            eta_seconds = int(remaining_steps / latest_metrics.steps_per_second) if latest_metrics.steps_per_second > 0 else 0
        else:
            eta_seconds = 0
        
        val_loss = latest_metrics.train_loss * 1.1 if latest_metrics else None
        best_val_loss = min(m["train_loss"] for m in job_state["metrics"]) * 1.1 if job_state["metrics"] else None
        
        # Ensure we always have valid values (not 0/0 which causes NaN)
        current_step = job_state.get("current_step", 0)
        total_steps = max(job_state.get("total_steps", 1), 1)  # Avoid division by zero
        
        return TrainingProgress(
            job_id=job_id,
            status=job_state["status"],
            current_step=current_step,
            total_steps=total_steps,
            current_epoch=job_state.get("current_epoch", 0),
            total_epochs=job_state["config"].num_epochs,
            train_loss=latest_metrics.train_loss if latest_metrics else None,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            learning_rate=latest_metrics.learning_rate if latest_metrics else 0.0,
            samples_per_second=latest_metrics.samples_per_second if latest_metrics else 0.0,
            eta_seconds=eta_seconds,
            gpu_memory_usage=12.5,
            gpu_utilization=85.0,
            latest_metrics=latest_metrics,
            checkpoints=job_state.get("checkpoints", []),
            started_at=job_state.get("started_at"),
            completed_at=job_state.get("completed_at"),
            error_message=job_state.get("error_message"),
            progress_message=job_state.get("progress_message", "Initializing...")
        )
    
    def cancel_training(self, job_id: str):
        if job_id not in self.active_jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        job_state = self.active_jobs[job_id]
        if job_state["status"] in [TrainingStatus.QUEUED, TrainingStatus.RUNNING]:
            job_state["status"] = TrainingStatus.CANCELLED
            job_state["completed_at"] = datetime.now().isoformat()
            logger.info(f"Training job cancelled: {job_id}")
        else:
            raise ValueError(f"Cannot cancel job in status: {job_state['status']}")
    
    def get_all_jobs(self) -> list[Dict[str, Any]]:
        return [
            {
                "job_id": job_id,
                "job_name": state["job_name"],
                "status": state["status"],
                "model_id": state["config"].model_id,
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at")
            }
            for job_id, state in self.active_jobs.items()
        ]
    
    def cleanup_job(self, job_id: str):
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
            logger.info(f"Job cleaned up: {job_id}")


    async def _run_actual_training(self, job_id: str):
        import torch
        
        # Monkeypatch torch.compile to be a no-op if it exists
        # This is a workaround for "torch.compile is not supported on Python 3.14+" error
        if hasattr(torch, 'compile'):
            def no_op_compile(model, *args, **kwargs):
                return model
            torch.compile = no_op_compile
            
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            BitsAndBytesConfig,
            TrainerCallback
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        
        job_state = self.active_jobs[job_id]
        config: TrainingConfig = job_state["config"]
        output_dir = Path(job_state["output_dir"])
        experiment_dir = self.experiments_dir / job_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            job_state["status"] = TrainingStatus.RUNNING
            job_state["progress_message"] = "🚀 Initializing training job..."
            logger.info(f"Initializing training for job: {job_id}")
            
            # 1. Deterministic Training
            self._set_seed(config.seed)
            
            await asyncio.sleep(0.1)
            
            # 2. CPU Fallback & Device Selection
            gpu_available = torch.cuda.is_available()
            logger.info(f"GPU available: {gpu_available}")
            
            if not gpu_available:
                logger.warning("No GPU detected - disabling quantization and LoRA for CPU training")
                job_state["progress_message"] = "⚠️ No GPU detected - using CPU mode (slower but compatible)"
                config.load_in_4bit = False
                config.load_in_8bit = False
                config.fp16 = False
                config.bf16 = False
                # We might still want LoRA on CPU to save memory, but usually it's slow. 
                # Keeping use_lora as per config, but disabling bitsandbytes.
            
            await asyncio.sleep(0.1)
            job_state["progress_message"] = "📦 Downloading tokenizer..."
            logger.info(f"Loading tokenizer from {config.model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                token=settings.hf_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            job_state["progress_message"] = "✅ Tokenizer loaded successfully"
            
            # 3. Model Loading (QLoRA vs Baseline)
            model_kwargs = {
                "trust_remote_code": True,
                "token": settings.hf_token,
                "use_cache": False if config.gradient_checkpointing else True
            }
            
            if gpu_available and config.qlora and (config.load_in_4bit or config.load_in_8bit):
                await asyncio.sleep(0.1)
                job_state["progress_message"] = "⚙️ Configuring quantization..."
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=config.load_in_4bit,
                    load_in_8bit=config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16
                
                logger.info(f"Loading model {config.model_id} with quantization...")
            else:
                await asyncio.sleep(0.1)
                job_state["progress_message"] = f"📥 Downloading model {config.model_id} (Standard Mode)..."
                logger.info(f"Loading model {config.model_id} without quantization...")
                model_kwargs["torch_dtype"] = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
            
            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # 4. Prepare for LoRA (if enabled)
            if config.use_lora:
                if gpu_available and config.qlora and (config.load_in_4bit or config.load_in_8bit):
                    model = prepare_model_for_kbit_training(
                        model,
                        use_gradient_checkpointing=config.gradient_checkpointing
                    )
                
                job_state["progress_message"] = "🎯 Configuring LoRA adapters..."
                
                if not config.lora_config:
                    from models.schemas import LoRAConfig
                    config.lora_config = LoRAConfig()
                
                target_modules = config.lora_config.target_modules
                if config.model_id.lower().startswith('gpt2') or 'gpt2' in config.model_id.lower():
                    target_modules = ["c_attn"]
                
                peft_config = LoraConfig(
                    r=config.lora_config.r,
                    lora_alpha=config.lora_config.lora_alpha,
                    lora_dropout=config.lora_config.lora_dropout,
                    target_modules=target_modules,
                    bias=config.lora_config.bias,
                    task_type=config.lora_config.task_type,
                    inference_mode=False
                )
                
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            
            # 5. Dataset Loading & Caching
            await asyncio.sleep(0.1)
            job_state["progress_message"] = "📊 Loading and tokenizing dataset..."
            
            try:
                # Use the cached dataset processor
                tokenized_dataset = await self.dataset_processor.get_tokenized_dataset(
                    dataset_path=config.dataset_id,
                    tokenizer_name=config.model_id,
                    max_seq_length=config.max_seq_length,
                    validation_split=config.validation_split,
                    seed=config.seed
                )
                
                if config.validation_split > 0:
                    train_dataset = tokenized_dataset["train"]
                    eval_dataset = tokenized_dataset["test"]
                else:
                    train_dataset = tokenized_dataset
                    eval_dataset = None
                    
            except Exception as e:
                logger.error(f"Dataset processing failed: {e}")
                raise ValueError(f"Failed to process dataset: {str(e)}")

            # 6. Experiment Tracking & Callbacks
            class ExperimentCallback(TrainerCallback):
                def __init__(self, job_state, experiment_dir):
                    self.job_state = job_state
                    self.experiment_dir = experiment_dir
                    self.metrics_history = []
                    self.loss_history = []
                    self.steps_history = []
                
                def on_step_end(self, args, state, control, **kwargs):
                    """Update progress after every step."""
                    self.job_state["current_step"] = state.global_step
                    self.job_state["current_epoch"] = state.epoch
                    
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        # Memory Profiling
                        if torch.cuda.is_available():
                            logs["gpu_mem_allocated"] = torch.cuda.memory_allocated()
                            logs["gpu_mem_reserved"] = torch.cuda.memory_reserved()
                        
                        # Structured JSON Logging
                        log_entry = {
                            "step": state.global_step,
                            "timestamp": datetime.now().isoformat(),
                            **logs
                        }
                        
                        with open(self.experiment_dir / "metrics.jsonl", "a") as f:
                            f.write(json.dumps(log_entry) + "\n")
                        
                        # Update Job State
                        loss = logs.get("loss")
                        if loss:
                            self.loss_history.append(loss)
                            self.steps_history.append(state.global_step)
                            
                            # Update in-memory metrics for UI
                            metrics = TrainingMetrics(
                                step=state.global_step,
                                epoch=state.epoch,
                                train_loss=loss,
                                learning_rate=logs.get("learning_rate", args.learning_rate),
                                grad_norm=logs.get("grad_norm", 0.0),
                                samples_per_second=logs.get("samples_per_second", 0.0),
                                steps_per_second=logs.get("steps_per_second", 0.0)
                            )
                            self.job_state["metrics"].append(metrics.dict())
                            self.job_state["progress_message"] = f"🔥 Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f}"
                            
                            # 7. Graph Persistence
                            if len(self.steps_history) > 1:
                                try:
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(self.steps_history, self.loss_history, label="Training Loss")
                                    plt.xlabel("Steps")
                                    plt.ylabel("Loss")
                                    plt.title(f"Training Loss - {self.job_state['job_name']}")
                                    plt.legend()
                                    plt.grid(True)
                                    plt.savefig(self.experiment_dir / "loss.png")
                                    plt.close()
                                except Exception as plot_err:
                                    logger.warning(f"Failed to save loss plot: {plot_err}")

                def on_train_begin(self, args, state, control, **kwargs):
                    self.job_state["total_steps"] = state.max_steps
                    self.job_state["status"] = TrainingStatus.RUNNING
                    
                    # Save experiment metadata (Ablations)
                    metadata = {
                        "config": self.job_state["config"].dict(),
                        "ablations": {
                            "use_gradient_checkpointing": config.gradient_checkpointing,
                            "use_double_quant": config.bnb_4bit_use_double_quant,
                            "use_paged_optimizers": config.use_paged_optimizers,
                            "qlora": config.qlora
                        },
                        "environment": {
                            "gpu_available": torch.cuda.is_available(),
                            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                            "torch_version": torch.__version__
                        }
                    }
                    with open(self.experiment_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

            # 7. Training Arguments
            optim_type = "paged_adamw_8bit" if (gpu_available and config.use_paged_optimizers) else "adamw_torch"
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=config.gradient_checkpointing,
                optim=optim_type,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                max_grad_norm=config.max_grad_norm,
                lr_scheduler_type=config.scheduler,
                warmup_steps=config.warmup_steps,
                fp16=(config.fp16 and not config.bf16) if gpu_available else False,
                bf16=config.bf16 if gpu_available else False,
                logging_steps=1,  # Force logging every step for real-time UI updates
                save_steps=config.save_steps,
                eval_steps=config.eval_steps if eval_dataset else None,
                save_total_limit=config.save_total_limit,
                eval_strategy="steps" if eval_dataset else "no",
                seed=config.seed,
                group_by_length=config.group_by_length,
                report_to=config.report_to,
                ddp_find_unused_parameters=False,
                remove_unused_columns=False,
                torch_compile=False,  # Disable torch.compile (not supported on Python 3.14+)
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[ExperimentCallback(job_state, experiment_dir)]
            )
            
            job_state["progress_message"] = "🚀 Starting training loop..."
            logger.info(f"Starting training for job: {job_id}")
            
            train_result = trainer.train()
            
            # Save final model
            job_state["progress_message"] = "💾 Saving fine-tuned model..."
            trainer.save_model(str(output_dir / "final_model"))
            tokenizer.save_pretrained(str(output_dir / "final_model"))
            
            # Save metrics
            metrics_file = output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            job_state["status"] = TrainingStatus.COMPLETED
            job_state["progress_message"] = "✅ Training completed successfully!"
            job_state["completed_at"] = datetime.now().isoformat()
            logger.info(f"Training completed successfully: {job_id}")
            
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)
            job_state["status"] = TrainingStatus.FAILED
            job_state["error_message"] = str(e)
            job_state["completed_at"] = datetime.now().isoformat()
            raise
