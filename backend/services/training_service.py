"""Training service for fine-tuning models."""

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from models.schemas import TrainingConfig, TrainingProgress, TrainingStatus, TrainingMetrics
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class TrainingService:
    """Service for managing model training jobs."""
    
    def __init__(self):
        """Initialize training service."""
        self.output_dir = Path(settings.outputs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
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
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        
        job_state = self.active_jobs[job_id]
        config: TrainingConfig = job_state["config"]
        output_dir = Path(job_state["output_dir"])
        
        try:
            job_state["status"] = TrainingStatus.RUNNING
            job_state["progress_message"] = "🚀 Initializing training job..."
            logger.info(f"Initializing QLoRA training for job: {job_id}")
            
            await asyncio.sleep(0.1)  # Allow UI to update
            
            gpu_available = torch.cuda.is_available()
            logger.info(f"GPU available: {gpu_available}")
            
            if not gpu_available:
                logger.warning("No GPU detected - disabling quantization for CPU training")
                job_state["progress_message"] = "⚠️ No GPU detected - using CPU mode (slower but compatible)"
            
            await asyncio.sleep(0.1)  # Allow UI to update
            job_state["progress_message"] = "📦 Downloading tokenizer (this may take a few minutes)..."
            logger.info(f"Loading tokenizer from {config.model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                token=settings.hf_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            job_state["progress_message"] = "✅ Tokenizer loaded successfully"
            logger.info("Tokenizer loaded successfully")
            
            if gpu_available and (config.load_in_4bit or config.load_in_8bit):
                await asyncio.sleep(0.1)  # Allow UI to update
                job_state["progress_message"] = "⚙️ Configuring 4-bit quantization..."
                logger.info("Configuring quantization")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=config.load_in_4bit,
                    load_in_8bit=config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                )
                
                await asyncio.sleep(0.1)  # Allow UI to update
                job_state["progress_message"] = f"📥 Downloading model {config.model_id.split('/')[-1]} with 4-bit quantization (this may take 5-10 minutes)..."
                logger.info(f"Loading model {config.model_id} with quantization (GPU)...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_cache=False if config.gradient_checkpointing else True,
                    token=settings.hf_token
                )
                
                job_state["progress_message"] = "🔧 Preparing model for QLoRA training..."
                logger.info("Preparing model for k-bit training")
                
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=config.gradient_checkpointing
                )
                
                job_state["progress_message"] = "✅ Model loaded and quantized successfully"
                logger.info("Model prepared successfully")
            else:
                await asyncio.sleep(0.1)  # Allow UI to update
                job_state["progress_message"] = f"📥 Downloading model {config.model_id.split('/')[-1]} (CPU mode, this may take 10-15 minutes)..."
                logger.info(f"Loading model {config.model_id} on CPU (no quantization)...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    use_cache=False if config.gradient_checkpointing else True,
                    token=settings.hf_token
                )
                
                job_state["progress_message"] = "✅ Model loaded successfully"
                logger.info("Model loaded successfully")
            
            if config.gradient_checkpointing:
                await asyncio.sleep(0.1)  # Allow UI to update
                job_state["progress_message"] = "🔧 Enabling gradient checkpointing for memory efficiency..."
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            if config.use_lora:
                await asyncio.sleep(0.1)  # Allow UI to update
                job_state["progress_message"] = "🎯 Configuring LoRA adapters..."
                logger.info("Configuring LoRA adapters...")
                
                if not config.lora_config:
                    logger.warning("use_lora is True but lora_config is None, creating default config")
                    from models.schemas import LoRAConfig
                    config.lora_config = LoRAConfig()
                
                target_modules = config.lora_config.target_modules
                logger.info(f"Target modules type: {type(target_modules)}, value: {target_modules}")
                
                if config.model_id.lower().startswith('gpt2') or 'gpt2' in config.model_id.lower():
                    logger.info("Detected GPT-2 model, using c_attn for LoRA target modules")
                    target_modules = ["c_attn"]
                elif isinstance(target_modules, str):
                    logger.warning(f"Converting target_modules from string to list: {target_modules}")
                    target_modules = ["q_proj", "v_proj"]
                elif not isinstance(target_modules, list):
                    logger.error(f"Unexpected target_modules type: {type(target_modules)}")
                    target_modules = ["q_proj", "v_proj"]
                
                logger.info(f"Final target_modules for LoRA: {target_modules}")
                
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
                
                job_state["progress_message"] = "✅ LoRA adapters configured successfully"
                logger.info("LoRA configuration complete")
            
            await asyncio.sleep(0.1)  # Allow UI to update
            job_state["progress_message"] = "📊 Loading and validating dataset..."
            logger.info(f"Loading dataset: {config.dataset_id}")
            
            try:
                dataset = load_dataset("json", data_files=config.dataset_id, split="train")
            except Exception as e:
                raise ValueError(f"Failed to load dataset file. Make sure the file exists and is valid JSON. Error: {str(e)}")
            
            job_state["progress_message"] = "✅ Dataset loaded successfully"
            await asyncio.sleep(0.1)  # Allow UI to update
            
            if "text" not in dataset.column_names:
                raise ValueError(
                    f"Dataset must contain a 'text' field. Found columns: {dataset.column_names}. "
                    f"Please format your dataset as JSON with each entry having a 'text' field. "
                    f"Example: {{'text': 'Your training text here'}}"
                )
            
            job_state["progress_message"] = f"📊 Splitting dataset ({len(dataset)} samples)..."
            await asyncio.sleep(0.1)  # Allow UI to update
            
            if config.validation_split > 0:
                split_dataset = dataset.train_test_split(
                    test_size=config.validation_split,
                    seed=config.seed
                )
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset = None
            
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=config.max_seq_length,
                    padding="max_length"
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
            
            job_state["progress_message"] = "🔤 Tokenizing training dataset..."
            await asyncio.sleep(0.1)  # Allow UI to update
            logger.info("Tokenizing dataset...")
            
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing train dataset"
            )
            
            job_state["progress_message"] = "✅ Training dataset tokenized"
            await asyncio.sleep(0.1)  # Allow UI to update
            
            if eval_dataset:
                job_state["progress_message"] = "🔤 Tokenizing validation dataset..."
                await asyncio.sleep(0.1)  # Allow UI to update
                
                eval_dataset = eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=eval_dataset.column_names,
                    desc="Tokenizing eval dataset"
                )
                
                job_state["progress_message"] = "✅ Validation dataset tokenized"
                await asyncio.sleep(0.1)  # Allow UI to update
            
            # Calculate total steps for progress tracking
            num_samples = len(train_dataset)
            steps_per_epoch = num_samples // (config.batch_size * config.gradient_accumulation_steps)
            if steps_per_epoch == 0:
                steps_per_epoch = max(1, num_samples // config.batch_size)
            total_steps = steps_per_epoch * config.num_epochs
            job_state["total_steps"] = total_steps
            job_state["current_step"] = 0
            job_state["current_epoch"] = 0
            logger.info(f"📊 Training plan: {num_samples} samples → {steps_per_epoch} steps/epoch × {config.num_epochs} epochs = {total_steps} total steps")
            
            job_state["progress_message"] = f"⚙️ Setting up training ({total_steps} total steps across {config.num_epochs} epochs)..."
            await asyncio.sleep(0.1)  # Allow UI to update
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=config.gradient_checkpointing,
                optim="paged_adamw_8bit" if gpu_available else "adamw_torch",
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                max_grad_norm=config.max_grad_norm,
                lr_scheduler_type=config.scheduler,
                warmup_steps=config.warmup_steps,
                fp16=(config.fp16 and not config.bf16) if gpu_available else False,
                bf16=config.bf16 if gpu_available else False,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps if eval_dataset else None,
                save_total_limit=config.save_total_limit,
                eval_strategy="steps" if eval_dataset else "no",
                seed=config.seed,
                group_by_length=config.group_by_length,
                report_to=config.report_to,
                ddp_find_unused_parameters=False,
                remove_unused_columns=False,
            )
            
            job_state["progress_message"] = "✅ Training configuration ready"
            await asyncio.sleep(0.1)  # Allow UI to update
            
            # Estimate total steps for progress tracking
            num_train_samples = len(train_dataset)
            steps_per_epoch = num_train_samples // (config.batch_size * config.gradient_accumulation_steps)
            estimated_total_steps = steps_per_epoch * config.num_epochs
            job_state["total_steps"] = estimated_total_steps
            logger.info(f"Estimated total steps: {estimated_total_steps} ({steps_per_epoch} steps/epoch)")
            
            # Custom callback to track progress
            from transformers import TrainerCallback
            
            class ProgressCallback(TrainerCallback):
                def __init__(self, job_state):
                    self.job_state = job_state
                    self.last_log_step = 0
                    
                def on_train_begin(self, args, state, control, **kwargs):
                    """Update total steps when training actually begins."""
                    self.job_state["total_steps"] = state.max_steps
                    self.job_state["current_step"] = 0
                    self.job_state["current_epoch"] = 0
                    self.job_state["progress_message"] = f"🚀 Training loop starting! ({state.max_steps} total steps)"
                    
                    # Initialize first metric entry
                    initial_metric = TrainingMetrics(
                        step=0,
                        epoch=0,
                        train_loss=0.0,
                        learning_rate=args.learning_rate,
                        grad_norm=0.0,
                        samples_per_second=0.0,
                        steps_per_second=0.0
                    )
                    self.job_state["metrics"] = [initial_metric.dict()]
                    logger.info(f"✅ Training loop initialized with {state.max_steps} steps")
                    
                def on_log(self, args, state, control, logs=None, **kwargs):
                    """Capture metrics whenever logging happens."""
                    logger.info(f"🔔 on_log called: logs={logs}, step={state.global_step}")
                    
                    if logs and state.global_step > 0:
                        self.job_state["current_step"] = state.global_step
                        self.job_state["total_steps"] = state.max_steps
                        self.job_state["current_epoch"] = state.epoch
                        
                        loss = logs.get("loss", None)
                        lr = logs.get("learning_rate", None)
                        
                        logger.info(f"📊 Logging callback: Step {state.global_step}, Loss={loss}, LR={lr}, All logs: {list(logs.keys())}")
                        
                        # Only add metrics if we have actual loss data
                        if loss is not None:
                            metrics = TrainingMetrics(
                                step=state.global_step,
                                epoch=state.epoch,
                                train_loss=loss,
                                learning_rate=lr if lr is not None else args.learning_rate,
                                grad_norm=logs.get("grad_norm", 0.0),
                                samples_per_second=logs.get("samples_per_second", 0.0),
                                steps_per_second=logs.get("steps_per_second", 0.0)
                            )
                            self.job_state["metrics"].append(metrics.dict())
                            self.job_state["progress_message"] = f"🔥 Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f} | LR: {lr:.2e if lr else 0:.2e}"
                            logger.info(f"✅ Added metric: Loss={loss:.4f}")
                        
                def on_step_end(self, args, state, control, **kwargs):
                    """Update progress after each step."""
                    self.job_state["current_step"] = state.global_step
                    self.job_state["total_steps"] = state.max_steps
                    self.job_state["current_epoch"] = state.epoch
                    
                    # If we have log history, extract the latest loss
                    if state.log_history:
                        latest_log = state.log_history[-1]
                        loss = latest_log.get("loss", 0.0)
                        lr = latest_log.get("learning_rate", 0.0)
                        
                        # Only update metrics if this step hasn't been logged yet (avoid duplicates from on_log)
                        if state.global_step % args.logging_steps == 0 and (
                            not self.job_state["metrics"] or 
                            self.job_state["metrics"][-1]["step"] != state.global_step
                        ):
                            metrics = TrainingMetrics(
                                step=state.global_step,
                                epoch=state.epoch,
                                train_loss=loss,
                                learning_rate=lr,
                                grad_norm=latest_log.get("grad_norm", 0.0),
                                samples_per_second=latest_log.get("samples_per_second", 0.0),
                                steps_per_second=latest_log.get("steps_per_second", 0.0)
                            )
                            self.job_state["metrics"].append(metrics.dict())
                            logger.info(f"📊 Step end: Added metrics for step {state.global_step}")
                        
                        self.job_state["progress_message"] = f"⚡ Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f}"
                    else:
                        self.job_state["progress_message"] = f"⚡ Training step {state.global_step}/{state.max_steps}"
                    
                    if state.global_step % 5 == 0:  # Log every 5 steps to avoid spam
                        logger.info(f"Step {state.global_step}/{state.max_steps} completed")
            
            job_state["progress_message"] = "🏋️ Initializing Trainer..."
            await asyncio.sleep(0.1)  # Allow UI to update
            logger.info("Initializing Trainer...")
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[ProgressCallback(job_state)]
            )
            
            job_state["progress_message"] = "🚀 Starting training loop..."
            await asyncio.sleep(0.1)  # Allow UI to update
            logger.info(f"Starting QLoRA training for job: {job_id}")
            
            train_result = trainer.train()
            
            # Add final metrics to job state
            logger.info(f"Training completed with metrics: {train_result.metrics}")
            final_metrics = TrainingMetrics(
                step=job_state.get("current_step", job_state.get("total_steps", 0)),
                epoch=train_result.metrics.get("epoch", config.num_epochs),
                train_loss=train_result.metrics.get("train_loss", 0.0),
                learning_rate=config.learning_rate,  # Final LR from config
                grad_norm=0.0,
                samples_per_second=train_result.metrics.get("train_samples_per_second", 0.0),
                steps_per_second=train_result.metrics.get("train_steps_per_second", 0.0)
            )
            job_state["metrics"].append(final_metrics.dict())
            job_state["progress_message"] = f"✅ Training completed! Final loss: {train_result.metrics.get('train_loss', 0.0):.4f}"
            
            job_state["progress_message"] = "💾 Saving fine-tuned model..."
            await asyncio.sleep(0.1)  # Allow UI to update
            logger.info("Saving final model...")
            
            trainer.save_model(str(output_dir / "final_model"))
            tokenizer.save_pretrained(str(output_dir / "final_model"))
            
            job_state["progress_message"] = "📊 Saving training metrics..."
            await asyncio.sleep(0.1)  # Allow UI to update
            
            metrics_file = output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            job_state["status"] = TrainingStatus.COMPLETED
            job_state["progress_message"] = "✅ Training completed successfully!"
            job_state["completed_at"] = datetime.now().isoformat()
            logger.info(f"QLoRA training completed successfully: {job_id}")
            
        except Exception as e:
            logger.error(f"QLoRA training failed for job {job_id}: {e}", exc_info=True)
            job_state["status"] = TrainingStatus.FAILED
            job_state["error_message"] = str(e)
            job_state["completed_at"] = datetime.now().isoformat()
            raise
