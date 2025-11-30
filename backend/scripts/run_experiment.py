import asyncio
import sys
import json
import argparse
from pathlib import Path
import logging
import os

# Disable torch.compile before importing torch (not supported on Python 3.14+)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.model_analyzer import ModelAnalyzer
from services.dataset_processor import DatasetProcessor
from services.training_service import TrainingService
from services.eval_service import EvalService
from models.schemas import TrainingConfig, TrainingStatus
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experiment_runner")

async def run_experiment(config_path: str):
    logger.info(f"Starting experiment from config: {config_path}")
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    
    training_config_data = exp_config.get("training_config")
    training_config = TrainingConfig(**training_config_data)
    
    # Initialize Services
    model_analyzer = ModelAnalyzer()
    dataset_processor = DatasetProcessor()
    training_service = TrainingService()
    eval_service = EvalService()
    
    try:
        # 2. Analyze Model
        logger.info("Step 1: Analyzing Model...")
        model_info = await model_analyzer.analyze_model(training_config.model_id)
        logger.info(f"Model Analysis: {model_info.parameter_size} params, {model_info.architecture}")
        
        # 3. Prepare Dataset (Tokenization is handled inside training service now with caching)
        logger.info("Step 2: Dataset Preparation...")
        # We validate it exists
        dataset_path = Path(training_config.dataset_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        # 4. Train
        logger.info("Step 3: Training...")
        job_id = await training_service.start_training(
            config=training_config,
            job_name=exp_config.get("job_name", "experiment")
        )
        
        logger.info(f"Training Job ID: {job_id}")
        
        # Poll for completion
        while True:
            progress = training_service.get_training_progress(job_id)
            if progress.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                break
            logger.info(f"Training Status: {progress.status} - Step {progress.current_step}/{progress.total_steps}")
            await asyncio.sleep(10)
            
        if progress.status != TrainingStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {progress.error_message}")
            
        logger.info("Training Completed Successfully!")
        
        # 5. Evaluate
        logger.info("Step 4: Evaluation...")
        output_dir = Path(settings.outputs_dir) / job_id / "final_model"
        
        eval_metrics = await eval_service.evaluate_model(
            model_path=str(output_dir),
            dataset_path=training_config.dataset_id,
            config=training_config.dict()
        )
        logger.info(f"Evaluation Metrics: {eval_metrics}")
        
        # 6. Generate Model Card
        logger.info("Step 5: Generating Model Card...")
        dataset_stats = await dataset_processor.validate_dataset(training_config.dataset_id)
        
        await eval_service.generate_model_card(
            job_id=job_id,
            training_args=training_config.dict(),
            dataset_stats=dataset_stats.dict(),
            eval_metrics=eval_metrics,
            ablations={
                "qlora": training_config.qlora,
                "gradient_checkpointing": training_config.gradient_checkpointing
            },
            environment={
                "platform": sys.platform
            }
        )
        
        logger.info(f"Experiment {exp_config.get('job_name')} completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Experiment")
    parser.add_argument("config", help="Path to experiment config JSON")
    args = parser.parse_args()
    
    asyncio.run(run_experiment(args.config))
