from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from typing import List
import json
import zipfile
import io
from models.schemas import (
    ModelAnalyzeRequest, ModelInfo,
    DatasetUploadRequest, DatasetInfo,
    HyperparameterRequest, HyperparameterRecommendation,
    StartTrainingRequest, JobResponse, TrainingProgress,
    QuantizationRequest, QuantizationResult,
    CodeGenerationRequest, GeneratedCode,
    ErrorResponse
)
from services.model_analyzer import ModelAnalyzer
from services.hyperparameter_optimizer import HyperparameterOptimizer
from services.dataset_processor import DatasetProcessor
from services.training_service import TrainingService
from services.quantization_service import QuantizationService
from services.code_generator import CodeGenerator
from utils.logger import get_logger
from config import settings, POPULAR_MODELS

logger = get_logger(__name__)
model_analyzer = ModelAnalyzer()
hyperparameter_optimizer = HyperparameterOptimizer()
dataset_processor = DatasetProcessor()
training_service = TrainingService()
quantization_service = QuantizationService()
code_generator = CodeGenerator()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job: {job_id}")
    
    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job: {job_id}")
    
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(job_id, conn)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Fine-Tuning Pipeline API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Allowed origins: {settings.allowed_origins}")
    
    yield
    
    logger.info("Shutting down API")

app = FastAPI(
    title="LLM Fine-Tuning Pipeline API",
    description="Production-ready API for automated LLM fine-tuning",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "LLM Fine-Tuning Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "model_analyzer": "operational",
            "training": "operational",
            "quantization": "operational"
        }
    }

@app.post("/api/analyze-model", response_model=ModelInfo)
async def analyze_model(request: ModelAnalyzeRequest):
    try:
        logger.info(f"Analyzing model: {request.model_id}")
        model_info = await model_analyzer.analyze_model(request.model_id)
        return model_info
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/popular", response_model=List[dict])
async def get_popular_models():
    try:
        models = []
        for model_id in POPULAR_MODELS[:10]:
            try:
                info = await model_analyzer.analyze_model(model_id)
                models.append({
                    "model_id": info.model_id,
                    "parameter_size": info.parameter_size,
                    "architecture": info.architecture,
                    "supported_tasks": info.supported_tasks,
                    "vram_inference": info.vram_requirements.get("inference_fp16", 0),
                    "vram_training": info.vram_requirements.get("training_lora_fp16", 0)
                })
            except:
                continue
        
        return models
    except Exception as e:
        logger.error(f"Failed to fetch popular models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@app.post("/api/upload-dataset", response_model=DatasetInfo)
async def upload_dataset(
    file: UploadFile = File(None),
    dataset_id: str = None,
    format: str = None
):
    try:
        if file:
            if not format:
                if file.filename.endswith('.csv'):
                    format = 'csv'
                elif file.filename.endswith('.jsonl'):
                    format = 'jsonl'
                elif file.filename.endswith('.json'):
                    format = 'json'
                else:
                    format = 'json'
            
            file_path = Path(settings.datasets_dir) / file.filename
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"Dataset uploaded: {file.filename} (format: {format})")
            
            dataset_info = await dataset_processor.validate_dataset(
                str(file_path),
                format=format,
                dataset_id=file.filename
            )
            return dataset_info
            
        elif dataset_id:
            raise HTTPException(
                status_code=501,
                detail="Hugging Face dataset loading not yet implemented"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or dataset_id must be provided"
            )
            
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/validate-dataset", response_model=DatasetInfo)
async def validate_dataset(request: DatasetUploadRequest):
    try:
        raise HTTPException(status_code=501, detail="Not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/recommend-hyperparameters", response_model=HyperparameterRecommendation)
async def recommend_hyperparameters(request: HyperparameterRequest):
    try:
        logger.info(f"Generating recommendations for {request.model_id}")
        
        model_info = await model_analyzer.analyze_model(request.model_id)
        
        dataset_info = DatasetInfo(
            dataset_id=request.dataset_id,
            num_samples=1000,
            num_train_samples=900,
            num_validation_samples=100,
            avg_tokens=256,
            max_tokens=512,
            min_tokens=50,
            data_preview=[],
            columns=["text"],
            validation_warnings=[],
            format="json",
            size_mb=10.0
        )
        
        recommendations = hyperparameter_optimizer.recommend_hyperparameters(
            model_info=model_info,
            dataset_info=dataset_info,
            compute_tier=request.compute_tier,
            task_type=request.task_type
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Hyperparameter recommendation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/start-training", response_model=JobResponse)
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("="*60)
        logger.info(f"🚀 STARTING TRAINING JOB")
        logger.info(f"Model: {request.config.model_id}")
        logger.info(f"Dataset: {request.config.dataset_id}")
        logger.info(f"Epochs: {request.config.num_epochs}")
        logger.info(f"Learning Rate: {request.config.learning_rate}")
        logger.info(f"Use LoRA: {request.config.use_lora}")
        if request.config.use_lora and request.config.lora_config:
            logger.info(f"LoRA Config: r={request.config.lora_config.r}, alpha={request.config.lora_config.lora_alpha}")
            logger.info(f"Target Modules: {request.config.lora_config.target_modules} (type: {type(request.config.lora_config.target_modules).__name__})")
        logger.info(f"Job Name: {request.job_name}")
        logger.info("="*60)
        
        job_id = await training_service.start_training(
            config=request.config,
            job_name=request.job_name
        )
        
        logger.info(f"✅ Training job created successfully: {job_id}")
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Training job created successfully",
            estimated_duration_minutes=30
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start training: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training-progress/{job_id}", response_model=TrainingProgress)
async def get_training_progress(job_id: str):
    try:
        progress = training_service.get_training_progress(job_id)
        return progress
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cancel-training/{job_id}")
async def cancel_training(job_id: str):
    try:
        training_service.cancel_training(job_id)
        return {"message": f"Training job {job_id} cancelled"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training-jobs")
async def list_training_jobs():
    try:
        jobs = training_service.get_all_jobs()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-model/{job_id}")
async def download_model(job_id: str):
    try:
        logger.info(f"Preparing model download for job: {job_id}")
        
        output_dir = Path(settings.outputs_dir) / job_id / "final_model"
        
        if not output_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for job {job_id}. Training may not be complete."
            )
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zip_file.write(file_path, arcname)
                    logger.info(f"Added to ZIP: {arcname}")
            
            job_dir = Path(settings.outputs_dir) / job_id
            config_file = job_dir / "config.json"
            metrics_file = job_dir / "training_metrics.json"
            
            if config_file.exists():
                zip_file.write(config_file, "config.json")
            if metrics_file.exists():
                zip_file.write(metrics_file, "training_metrics.json")
        
        zip_buffer.seek(0)
        
        logger.info(f"Model package created for job: {job_id}")
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=model-{job_id[:8]}.zip"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    
    try:
        try:
            progress = training_service.get_training_progress(job_id)
            await websocket.send_json({
                "type": "progress",
                "data": progress.dict()
            })
        except:
            await websocket.send_json({
                "type": "error",
                "message": "Job not found"
            })
            return
        
        while True:
            try:
                progress = training_service.get_training_progress(job_id)
                
                await websocket.send_json({
                    "type": "progress",
                    "data": progress.dict()
                })
                
                if progress.status in ["completed", "failed", "cancelled"]:
                    await websocket.send_json({
                        "type": "complete",
                        "status": progress.status
                    })
                    break
                
                await asyncio.sleep(1)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        manager.disconnect(job_id, websocket)

@app.post("/api/quantize", response_model=QuantizationResult)
async def quantize_model(request: QuantizationRequest):
    try:
        logger.info(f"Quantizing model at {request.model_path}")
        
        result = await quantization_service.quantize_model(
            model_path=request.model_path,
            method=request.method,
            bits=request.bits
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/quantization-comparison/{job_id}")
async def compare_quantization_methods(job_id: str):
    try:
        model_path = f"{settings.outputs_dir}/{job_id}/final_model"
        
        comparison = await quantization_service.compare_quantization_methods(model_path)
        return comparison
        
    except Exception as e:
        logger.error(f"Quantization comparison failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate-code", response_model=GeneratedCode)
async def generate_code(request: CodeGenerationRequest):
    try:
        logger.info(f"Generating {request.code_type} code")
        
        if request.code_type == "inference":
            code = code_generator.generate_inference_script(
                request.model_info,
                request.config
            )
        elif request.code_type == "gradio":
            code = code_generator.generate_gradio_app(
                request.model_info,
                request.config
            )
        elif request.code_type == "api":
            code = code_generator.generate_api_wrapper(
                request.model_info,
                request.config
            )
        elif request.code_type == "readme":
            code = code_generator.generate_readme(
                request.model_info,
                request.config,
                training_summary=request.config
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid code_type: {request.code_type}")
        
        return code
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/download-package/{job_id}")
async def download_package(job_id: str):
    try:
        logger.info(f"Creating download package for job: {job_id}")
        
        job_dir = Path(settings.outputs_dir) / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in job_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(job_dir)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=model_{job_id}.zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Package download failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/download-file/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    try:
        file_path = Path(settings.outputs_dir) / job_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = ErrorResponse(
        error=exc.detail,
        detail=None,
        status_code=exc.status_code
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        status_code=500
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
