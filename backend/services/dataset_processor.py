import json
import csv
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from models.schemas import DatasetInfo
from utils.logger import get_logger
from config import settings
import math

logger = get_logger(__name__)


class DatasetProcessor:
    def __init__(self):
        self.datasets_dir = Path(settings.datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    async def validate_dataset(
        self,
        dataset_path: str,
        format: str = "json",
        dataset_id: Optional[str] = None
    ) -> DatasetInfo:
        logger.info(f"Validating dataset: {dataset_path}")
        
        try:
            if format == "json":
                data = self.load_json(dataset_path)
            elif format == "jsonl":
                data = self.load_jsonl(dataset_path)
            elif format == "csv":
                data = self.load_csv(dataset_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            validation_warnings = self.validate_structure(data)
            
            stats = self.calculate_statistics(data)
            
            preview = data[:5] if len(data) > 5 else data
            
            columns = list(data[0].keys()) if data else []
            
            file_path = Path(dataset_path)
            size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
            
            num_samples = len(data)
            num_train = int(num_samples * 0.9)
            num_val = num_samples - num_train
            
            dataset_info = DatasetInfo(
                dataset_id=dataset_id or file_path.stem,
                num_samples=num_samples,
                num_train_samples=num_train,
                num_validation_samples=num_val,
                avg_tokens=stats["avg_tokens"],
                max_tokens=stats["max_tokens"],
                min_tokens=stats["min_tokens"],
                data_preview=preview,
                columns=columns,
                validation_warnings=validation_warnings,
                format=format,
                size_mb=round(size_mb, 2)
            )
            
            logger.info(f"Dataset validated: {num_samples} samples, avg {stats['avg_tokens']:.0f} tokens")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise Exception(f"Dataset validation error: {str(e)}")
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            for key in ['data', 'examples', 'train', 'samples']:
                if key in data:
                    data = data[key]
                    break
        
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of examples")
        
        return data
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def validate_structure(self, data: List[Dict[str, Any]]) -> List[str]:
        warnings = []
        
        if not data:
            warnings.append("Dataset is empty")
            return warnings
        
        first_item = data[0]
        has_text_field = any(
            key in first_item 
            for key in ['text', 'prompt', 'instruction', 'input', 'question', 'content']
        )
        
        if not has_text_field:
            warnings.append("No standard text field found (text, prompt, instruction, etc.)")
        
        text_fields = [k for k in first_item.keys() if isinstance(first_item.get(k), str)]
        if text_fields:
            short_count = 0
            for item in data[:min(100, len(data))]:
                for field in text_fields:
                    text = str(item.get(field, ''))
                    if len(text) < 10:
                        short_count += 1
                        break
            
            if short_count > len(data) * 0.1:
                warnings.append(f"Many examples are very short (<10 chars) - {short_count} found in sample")
        
        keys_set = set(first_item.keys())
        inconsistent_count = 0
        for item in data[:min(100, len(data))]:
            if set(item.keys()) != keys_set:
                inconsistent_count += 1
        
        if inconsistent_count > 0:
            warnings.append(f"Inconsistent fields across examples - {inconsistent_count} items differ")
        
        null_count = 0
        for item in data[:min(100, len(data))]:
            if any(v is None or v == '' for v in item.values()):
                null_count += 1
        
        if null_count > 0:
            warnings.append(f"Found {null_count} examples with null/empty values in sample")
        
        return warnings
    
    def calculate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not data:
            return {
                "avg_tokens": 0,
                "max_tokens": 0,
                "min_tokens": 0
            }
        
        token_counts = []
        
        for item in data:
            text = ' '.join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
            word_count = len(text.split())
            token_count = int(word_count * 1.3)
            token_counts.append(token_count)
        
        return {
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0
        }
    
    def preprocess_for_training(
        self,
        data: List[Dict[str, Any]],
        task_type: str,
        text_field: str = "text",
        label_field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        processed = []
        
        for item in data:
            processed_item = {}
            
            if text_field in item:
                processed_item['text'] = item[text_field]
            elif 'prompt' in item and 'completion' in item:
                processed_item['text'] = f"{item['prompt']}\n\n{item['completion']}"
            elif 'instruction' in item:
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item.get('output', '')
                
                if input_text:
                    processed_item['text'] = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    processed_item['text'] = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                text_fields = [k for k, v in item.items() if isinstance(v, str)]
                if text_fields:
                    processed_item['text'] = item[text_fields[0]]
            
            if label_field and label_field in item:
                processed_item['label'] = item[label_field]
            
            if 'text' in processed_item:
                processed.append(processed_item)
        
        return processed
    
    def create_train_val_split(
        self,
        data: List[Dict[str, Any]],
        split_ratio: float = 0.1
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - split_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    async def save_dataset(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        format: str = "json"
    ) -> str:
        output_path = self.datasets_dir / filename
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported save format: {format}")
        
        logger.info(f"Dataset saved to {output_path}")
        return str(output_path)
