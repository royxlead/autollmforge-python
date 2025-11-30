import unittest
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.compute_estimator import ComputeEstimator

class TestComputeEstimator(unittest.TestCase):
    
    def test_estimate_model_memory_fp16(self):
        # 7B params * 2 bytes (fp16) = 14GB
        # But the estimator might have overheads or different calculations.
        # Let's check the implementation logic:
        # num_params * bytes_per_param / (1024**3)
        
        # Case 1: FP16 (2 bytes)
        # 1 billion params -> 2GB roughly
        memory = ComputeEstimator.estimate_model_memory(1_000_000_000, precision="fp16", quantization=None)
        self.assertAlmostEqual(memory, 1.86, delta=0.1) # 1 * 2 / 1.073... = 1.86

    def test_estimate_model_memory_4bit(self):
        # Case 2: 4bit (0.55 bytes per param in estimator logic)
        # 1 billion params
        memory = ComputeEstimator.estimate_model_memory(1_000_000_000, quantization="4bit")
        # 1 * 0.55 / 1.073... = 0.51
        self.assertAlmostEqual(memory, 0.51, delta=0.1)

    def test_estimate_training_memory_qlora(self):
        # Test QLoRA estimation
        # 7B model, batch 1, seq 512
        res = ComputeEstimator.estimate_training_memory(
            num_params=7_000_000_000,
            batch_size=1,
            seq_length=512,
            use_lora=True,
            gradient_checkpointing=True,
            quantization="4bit"
        )
        
        self.assertIn("total_memory_gb", res)
        self.assertIn("model_memory_gb", res)
        self.assertTrue(res["total_memory_gb"] > 0)
        
        # 7B 4bit model is ~3.5-4GB
        # Plus overheads. Should be around 6-8GB for minimal batch.
        self.assertTrue(4.0 < res["total_memory_gb"] < 10.0)

    def test_recommend_batch_size(self):
        # 7B model on 24GB VRAM (3090/4090)
        batch_size = ComputeEstimator.recommend_batch_size(
            num_params=7_000_000_000,
            available_vram_gb=24.0,
            seq_length=512,
            quantization="4bit"
        )
        self.assertTrue(batch_size >= 4)
        
        # 7B model on 10GB VRAM (3080)
        batch_size_small = ComputeEstimator.recommend_batch_size(
            num_params=7_000_000_000,
            available_vram_gb=10.0,
            seq_length=512,
            quantization="4bit"
        )
        self.assertTrue(batch_size_small < batch_size)

if __name__ == '__main__':
    unittest.main()
