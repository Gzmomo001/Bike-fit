from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import platform
import os
import warnings
from huggingface_hub import HfFolder, HfApi
import shutil

class LocalModelProcessor:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """初始化本地模型处理器"""
        self.device = self._get_optimal_device()
        print(f"Using local model on device: {self.device}")
        
        # 设置环境变量以启用Metal性能优化
        if self.device == "mps":
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 设置本地模型目录
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 设置具体的模型目录
        self.local_model_path = os.path.join(self.model_dir, model_name.split('/')[-1])
        
        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)
            print(f"创建本地模型目录: {self.local_model_path}")
        
        self._load_local_model(model_name)

    def _load_local_model(self, model_name: str):
        """加载本地模型"""
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                trust_remote_code=True,
                use_fast=True
            )
            print("成功从本地加载tokenizer")
        except Exception as e:
            print(f"本地加载失败: {e}")
            print("尝试从在线加载并保存到本地...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            self.tokenizer.save_pretrained(self.local_model_path)
        
        print("Loading model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                torch_dtype=self._get_optimal_dtype(),
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            print("成功从本地加载模型")
        except Exception as e:
            print(f"本地加载失败: {e}")
            print("尝试从在线加载并保存到本地...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._get_optimal_dtype(),
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.save_pretrained(self.local_model_path)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        print("Model loaded successfully!")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1
    ) -> str:
        """使用本地模型生成回复"""
        try:
            outputs = self.pipe(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=False,
                truncation=True
            )
            
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            return f"生成回复时出错: {str(e)}"

    def clear_cache(self):
        """清理缓存"""
        if hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _get_optimal_device(self) -> str:
        """获取最优的计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and platform.processor() == 'arm':
            return "mps"
        return "cpu"

    def _get_optimal_dtype(self) -> torch.dtype:
        """获取最优的数据类型"""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float16
        return torch.float32

def test_local_model():
    """测试本地模型功能"""
    processor = LocalModelProcessor()
    
    test_prompt = "Hi, how are you?"
    
    print("\nTesting local model:")
    print("Prompt:", test_prompt)
    response = processor.generate_response(test_prompt)
    print("Response:", response)
    
    processor.clear_cache()

if __name__ == "__main__":
    test_local_model() 