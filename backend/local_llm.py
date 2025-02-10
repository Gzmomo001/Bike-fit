from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import platform
import os
import warnings
from huggingface_hub import HfFolder, HfApi
import requests
import ssl
import urllib3
import shutil

# 禁用SSL验证警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LocalLLMProcessor:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):  # 使用更小的模型
        """初始化本地LLM处理器"""
        self.device = self._get_optimal_device()
        print(f"Using device: {self.device}")
        
        # 设置环境变量以启用Metal性能优化
        if self.device == "mps":
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 设置本地模型目录
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 设置具体的模型目录
        self.local_model_path = os.path.join(self.model_dir, model_name.split('/')[-1])
        
        # 如果本地模型目录不存在，创建它
        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)
            print(f"创建本地模型目录: {self.local_model_path}")
        
        print("Loading tokenizer...")
        try:
            # 尝试从本地加载
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
            # 保存到本地
            self.tokenizer.save_pretrained(self.local_model_path)
            print(f"Tokenizer已保存到: {self.local_model_path}")
        
        print("Loading model...")
        try:
            # 尝试从本地加载
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
            # 保存到本地
            self.model.save_pretrained(self.local_model_path)
            print(f"模型已保存到: {self.local_model_path}")
        
        # 创建生成管道
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        print("Model loaded successfully!")

    @staticmethod
    def download_model(model_id: str, local_dir: str):
        """下载模型到指定目录"""
        try:
            # 使用transformers的from_pretrained下载
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # 保存到本地
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
            
            print(f"模型已成功下载到: {local_dir}")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False

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

    @torch.inference_mode()
    def generate_response(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1
    ) -> str:
        """生成回复"""
        try:
            # 使用pipeline生成
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
                early_stopping=False,  # 当num_beams=1时，禁用early_stopping
                truncation=True  # 显式启用truncation
            )
            
            # 获取生成的文本
            generated_text = outputs[0]['generated_text']
            
            # 移除原始提示
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

def test_local_llm():
    """测试本地LLM功能"""
    processor = LocalLLMProcessor()
    
    test_prompt = """Hi"""
    
    print("\nTest prompt:")
    print(test_prompt)
    
    print("\nGenerating response:")
    response = processor.generate_response(test_prompt)
    print(response)
    
    processor.clear_cache()

if __name__ == "__main__":
    test_local_llm() 