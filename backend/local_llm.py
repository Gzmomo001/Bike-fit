from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import platform
import os

class LocalLLMProcessor:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """初始化本地LLM处理器"""
        self.device = self._get_optimal_device()
        print(f"Using device: {self.device}")
        
        # 设置环境变量以启用Metal性能优化
        if self.device == "mps":
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 加载分词器和模型
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True  # 使用快速分词器
        )
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._get_optimal_dtype(),
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 降低CPU内存使用
            use_cache=True  # 启用KV缓存
        ).to(self.device)
        
        # 如果不在训练，启用模型优化
        self.model.eval()
        
        # 如果是MPS设备，进行额外优化
        if self.device == "mps":
            self._optimize_for_mps()
        
        print("Model loaded successfully!")

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
            return torch.float16  # M1/M2 芯片支持 float16
        return torch.float32

    def _optimize_for_mps(self):
        """为MPS设备进行特定优化"""
        # 使用torch.compile进行优化（如果可用）
        if hasattr(torch, 'compile') and platform.processor() == 'arm':
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully for MPS acceleration!")
            except Exception as e:
                print(f"Model compilation failed: {e}")

    @torch.inference_mode()  # 比no_grad更严格的推理模式
    def generate_response(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1
    ) -> str:
        """生成回复，添加更多生成参数控制"""
        try:
            # 准备输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # 生成回复
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # 使用KV缓存
                repetition_penalty=1.1,  # 避免重复
                length_penalty=1.0,  # 长度惩罚
                early_stopping=True
            )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 移除原始提示
            response = response[len(prompt):].strip()
            
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
    # 初始化处理器
    processor = LocalLLMProcessor()
    
    # 测试提示
    test_prompt = """作为一个专业的自行车适配专家，请分析以下骑行姿势数据：
    - 膝关节角度: 80°
    - 髋关节角度: 95°
    - 肩膀角度: 30°
    
    请给出专业的建议。
    """
    
    print("\n测试提示:")
    print(test_prompt)
    
    print("\n生成回复:")
    response = processor.generate_response(test_prompt)
    print(response)
    
    # 清理缓存
    processor.clear_cache()

if __name__ == "__main__":
    test_local_llm() 