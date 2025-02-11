import os
from openai import OpenAI
from typing import Dict, Any, Tuple, Generator

class APIModelProcessor:
    def __init__(self, api_key: str):
        """初始化API模型处理器"""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def generate_response(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 1,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1
    ) -> Generator[Tuple[str, str], None, None]:
        """
        使用API生成流式回复
        返回: 生成器，每次返回(思考过程片段, 回复内容片段)的元组
        """
        try:
            stream = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_length,
                stream=True
            )
            
            for chunk in stream:
                if hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, 'reasoning_content', "") or ""
                    content = getattr(delta, 'content', "") or ""
                    if reasoning or content:
                        yield reasoning, content
                
        except Exception as e:
            yield "", f"API调用失败: {str(e)}"

def test_api_model():
    """测试API模型功能"""
    # 优先使用环境变量中的API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-4f1bb64e2d5b4099b69ceaf8ab0d8d72")
    processor = APIModelProcessor(api_key)
    
    test_prompt = "Hi, how are you?"
    
    print("\nTesting API model:")
    print("Prompt:", test_prompt)
    
    # 定义完整思考过程和回复内容
    full_reasoning = ""
    full_content = ""
    is_answering = False
    
    print("="*20 + "思考过程" + "="*20)
    for reasoning, content in processor.generate_response(test_prompt):
        # 处理思考过程
        if reasoning:
            print(reasoning, end="", flush=True)
            full_reasoning += reasoning
            
        # 处理回复内容
        if content:
            if not is_answering:
                print("\n" + "="*20 + "完整回复" + "="*20)
                is_answering = True
            print(content, end="", flush=True)
            full_content += content
    
    print()  # 添加最后的换行
    
    # 可选：打印完整内容
    # print("\n" + "="*20 + "完整思考过程" + "="*20)
    # print(full_reasoning)
    # print("="*20 + "完整回复" + "="*20)
    # print(full_content)

if __name__ == "__main__":
    test_api_model() 