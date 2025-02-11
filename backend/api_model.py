import os
from openai import OpenAI
from typing import Dict, Any, Tuple

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
    ) -> Tuple[str, str]:
        """
        使用API生成回复
        返回: (思考过程, 最终答案)的元组
        """
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_length
            )
            reasoning = completion.choices[0].message.reasoning_content
            content = completion.choices[0].message.content
            return reasoning, content
        except Exception as e:
            return "", f"API调用失败: {str(e)}"

def test_api_model():
    """测试API模型功能"""
    # 优先使用环境变量中的API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-4f1bb64e2d5b4099b69ceaf8ab0d8d72")
    processor = APIModelProcessor(api_key)
    
    test_prompt = "Hi, how are you?"
    
    print("\nTesting API model:")
    print("Prompt:", test_prompt)
    reasoning, response = processor.generate_response(test_prompt)
    
    print("\n思考过程：")
    print(reasoning if reasoning else "无思考过程")
    print("\n最终答案：")
    print(response)

if __name__ == "__main__":
    test_api_model() 