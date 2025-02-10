from local_llm import LocalLLMProcessor
import os

def main():
    """下载模型到本地"""
    # 设置本地模型目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置要下载的模型ID
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 使用更小的模型
    local_model_path = os.path.join(model_dir, model_id.split('/')[-1])
    
    print(f"开始下载模型 {model_id} 到 {local_model_path}")
    
    # 下载模型
    success = LocalLLMProcessor.download_model(model_id, local_model_path)
    
    if success:
        print("模型下载完成！")
    else:
        print("模型下载失败，请检查网络连接或尝试使用代理。")

if __name__ == "__main__":
    main() 