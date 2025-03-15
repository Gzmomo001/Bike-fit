import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pose_detection.pose_analyzer import PoseAnalyzer
from bike_fit_advisor import BikeFitAdvisor
import os
import dotenv
import json
from Light_rag.lightrag_ollama_demo import LightRagManager
import ollama
from openai import OpenAI
import logging
import uvicorn

# 配置日志格式
logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    force=True  # 强制重新配置日志
)

# 获取logger
logger = logging.getLogger(__name__)

# from local_rag.chat import get_model_response

dotenv.load_dotenv()

# 从环境变量获取配置
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:72b-instruct-q5_1")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_MODEL = os.getenv("API_MODEL", "qwen-plus")

# 初始化OpenAI客户端
if not USE_LOCAL_OLLAMA:
    openai_client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

app = FastAPI(root_path="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"],
)

# 允许跨域预检请求
@app.options("/analyze/video")
async def options_analyze_video():
    return {"message": "OK"}

# print("API_KEY:", os.getenv("DASHSCOPE_API_KEY"))
# bike_advisor = BikeFitAdvisor(use_api=True, api_key=os.getenv("DASHSCOPE_API_KEY"))
# bike_advisor = BikeFitAdvisor(use_api=False)

if_useRAG = True
@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    analyzer = PoseAnalyzer()
    # 使用pose_analyzer返回的GIF路径
    result, result_frames, gif_path = analyzer.pose_analyzer(video_bytes)
    
    # 记录结果信息
    logger.info(f"分析结果: {result}")
    logger.info(f"结果帧数量: {len(result_frames) if result_frames else 0}")
    logger.info(f"GIF路径: {gif_path}")
    
    # 读取GIF文件内容
    import base64
    
    if gif_path and os.path.exists(gif_path):
        try:
            with open(gif_path, "rb") as f:
                gif_bytes = f.read()
            logger.info(f"读取GIF文件成功: {gif_path}, 大小: {len(gif_bytes)} 字节")
            
            # 验证GIF文件格式
            if gif_bytes[:6] in [b'GIF87a', b'GIF89a']:
                logger.info("GIF文件格式有效")
            else:
                logger.warning(f"GIF文件格式可能无效，前6个字节: {gif_bytes[:6]}")
        except Exception as e:
            logger.error(f"读取GIF文件失败: {str(e)}")
            gif_bytes = None
    else:
        # 如果没有GIF路径或文件不存在，则生成一个新的GIF
        import tempfile
        from pose_detection.pose_analyzer import to_gif
        
        logger.info("GIF路径不存在，生成新的GIF")
        gif_bytes = None
        
        try:
            # 检查结果帧是否有效
            if result_frames and len(result_frames) > 0:
                logger.info(f"结果帧数量: {len(result_frames)}")
                logger.info(f"第一帧形状: {result_frames[0].shape}")
                
                # 确保所有帧都是RGB格式
                valid_frames = []
                for i, frame in enumerate(result_frames):
                    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                        valid_frames.append(frame)
                    else:
                        logger.warning(f"跳过无效帧 {i}: {frame.shape if frame is not None else 'None'}")
                
                logger.info(f"有效帧数量: {len(valid_frames)}")
                
                if valid_frames:
                    # 创建临时文件用于保存GIF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_gif:
                        gif_path = temp_gif.name
                    
                    # 生成GIF文件
                    to_gif(valid_frames, fps=10, output_path=gif_path)
                    logger.info(f"GIF生成成功: {gif_path}")
                    
                    # 读取GIF文件内容
                    with open(gif_path, "rb") as f:
                        gif_bytes = f.read()
                    logger.info(f"读取新生成的GIF文件成功，大小: {len(gif_bytes)} 字节")
                else:
                    logger.error("没有有效的帧用于生成GIF")
            else:
                logger.error("没有结果帧用于生成GIF")
        except Exception as e:
            logger.error(f"生成GIF失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    # 删除临时文件
    if gif_path and os.path.exists(gif_path):
        try:
            os.unlink(gif_path)
            logger.info(f"临时GIF文件已删除: {gif_path}")
        except Exception as e:
            logger.warning(f"删除临时GIF文件失败: {str(e)}")
    
    def generate_streaming_response():
        # 首先发送GIF数据的base64编码
        if gif_bytes:
            try:
                gif_base64 = base64.b64encode(gif_bytes).decode('utf-8')
                logger.info(f"GIF base64编码大小: {len(gif_base64)} 字符")
                yield json.dumps({"type": "gif", "data": gif_base64}) + "\n"
                logger.info("GIF数据已发送")
            except Exception as e:
                logger.error(f"编码或发送GIF数据失败: {str(e)}")
                # 发送错误信息
                yield json.dumps({"type": "error", "message": f"GIF处理失败: {str(e)}"}) + "\n"
        else:
            logger.warning("没有GIF数据可发送")
            # 发送错误信息
            yield json.dumps({"type": "error", "message": "无法生成GIF"}) + "\n"
        
        # 确保result是一个有效的JSON对象
        if isinstance(result, dict):
            result_data = result
        else:
            # 如果不是字典，尝试转换为字典
            try:
                result_data = json.loads(result) if isinstance(result, str) else {"error": "无效的结果数据"}
            except:
                result_data = {"error": "无法解析结果数据"}
        
        # 在发送前记录详细的结果数据信息
        logger.info(f"测量结果数据类型: {type(result_data)}")
        logger.info(f"测量结果数据内容: {json.dumps(result_data, ensure_ascii=False)}")
        
        # 然后发送测量结果
        logger.info(f"发送测量结果: {result_data}")
        yield json.dumps({"type": "info", "message": result_data}, ensure_ascii=False) + "\n"
        logger.info("测量结果已发送")
        
        if if_useRAG:
            logger.info("Using RAG for analysis")
            advisor = BikeFitAdvisor(measurements=result)
            measurement_text = advisor.generate_prompt()
            query_text = rag.query(measurement_text)

            logger.info("Query completed")
            # logger.info(f"Query result: {query_text}")
            template_prompt = advisor.generate_prompt(query_text)
            
            if USE_LOCAL_OLLAMA:
                logger.info("Using local Ollama model")
                stream = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': template_prompt}],
                    stream=True
                )
                for chunk in stream:
                    content = chunk["message"]["content"]
                    yield json.dumps({"type": "response", "message": content}) + "\n"
            else:
                logger.info("Using online API service")
                try:
                    completion = openai_client.chat.completions.create(
                        model=API_MODEL,
                        messages=[
                            {'role': 'system', 'content': '你是一个专业的自行车fitter。在你收到的prompt中,你会先看到rag查询的内容，然后是用户的骑行姿势设定。你需要对查询到的权威知识对用户的骑行自行进行评定。1.判断现在的设定是否属于有伤病风险。2.判断现在的设定是否属于休闲还是激进。3.如果想要休闲，应该如何调整。4.如果想要激进，应该如何调整。5.如何调整设定能减少伤病。每一步都需要引用权威知识，以提供用户参考。'},
                            {'role': 'user', 'content': template_prompt}
                        ],
                        stream=True
                    )
                    for chunk in completion:
                        if hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                yield json.dumps({"type": "response", "message": content}) + "\n"
                except Exception as e:
                    error_message = f"API call failed: {str(e)}"
                    logger.error(error_message)
                    yield json.dumps({"type": "error", "message": error_message}) + "\n"
        else:
            pass

    return StreamingResponse(generate_streaming_response(), media_type="application/json")

if __name__ == "__main__":
    logger.info("Initializing LightRagManager...")
    rag = LightRagManager(os.getenv("VectorData_path"))
    logger.info("Starting FastAPI server...")
    
    # 配置uvicorn的日志
    log_config = {
        "version": 1,
        "formatters": {
            "default": {
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "access": {
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr"
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False}
        }
    }
    
    uvicorn.run(app, 
                host="0.0.0.0", 
                port=8000,
                log_config=log_config)


