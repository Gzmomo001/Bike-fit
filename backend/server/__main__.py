import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pose_detection.pose_analyzer import PoseAnalyzer
from bike_fit_advisor import BikeFitAdvisor
import os
import dotenv
import json
from backend.Light_rag.lightrag_ollama_demo import LightRagManager
import ollama

# from local_rag.chat import get_model_response

dotenv.load_dotenv()

app = FastAPI(root_path="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# print("API_KEY:", os.getenv("DASHSCOPE_API_KEY"))
# bike_advisor = BikeFitAdvisor(use_api=True, api_key=os.getenv("DASHSCOPE_API_KEY"))
# bike_advisor = BikeFitAdvisor(use_api=False)

if_useRAG = True
@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    analyzer = PoseAnalyzer()
    # TODO 需要增加帧返回值
    result, result_frames = analyzer.pose_analyzer(video_bytes)
    # result_frames = result_frames[:2] + result_frames[-2:]
    def generate_streaming_response():
        yield json.dumps({"type": "info", "message": result}) + "\n"
        if if_useRAG:
            print("Using RAG")
            # Convert measurements to text format for RAG
            advisor = BikeFitAdvisor(measurements=result)
            measurement_text = advisor.generate_prompt()
            query_text = rag.query(measurement_text)

            print("完成query")
            template_prompt = advisor.generate_prompt(query_text)
            stream = ollama.chat(
                model='qwen2.5:32B',
                messages=[{'role': 'user', 'content': template_prompt}],
                stream=True
            )
            # 逐块打印响应内容
            for chunk in stream:
                content =  chunk["message"]["content"]
                yield json.dumps({"type": "response", "message": content}) + "\n"
        else:
            pass

    return StreamingResponse(generate_streaming_response(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    rag = LightRagManager(os.getenv("VectorData_path"))
    uvicorn.run(app, host="0.0.0.0", port=8000)


