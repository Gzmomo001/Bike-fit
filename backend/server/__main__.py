import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from pose_detection import pose_analyzer
from bike_fit_advisor import BikeFitAdvisor
import os
import dotenv
import json

from local_rag.chat import get_model_response

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

print("API_KEY:", os.getenv("DASHSCOPE_API_KEY"))
bike_advisor = BikeFitAdvisor(use_api=True, api_key=os.getenv("DASHSCOPE_API_KEY"))

if_useRAG = True
@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    result = pose_analyzer.pose_analyzer(video_bytes)
    def generate_streaming_response():
        if if_useRAG:
            # Convert measurements to text format for RAG
            measurement_text = bike_advisor.generate_prompt(result)
            
            # Create a mock history with just the current question
            history = [[measurement_text, None]]
            
            # Use local_rag's get_model_response
            for response in get_model_response(
                {'text': measurement_text, 'files': []},
                history,
                model='qwen-max',
                temperature=0.7,
                max_tokens=1024,
                history_round=1,
                db_name='bike-fit',
                similarity_threshold=0.2,
                chunk_cnt=5
            ):
                yield json.dumps({"response": response[0][-1][-1]}) + "\n"
        else:
            for message in bike_advisor.stream_advisor(measurements=result):
                yield json.dumps(message) + "\n"
            
    return StreamingResponse(generate_streaming_response(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


