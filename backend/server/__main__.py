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

@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    result = pose_analyzer.pose_analyzer(video_bytes)
    def generate_streaming_response():
        for message in bike_advisor.stream_advisor(measurements=result):
            yield json.dumps(message) + "\n"
            
    return StreamingResponse(generate_streaming_response(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


