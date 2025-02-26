import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_detection.pose_analyzer import pose_analyzer

def test_pose_analyzer():
    video_path = "./uploads/raw.mp4"
    result = pose_analyzer.test_pose_analyzer(video_path)
    print(result)

test_pose_analyzer()
