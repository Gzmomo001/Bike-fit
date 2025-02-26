import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_detection.pose_analyzer import PoseAnalyzer

analyzer = PoseAnalyzer()
result = analyzer.test_pose_analyzer()
print(result)

