import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_detection import pose_analyzer
from pose_detection.model import load_model_from_tfhub

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_video_path = os.path.join(current_dir, "uploads", "raw.mp4")

import cv2
import numpy as np
import unittest

class TestPoseAnalyzer(unittest.TestCase):
    def setUp(self):
        self.frames = []
        self.all_keypoints = []
        self.output_dir = os.path.join(current_dir, "outputs")
        
        # 读取测试视频并处理
        cap = cv2.VideoCapture(test_video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

        # 模拟关键点数据
        self.all_keypoints = [np.random.rand(17, 3) for _ in range(len(self.frames))]

    def test_draw_prediction_on_image(self):
        # 测试绘制关键点的功能
        for idx, keypoints in enumerate(self.all_keypoints):
            image_with_keypoints = pose_analyzer.draw_prediction_on_image(self.frames[idx], [(kp[0:-1], kp[-1]) for kp in keypoints])
            output_path = os.path.join(self.output_dir, f"frame_with_keypoints_{idx}.png")
            cv2.imwrite(output_path, image_with_keypoints)
            self.assertTrue(os.path.exists(output_path), f"未能保存带有关键点的图像: {output_path}")



import random

def save_random_slices(num_slices=5):
    if not os.path.exists(os.path.join(current_dir, "uploads")):
        os.makedirs(os.path.join(current_dir, "uploads"))

    cap = cv2.VideoCapture(test_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = random.sample(range(total_frames), num_slices)

    for frame_index in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(current_dir, "uploads", f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
        else:
            print(f"未能读取帧 {frame_index}")

    cap.release()



if __name__ == '__main__':
    unittest.main()
    # save_random_slices(5)

