from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from imageio.config.extensions import video_extensions
from pose_detection import pose_analyzer
from bike_fit_advisor import BikeFitAdvisor

class App:
    def __init__(self):
        self.advisor = BikeFitAdvisor(use_api=True, api_key="sk-4f1bb64e2d5b4099b69ceaf8ab0d8d72")
        
    def run(self):
        result = pose_analyzer.test_pose_analyzer()
        self.advisor.test_advisor(measurements=result)

if __name__ == '__main__':
    app = App()
    app.run()