from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from imageio.config.extensions import video_extensions

from main import upload_video


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def receive_video():
    print("Received upload request")
    if 'video' not in request.files:
        print("No video file provided")
        return jsonify({"error": "No video file provided"}), 400
    video = request.files['video']
    if video.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # 保存上传的文件
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], "raw.mp4")
    video.save(video_path)
    print(f"Video saved to {video_path}")

    result = upload_video(video_path)
    print(f"Backend processing result: {result}")

    return jsonify({"message": f"{video.filename} uploaded successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
