import tensorflow as tf
import numpy as np
from moviepy import VideoFileClip
import cv2
import tempfile

#压缩视频质量
def reduce_video_quality(video_path, max_pixels, max_fps, max_duration):

    clip = VideoFileClip(video_path, audio=False)

    max_duration = clip.duration / 2 if max_duration > clip.duration else max_duration
    # Reduce fps
    max_fps = min(clip.fps, max_fps)
    clip.fps = max_fps
    # Reduce resolution
    max_pixels = min(min(clip.h, clip.w), max_pixels)
    clip = (
        clip.resized(height=max_pixels)
        if clip.h < clip.w
        else clip.resized(width=max_pixels)
    )
    # Reduce duration
    mid_point = clip.duration / 2
    lower_point = mid_point - max_duration / 2
    upper_point = mid_point + max_duration / 2
    clip = clip.subclipped(lower_point, upper_point)
    print(
        f"Clip with fps: {clip.fps} - width: {clip.w} - height: {clip.h} - duration: {clip.duration}"
    )
    return clip
#将视频帧图像转换为张量
def load_tensors_from_clip(videofileclip):
    # convert to uint8 array of frames
    video = tf.convert_to_tensor(
        np.array(list(videofileclip.iter_frames())), dtype=tf.uint8
    )
    return video

def pre_process_video(file:str|bytes)->tuple:
    """
    使用OpenCV预处理视频文件。
    
    参数:
    file (str): 视频文件的路径
    file (bytes): 视频文件的字节数据

    
    返回:
    tuple: 包含处理后的视频帧和张量数据
    """
    # 打开视频文件
    # 检查输入参数的类型
    if isinstance(file, str):
        # 如果是文件路径，直接打开视频文件
        cap = cv2.VideoCapture(file)
    elif isinstance(file, bytes):
        # 使用临时文件保存视频
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(file)  # 写入视频数据
            temp_video.flush()  # 确保数据写入文件
            cap = cv2.VideoCapture(temp_video.name)  # 读取临时文件
    else:
        raise ValueError("file 参数必须是字符串路径或字节数据")

    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 限制处理帧数
    max_frames = min(total_frames, fps * 10)  # 最多处理10秒的视频
    target_size = (256, 256)  # 目标尺寸
    
    frames = []
    tensors = []
    
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调整帧大小
        frame = cv2.resize(frame, target_size)
        # 转换颜色空间从BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
        # 转换为张量
        tensor = tf.convert_to_tensor(frame)
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.cast(tensor, dtype=tf.int32)
        tensors.append(tensor)
    
    cap.release()
    
    return frames, tf.concat(tensors, axis=0)