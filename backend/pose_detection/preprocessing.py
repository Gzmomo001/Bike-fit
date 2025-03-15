import tensorflow as tf
import numpy as np
from moviepy import VideoFileClip
import cv2
import tempfile
import os

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
    temp_video = None
    video_path = None
    
    try:
        # 打开视频文件
        # 检查输入参数的类型
        if isinstance(file, str):
            # 如果是文件路径，直接打开视频文件
            video_path = file
            cap = cv2.VideoCapture(video_path)
        elif isinstance(file, bytes):
            # 使用临时文件保存视频
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(file)  # 写入视频数据
            temp_video.flush()  # 确保数据写入文件
            video_path = temp_video.name
            cap = cv2.VideoCapture(video_path)  # 读取临时文件
        else:
            raise ValueError("file 参数必须是字符串路径或字节数据")

        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # 如果无法获取fps，使用默认值30
            
        # 手动计算总帧数
        total_frames = 0
        temp_cap = cv2.VideoCapture(video_path)
        while temp_cap.read()[0]:
            total_frames += 1
        temp_cap.release()
        
        # 重新打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 计算采样间隔，使总帧数为30
        target_frames = 30
        sample_interval = max(1, total_frames // target_frames) if total_frames > 0 else 1
        
        frames = []
        tensors = []
        target_size = 256  # 目标尺寸的边长
        frame_count = 0
        
        # 获取第一帧来确定视频尺寸
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")
            
        # 重新设置视频位置到开始
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 获取原始尺寸
        original_height, original_width = first_frame.shape[:2]
        
        # 计算保持比例的新尺寸
        if original_height > original_width:
            # 高度大于宽度，以高度为基准
            new_height = target_size
            new_width = int(original_width * (target_size / original_height))
            padding_left = (target_size - new_width) // 2
            padding_right = target_size - new_width - padding_left
            padding_top = 0
            padding_bottom = 0
        else:
            # 宽度大于或等于高度，以宽度为基准
            new_width = target_size
            new_height = int(original_height * (target_size / original_width))
            padding_top = (target_size - new_height) // 2
            padding_bottom = target_size - new_height - padding_top
            padding_left = 0
            padding_right = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 只处理采样帧
            if frame_count % sample_interval == 0 and len(frames) < target_frames:
                # 保持原始比例调整大小
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # 创建一个黑色背景的正方形图像
                square_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                
                # 将调整大小后的图像放在正方形中间
                square_frame[padding_top:padding_top+new_height, padding_left:padding_left+new_width] = resized_frame
                
                # 转换颜色空间从BGR到RGB
                square_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
                
                frames.append(square_frame)
                # 转换为张量
                tensor = tf.convert_to_tensor(square_frame)
                tensor = tf.expand_dims(tensor, axis=0)
                tensor = tf.cast(tensor, dtype=tf.int32)
                tensors.append(tensor)
                
            frame_count += 1
            
            # 如果已经采集了足够的帧，就退出
            if len(frames) >= target_frames:
                break
        
        cap.release()
        
        # 如果采集的帧数不足30帧，通过复制最后一帧来补足
        if frames:  # 确保至少有一帧
            while len(frames) < target_frames:
                frames.append(frames[-1])
                tensors.append(tensors[-1])
        else:
            raise ValueError("无法从视频中提取帧")
        
        return frames, tf.concat(tensors, axis=0)
        
    finally:
        # 清理临时文件
        if temp_video is not None:
            temp_video.close()
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)

