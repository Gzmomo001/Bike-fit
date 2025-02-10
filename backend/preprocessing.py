import tensorflow as tf
import numpy as np
from moviepy import VideoFileClip
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