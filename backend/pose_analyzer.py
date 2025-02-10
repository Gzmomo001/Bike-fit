from flask import Flask, request, jsonify
import os
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
##import cv2
import kagglehub

from preprocessing import reduce_video_quality,load_tensors_from_clip
from model import load_model_from_tfhub, get_keypoints_from_video
from postprocessing import (find_camera_facing_side,
                            get_front_keypoint_indices,
                            get_lowest_pedal_frames,
                            filter_bad_knee_angles,
                            get_hip_knee_ankle_angle,
                            get_highest_pedal_frames,
                            calculate_angle)

"""
    初始化函数，用于加载MoveNet Thunder模型。

    该函数通过调用load_model_from_tfhub函数来加载模型。
    并且初始化两个全局变量：model和input_size，分别用于存储加载的模型和模型输入的大小。
    """
def init():

    global model, input_size
    model, input_size = load_model_from_tfhub()

def upload_video(path):
    init()

    clip, tensors = pre_process_video(path)

    all_keypoints = get_keypoints_from_video(tensors, model, input_size)

    result = get_pose(all_keypoints)

    return result

#视频前处理
def pre_process_video(file_path):
    """
    预处理视频文件。

    本函数旨在降低视频文件的质量，以便于后续处理。它首先减少视频的像素数、帧率和时长，
    然后从处理后的视频中加载张量数据。这样做可以优化视频处理性能，同时减少计算资源的消耗。

    参数:
    file_path (str): 视频文件的路径。

    返回:
    tuple: 包含处理后的视频剪辑和从该剪辑中加载的张量数据。
    """
    # 减少视频质量，包括最大像素数、帧率和时长
    clip = reduce_video_quality(file_path, max_pixels=256, max_fps=30, max_duration=10)

    # 从处理后的视频剪辑中加载张量数据
    tensors = load_tensors_from_clip(clip)

    return clip, tensors

#找到膝盖角度的最小值和对应的帧率，并返回平均值和方差
def knee_pose_process_video(all_keypoints):
    facing_direction = find_camera_facing_side(all_keypoints[0])

    hip_knee_ankle_indices = get_front_keypoint_indices(facing_direction)[:4]  # 存储拍摄视角侧的髋部，膝盖和脚踝的位置索引

    # 获取所有帧的膝盖角度
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 找到踏板角度最低的帧
    lowest_pedal_point_indices = get_lowest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    # 从所有踏板角度中找到最位置最低的帧的膝盖角度
    angles_at_lowest_pedal_points = [
        knee_angles[i] for i in lowest_pedal_point_indices
    ]

    # 排除不正常的角度和帧，进一步增加稳定性
    filtered_angles, filtered_indices = filter_bad_knee_angles(
        angles_at_lowest_pedal_points, lowest_pedal_point_indices
    )

    # 取最小膝盖角度的平均数和方差
    if len(filtered_angles) > 0:
        angle_avg = np.mean(filtered_angles)
        angle_std = np.std(filtered_angles)
    else:
        angle_avg = 0
        angle_std = 0

    return (
        angle_avg,
        angle_std,
        filtered_indices,
        knee_angles,
    )

#获取髋，膝盖，肩膀，手肘的角度
def get_pose(all_keypoints):
    facing_direction = find_camera_facing_side(all_keypoints[0])
    front_indices = get_front_keypoint_indices(facing_direction)

    # 获取膝盖最低点角度的平均数
    knee_angle_lowest = get_knee_angle_at_lowest_pedal_points_avg(all_keypoints)
    
    # 获取膝盖最高点角度的平均数
    knee_angle_highest = get_knee_angle_at_highest_pedal_points_avg(all_keypoints)
    
    # 获取肩膀角度的平均数
    shoulder_angle = get_shoulder_angle_avg(all_keypoints, front_indices)
    
    # 获取手肘角度的平均数
    elbow_angle = get_elbow_angle_avg(all_keypoints, front_indices)

    # 获取髋关节角度的平均数
    hip_angle = get_hip_angle_avg(all_keypoints, front_indices)

    # 获取所有帧的膝盖角度（用于绘图）
    hip_knee_ankle_indices = front_indices[:4]
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 创建测量结果字典
    measurements = {
        'knee_angle_lowest': knee_angle_lowest,
        'knee_angle_highest': knee_angle_highest,
        'shoulder_angle': shoulder_angle,
        'elbow_angle': elbow_angle,
        'hip_angle': hip_angle,
        'knee_angles': knee_angles,
    }

    # 使用LLM分析姿势
    try:
        from llm_analysis import BikeFitAnalyzer
        analyzer = BikeFitAnalyzer()
        
        # 获取详细分析
        analysis = analyzer.analyze_pose(measurements)
        
        # 获取快速总结
        summary = analyzer.get_quick_summary(measurements)
        
        # 将分析结果添加到返回数据中
        measurements['analysis'] = analysis
        measurements['summary'] = summary
    except Exception as e:
        measurements['analysis'] = f"无法生成分析：{str(e)}"
        measurements['summary'] = {}

    return measurements

#获取膝盖最大角度的平均数
def get_knee_angle_at_lowest_pedal_points_avg(all_keypoints):
    facing_direction = find_camera_facing_side(all_keypoints[0])

    hip_knee_ankle_indices = get_front_keypoint_indices(facing_direction)[:4]  # 存储拍摄视角侧的髋部，膝盖和脚踝的位置索引

    # 获取所有帧的膝盖角度
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 找到踏板角度最低的帧
    lowest_pedal_point_indices = get_lowest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    # 从所有踏板角度中找到最位置最低的帧的膝盖角度
    angles_at_lowest_pedal_points = [
        knee_angles[i] for i in lowest_pedal_point_indices
    ]

    # 排除不正常的角度和帧，进一步增加稳定性
    filtered_angles, filtered_indices = filter_bad_knee_angles(
        angles_at_lowest_pedal_points, lowest_pedal_point_indices
    )

    # 取最小膝盖角度的平均数和方差
    if len(filtered_angles) > 0:
        angle_avg = np.mean(filtered_angles)
        angle_std = np.std(filtered_angles)
    else:
        angle_avg = 0
        angle_std = 0

    return angle_avg

#获取膝盖最小角度的平均数

def get_knee_angle_at_highest_pedal_points_avg(all_keypoints):
    facing_direction = find_camera_facing_side(all_keypoints[0])

    hip_knee_ankle_indices = get_front_keypoint_indices(facing_direction)[:4]  # 存储拍摄视角侧的髋部，膝盖和脚踝的位置索引

    # 获取所有帧的膝盖角度
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 找到踏板角度最高的帧
    highest_pedal_point_indices = get_highest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    # 从所有踏板角度中找到最位置最高的帧的膝盖角度
    angles_at_highest_pedal_points = [
        knee_angles[i] for i in highest_pedal_point_indices
    ]

    # 排除不正常的角度和帧，进一步增加稳定性
    filtered_angles, filtered_indices = filter_bad_knee_angles(
        angles_at_highest_pedal_points, highest_pedal_point_indices
    )

    # 取最小膝盖角度的平均数和方差
    if len(filtered_angles) > 0:
        angle_avg = np.mean(filtered_angles)
        angle_std = np.std(filtered_angles)
    else:
        angle_avg = 0
        angle_std = 0

    return angle_avg

#获取肩膀角度的平均数
def get_shoulder_angle_avg(all_keypoints, front_indices):
    hip_index = front_indices[0]  # 髋关节索引
    shoulder_index = front_indices[3]  # 肩膀索引
    elbow_index = front_indices[4]  # 手肘索引

    # 计算所有帧的肩膀角度
    shoulder_angles = []
    for kp in all_keypoints:
        # 获取三个点的坐标
        [hip_y, hip_x] = kp[hip_index][0:-1]
        [shoulder_y, shoulder_x] = kp[shoulder_index][0:-1]
        [elbow_y, elbow_x] = kp[elbow_index][0:-1]
        
        # 计算角度
        angle = calculate_angle((hip_y, hip_x), (shoulder_y, shoulder_x), (elbow_y, elbow_x))
        shoulder_angles.append(angle)

    # 取所有肩膀角度的平均值
    if len(shoulder_angles) > 0:
        shoulder_angle_avg = np.mean(shoulder_angles)
    else:
        shoulder_angle_avg = 0

    return shoulder_angle_avg

#获取手肘角度的平均数
def get_elbow_angle_avg(all_keypoints, front_indices):
    shoulder_index = front_indices[3]  # 肩膀索引
    elbow_index = front_indices[4]  # 手肘索引
    wrist_index = front_indices[5]  # 手腕索引

    # 计算所有帧的手肘角度
    elbow_angles = []
    for kp in all_keypoints:
        # 获取三个点的坐标
        [shoulder_y, shoulder_x] = kp[shoulder_index][0:-1]
        [elbow_y, elbow_x] = kp[elbow_index][0:-1]
        [wrist_y, wrist_x] = kp[wrist_index][0:-1]
        
        # 计算角度
        angle = calculate_angle((shoulder_y, shoulder_x), (elbow_y, elbow_x), (wrist_y, wrist_x))
        elbow_angles.append(angle)

    # 取所有手肘角度的平均值
    if len(elbow_angles) > 0:
        elbow_angle_avg = np.mean(elbow_angles)
    else:
        elbow_angle_avg = 0

    return elbow_angle_avg

#获取髋关节角度的平均数
def get_hip_angle_avg(all_keypoints, front_indices):
    shoulder_index = front_indices[3]  # 肩膀索引
    hip_index = front_indices[0]  # 髋关节索引
    knee_index = front_indices[1]  # 膝盖索引

    # 计算所有帧的髋关节角度
    hip_angles = []
    for kp in all_keypoints:
        # 获取三个点的坐标
        [shoulder_y, shoulder_x] = kp[shoulder_index][0:-1]
        [hip_y, hip_x] = kp[hip_index][0:-1]
        [knee_y, knee_x] = kp[knee_index][0:-1]
        
        # 计算角度
        angle = calculate_angle((shoulder_y, shoulder_x), (hip_y, hip_x), (knee_y, knee_x))
        hip_angles.append(angle)

    # 取所有髋关节角度的平均值
    if len(hip_angles) > 0:
        hip_angle_avg = np.mean(hip_angles)
    else:
        hip_angle_avg = 0

    return hip_angle_avg
