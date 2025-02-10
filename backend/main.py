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
                            get_highest_pedal_frames)

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

    (
        facing_direction,
        hip_knee_ankle_indices,
        all_angles,
        lowest_pedal_point_indices,
        angles_at_lowest_pedal_points,
        angle_at_lowest_pedal_points_avg,
        angle_at_lowest_pedal_points_std,
    ) = knee_pose_process_video(all_keypoints)

    result = {
        'facing_direction': facing_direction,
        'hip_knee_ankle_indices': hip_knee_ankle_indices,
        'all_angles': all_angles,
        'lowest_pedal_point_indices': lowest_pedal_point_indices,
        'angles_at_lowest_pedal_points': angles_at_lowest_pedal_points,
        'angle_at_lowest_pedal_points_avg': angle_at_lowest_pedal_points_avg,
        'angle_at_lowest_pedal_points_std': angle_at_lowest_pedal_points_std
    }

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


    hip_knee_ankle_indices = get_front_keypoint_indices(facing_direction)[:4]#存储拍摄视角侧的髋部，膝盖和脚踝的位置索引
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]
    #找到踏板角度最低的帧
    lowest_pedal_point_indices = get_lowest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )
    #从所有踏板角度中找到最位置最低的帧,并返回这些帧的膝盖角度
    angles_at_lowest_pedal_points = [
        knee_angles[i][1] for i in lowest_pedal_point_indices
    ]
    #排除不正常的角度和帧，进一步增加稳定性
    angles_at_lowest_pedal_points, lowest_pedal_point_indices = filter_bad_knee_angles(
        angles_at_lowest_pedal_points, lowest_pedal_point_indices
    )
    #取最小膝盖角度的平均数和方差
    angle_at_lowest_pedal_points_avg, angle_at_lowest_pedal_points_std = np.mean(
        angles_at_lowest_pedal_points
    ), np.std(angles_at_lowest_pedal_points)

    return (
        facing_direction,
        hip_knee_ankle_indices,
        knee_angles,
        lowest_pedal_point_indices,
        angles_at_lowest_pedal_points,
        angle_at_lowest_pedal_points_avg,
        angle_at_lowest_pedal_points_std,
    )
#获取髋，膝盖，肩膀，手肘的角度
def get_pose(all_keypoints):
    facing_direction = find_camera_facing_side(all_keypoints[0])
    front_indices = get_front_keypoint_indices(facing_direction)
    angles = []#找到所有帧的角度

    # 获取膝盖最大角度的平均数
    knee_angle_at_lowest_pedal_points_avg = get_knee_angle_at_lowest_pedal_points_avg(
        all_keypoints, front_indices
    )



#获取膝盖最大角度的平均数
def get_knee_angle_at_lowest_pedal_points_avg(all_keypoints, front_indices):
    hip_knee_ankle_indices = front_indices[:4] # 存储拍摄视角侧的髋部，膝盖和脚踝的位置索引

    #存储所有膝盖角度
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 找到踏板角度最低的帧
    lowest_pedal_point_indices = get_lowest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    #找到踏板角度最低帧的踏板角度
    angles_at_lowest_pedal_points = [
        knee_angles[i][1] for i in lowest_pedal_point_indices
    ]

    # 排除不正常的角度和帧，进一步增加稳定性
    angles_at_lowest_pedal_points, lowest_pedal_point_indices = filter_bad_knee_angles(
        angles_at_lowest_pedal_points, lowest_pedal_point_indices
    )

    # 取最小膝盖角度的平均数和方差
    angle_at_lowest_pedal_points_avg = np.mean(angles_at_lowest_pedal_points)

    return angle_at_lowest_pedal_points_avg

#获取膝盖最小角度的平均数

def get_knee_angle_at_highest_pedal_points_avg(all_keypoints, front_indices):
    hip_knee_ankle_indices = front_indices[:4] # 存储拍摄视角侧的髋部，膝盖和脚踝的位置索引

    #存储所有膝盖角度
    knee_angles = [
        get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
        for kp in all_keypoints
    ]

    # 找到踏板角度最低的帧
    highest_pedal_point_indices = get_highest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    #找到踏板角度最低帧的踏板角度
    angles_at_highest_pedal_points = [
        knee_angles[i][1] for i in highest_pedal_point_indices
    ]

    # 取最小膝盖角度的平均数和方差
    angle_at_lowest_pedal_points_avg = np.mean(angles_at_highest_pedal_points)

    return angle_at_lowest_pedal_points_avg

#获取肩膀角度的平均数
def get_shoulder_angle_avg(all_keypoints, front_indices){

}

#获取手肘角度的平均数
def get_elbow_angle_avg(all_keypoints, front_indices){

}