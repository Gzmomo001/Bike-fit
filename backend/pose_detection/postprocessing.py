"""""" """""" """""" """""" """
 POSTPROCESSING FUNCTIONS 所有的角度和距离的计算都在该文件中
""" """""" """""" """""" """"""

import math
import numpy as np
from scipy.signal import find_peaks
from pose_detection.keypoints import KEYPOINT_DICT

#分析视频的拍摄朝向
def find_camera_facing_side(keypoints):
    """Returns whether the cyclist is facing the camera with his left or right side
    Args:
        nosex: float representing the x coordinate of the nose (x=0 if point is on the left border of the image)
        hipx: float representing the x coordinate of the hip (either left or right)
    Returns:
        'left' if the left leg is facing the camera, 'right' otherwise
    """
    hipx = keypoints[KEYPOINT_DICT["left_hip"]][1] #比较鼻子和髋部的x坐标来确认朝向
    nosex = keypoints[KEYPOINT_DICT["nose"]][1]
    return "left" if nosex < hipx else "right"

#得到拍摄调度侧躯干的数据
def get_front_keypoint_indices(facing_dir):
    hip_index = KEYPOINT_DICT[f"{facing_dir}_hip"]
    knee_index = KEYPOINT_DICT[f"{facing_dir}_knee"]
    ankle_index = KEYPOINT_DICT[f"{facing_dir}_ankle"]
    shoulder_index = KEYPOINT_DICT[f"{facing_dir}_shoulder"]
    elbow_index = KEYPOINT_DICT[f"{facing_dir}_elbow"]
    wrist_index = KEYPOINT_DICT[f"{facing_dir}_wrist"]
    return hip_index, knee_index, ankle_index, shoulder_index, elbow_index, wrist_index

#分析每帧中脚踝点的y坐标，找出踏板位置最高的帧
def get_highest_pedal_frames(all_keypoints, hip_knee_ankle_indices):
    ankle_index = hip_knee_ankle_indices[2]  # 获取脚踝位置的索引
    ankle_y_values = []
    # Iterate through all frames to collect the y-coordinates of the ankle points
    for frame_idx in range(len(all_keypoints)):  # 遍历所有帧，收集所有脚踝y坐标
        ankle_y_values.append(all_keypoints[frame_idx][ankle_index][0])#找到最高点
    # the distance variable lets you to easily pick only the highest peak values and ignore local jitters in a pedal rotation
    peak_indices = (
        find_peaks(ankle_y_values, distance=10)
    )[0]  # 找到最高点的坐标索引
    # distance=10 参数确保了相邻的峰值之间至少有 20 个数据点的距离，从而过滤掉局部抖动，只保留明显的峰值。
    return peak_indices

#分析每帧中脚踝点的y坐标，找出踏板位置最低的帧
def get_lowest_pedal_frames(all_keypoints, hip_knee_ankle_indices):
    """
    Find the frames with the lowest pedal position.

    This function identifies the frames where the pedal position is the lowest by analyzing the y-coordinates of the ankle points
    across all frames. It helps to filter out the frames with the highest pedal positions, which is useful for analyzing the
    pedaling motion in cycling activities.

    Parameters:
    all_keypoints (list of list of tuple): A list containing keypoint information for each frame, each keypoint is represented
                                          as a tuple, and each frame's keypoints are stored in a list.
    hipkneeankleindices (list): A list containing the indices of the hip, knee, and ankle keypoints in the keypoints list.

    Returns:
    list: A list of indices representing the frames with the lowest pedal positions.
    """
    # Get the ankle index to track changes in the y-coordinate of the ankle point
    ankle_index = hip_knee_ankle_indices[2]#获取脚踝位置的索引
    ankle_y_values = []
    # Iterate through all frames to collect the y-coordinates of the ankle points
    for frame_idx in range(len(all_keypoints)):#遍历所有帧，收集所有脚踝y坐标
        ankle_y_values.append(-1 * all_keypoints[frame_idx][ankle_index][0])#将所有元素取反
    # the distance variable lets you to easily pick only the highest peak values and ignore local jitters in a pedal rotation
    peak_indices = find_peaks(ankle_y_values, distance=10)[0]#找到所有的峰值。
    # distance=10 参数确保了相邻的峰值之间至少有 10 个数据点的距离，从而过滤掉局部抖动，只保留明显的峰值。
    return peak_indices

#获取膝盖的角度
def get_hip_knee_ankle_angle(keypoint, indices):#indices中分别对应髋，膝盖和脚踝的坐标
    [hip_y, hip_x] = keypoint[indices[0]][0:-1]
    [knee_y, knee_x] = keypoint[indices[1]][0:-1]
    [ankle_y, ankle_x] = keypoint[indices[2]][0:-1]
    angle = calculate_angle((hip_y, hip_x), (knee_y, knee_x), (ankle_y, ankle_x))
    return angle

#根据坐标计算角度
def calculate_angle(coords1, coordsmid, coords2):
    [coords1_x, coords1_y] = coords1
    [coordsmid_x, coordsmid_y] = coordsmid
    [coords2_x, coords2_y] = coords2
    # 计算向量
    vector1 = (coords1_x - coordsmid_x, coords1_y - coordsmid_y)  # 从肩膀到肘部
    vector2 = (coords2_x - coordsmid_x, coords1_y - coordsmid_x)  # 从肩膀到髋部

    # 计算向量的模长
    length_shoulder_elbow = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    length_shoulder_hip = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # 检查分母是否为零
    if length_shoulder_elbow == 0 or length_shoulder_hip == 0:
        raise ValueError("Vectors cannot have zero length.")

    # 计算点积
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # 计算余弦值并处理浮点数精度问题
    cos_theta = dot_product / (length_shoulder_elbow * length_shoulder_hip)
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # 计算角度
    angle = math.degrees(math.acos(cos_theta))

    return angle

#筛选不合理的膝盖角度
def filter_bad_knee_angles(angles, indices, m=2.0):
    """Filters out outliers from the passed list.
    Args:
        angles: 需要筛选的角度
        indices: 需要筛选的帧
        m: the maximum distance
    """
    indices = np.array(indices)
    angles = np.array(angles)
    mask = (90 < angles) & (angles < 170)
    angles = angles[mask]
    indices = indices[mask]
    # calc dist to median (median is more robust to outliers than mean)
    dist = np.abs(angles - np.median(angles))
    # get median of distances
    mdev = np.median(dist)
    # scale the distances based on median of distances
    s = dist / mdev if mdev else 0.0
    mask = s < m
    return angles[mask], indices[mask]

#根据膝盖张角提出建议，此函数已不需要
"""
def make_recommendation(inner_knee_angle, ideal_angle=145, buffer=5):
    
    if inner_knee_angle < ideal_angle - buffer:
        return "UP"
    elif inner_knee_angle > ideal_angle + buffer:
        return "DOWN"
    return "NOOP"
"""

