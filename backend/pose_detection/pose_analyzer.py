# from flask import Flask, request, jsonify
import os
# import tempfile
import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
import cv2
# import kagglehub
import traceback
from scipy.signal import find_peaks
from typing import Union
# from tensorflow_docs.vis import embed
import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from tqdm import tqdm
import logging

from .model import load_model_from_tfhub, get_keypoints_from_video
from .preprocessing import pre_process_video
from .postprocessing import (find_camera_facing_side,
                          get_front_keypoint_indices,
                          get_lowest_pedal_frames,
                          get_highest_pedal_frames,
                          get_hip_knee_ankle_angle,
                          calculate_angle)

# 配置日志格式
logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    force=True  # 强制重新配置日志
)

# 获取logger
logger = logging.getLogger(__name__)

# 定义骨骼连接和颜色映射
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11,
                                     front_indices=None):
    """返回高置信度的关键点和边缘用于可视化。

    参数:
        keypoints_with_scores: 一个形状为[17, 3]的numpy数组，表示MoveNet模型返回的关键点坐标和分数。
        height: 图像高度（像素）。
        width: 图像宽度（像素）。
        keypoint_threshold: 关键点可视化的最小置信度分数。
        front_indices: 需要显示的关键点索引列表。如果为None，则显示所有关键点。

    返回:
        包含以下内容的元组(keypoints_xy, edges_xy, edge_colors):
        * 所有检测到的实体的所有关键点的坐标；
        * 所有检测到的实体的所有骨架边缘的坐标；
        * 边缘应该绘制的颜色。
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    
    # 如果指定了front_indices，只处理这些索引的关键点
    if front_indices is not None:
        valid_indices = front_indices
    else:
        valid_indices = range(len(keypoints_with_scores))
    
    # 只处理指定索引的关键点
    kpts_x = keypoints_with_scores[valid_indices, 1]
    kpts_y = keypoints_with_scores[valid_indices, 0]
    kpts_scores = keypoints_with_scores[valid_indices, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    # 只处理包含指定索引的边
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        # 检查边的两个端点是否都在front_indices中
        if front_indices is not None and (edge_pair[0] not in front_indices or edge_pair[1] not in front_indices):
            continue
            
        if (kpts_scores[valid_indices.index(edge_pair[0]) if front_indices is not None else edge_pair[0]] > keypoint_threshold and
            kpts_scores[valid_indices.index(edge_pair[1]) if front_indices is not None else edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[valid_indices.index(edge_pair[0]) if front_indices is not None else edge_pair[0], 0]
            y_start = kpts_absolute_xy[valid_indices.index(edge_pair[0]) if front_indices is not None else edge_pair[0], 1]
            x_end = kpts_absolute_xy[valid_indices.index(edge_pair[1]) if front_indices is not None else edge_pair[1], 0]
            y_end = kpts_absolute_xy[valid_indices.index(edge_pair[1]) if front_indices is not None else edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)
    
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, len(valid_indices), 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None,
    output_image_height=None, front_indices=None):
    """在图像上绘制关键点预测。

    参数:
        image: 一个形状为[height, width, channel]的numpy数组，表示输入图像的像素值。
        keypoints_with_scores: 一个形状为[17, 3]的numpy数组，表示MoveNet模型返回的关键点坐标和分数。
        crop_region: 一个定义裁剪区域坐标的字典（以归一化坐标表示）。如果提供，此函数还将在图像上绘制边界框。
        output_image_height: 一个表示输出图像高度的整数。注意，图像纵横比将与输入图像相同。
        front_indices: 需要显示的关键点索引列表。如果为None，则显示所有关键点。

    返回:
        一个形状为[out_height, out_width, channel]的numpy数组，表示叠加了关键点预测的图像。
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # 移除大的白色边框
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    # 翻转y轴方向，使其向上为正
    ax.invert_yaxis()

    # 显示原始图像
    ax.imshow(image)

    # 设置关键点和连接线的样式
    line_segments = LineCollection([], linewidths=4, linestyle='solid')
    ax.add_collection(line_segments)
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    # 获取关键点和边的信息
    (keypoint_locs, keypoint_edges,
     edge_colors) = _keypoints_and_edges_for_display(
         keypoints_with_scores, height, width, front_indices=front_indices)

    # 绘制边和关键点
    if keypoint_edges.shape[0]:
        line_segments.set_segments(list(keypoint_edges))
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    # 保存图像
    temp_file = 'temp_plot.png'
    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    
    # 读取保存的图像
    image_from_plot = cv2.imread(temp_file)
    image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2RGB)
    os.remove(temp_file)  # 删除临时文件
    
    # 调整输出图像大小
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
             interpolation=cv2.INTER_CUBIC)
    
    return image_from_plot

def to_gif(images, fps, output_path):
    """将图像序列转换为gif。"""
    imageio.mimsave(output_path, images, fps=fps)
    return output_path

class PoseAnalyzer:
    def __init__(self):
        self.model, self.input_size = load_model_from_tfhub()
        self.all_keypoints = None
        self.lowest_pedal_point_indices = None
        self.highest_pedal_point_indices = None
        self.front_indices = None
        self.knee_angles = None
        self.hip_angles = None
        self.original_aspect_ratio = 1.0  # 默认宽高比为1.0（正方形）


    def upload_video(self, file: str|bytes):
        # 预处理视频并确保tensors被赋值
        frames, tensors = pre_process_video(file)

        # 确保tensors不为空
        if tensors is None:
            return {'error': '视频预处理失败，未生成张量'}
            
        # 获取原始视频的宽高比
        if frames and len(frames) > 0:
            first_frame = frames[0]
            # 注意：预处理后的帧已经是正方形，我们需要从原始视频获取宽高比
            # 这里我们通过检查填充区域来推断原始宽高比
            height, width = first_frame.shape[:2]
            # 检查是否有水平填充（黑色区域）
            horizontal_padding = False
            for i in range(width):
                if np.all(first_frame[height//2, i] == 0):
                    horizontal_padding = True
                    break
            
            if horizontal_padding:
                # 有水平填充，说明原始视频是竖屏
                # 计算非黑色区域的宽度
                non_black_width = 0
                for i in range(width):
                    if not np.all(first_frame[height//2, i] == 0):
                        non_black_width += 1
                self.original_aspect_ratio = non_black_width / height
            else:
                # 有垂直填充，说明原始视频是横屏
                # 计算非黑色区域的高度
                non_black_height = 0
                for i in range(height):
                    if not np.all(first_frame[i, width//2] == 0):
                        non_black_height += 1
                self.original_aspect_ratio = width / non_black_height
                
            logger.info(f"原始视频宽高比: {self.original_aspect_ratio:.3f}")

        self.all_keypoints = get_keypoints_from_video(tensors, self.model, self.input_size)

        # 确保all_keypoints不为空
        if self.all_keypoints is None:
            return {'error': '姿态检测失败，未生成关键点'}

        result = self.get_pose()

        # 确保result不为空
        if result is None:
            return {'error': '姿态分析失败，未生成结果'}

        return result

    def get_pose(self):
        if self.all_keypoints is None:
            raise ValueError('all_keypoints未初始化')
        if self.lowest_pedal_point_indices is None:
            raise ValueError('lowest_pedal_point_indices未初始化')
        if self.highest_pedal_point_indices is None:
            raise ValueError('highest_pedal_point_indices未初始化')
            
        # 获取朝向
        facing_direction = find_camera_facing_side(self.all_keypoints[0])
        self.front_indices = get_front_keypoint_indices(facing_direction)
        hip_knee_ankle_indices = self.front_indices[:3]

        def calculate_joint_angle(p1, p2, p3):
            """计算关节角度，考虑原始视频的宽高比
            p1, p2, p3: [y, x] 坐标，p2是关节点
            返回: 角度（度数）
            """
            # 转换为[x, y]坐标系，并考虑宽高比
            # 对x坐标应用宽高比调整
            v1 = np.array([(p1[1] - p2[1]) * self.original_aspect_ratio, p1[0] - p2[0]])
            v2 = np.array([(p3[1] - p2[1]) * self.original_aspect_ratio, p3[0] - p2[0]])
            
            # 计算向量的点积
            dot_product = np.dot(v1, v2)
            # 计算向量的模
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            # 避免除以0
            if norms == 0:
                return 0
                
            # 计算夹角（弧度）
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            
            # 转换为度数
            angle_deg = np.degrees(angle_rad)
            
            # 根据叉积判断方向
            cross_product = np.cross([v1[0], v1[1], 0], [v2[0], v2[1], 0])[2]
            if cross_product < 0:
                angle_deg = 360 - angle_deg
                
            return angle_deg

        # 计算膝关节角度
        self.knee_angles = []
        logger.info("\n计算膝关节角度:")
        for i, kp in enumerate(self.all_keypoints):
            hip = kp[hip_knee_ankle_indices[0]][:-1]  # 髋关节
            knee = kp[hip_knee_ankle_indices[1]][:-1]  # 膝关节
            ankle = kp[hip_knee_ankle_indices[2]][:-1]  # 踝关节
            
            angle = calculate_joint_angle(hip, knee, ankle)
            # 确保角度在合理范围内
            angle = min(max(angle, 0), 180)
            self.knee_angles.append(angle)
            
            # 每10帧输出一次角度值
            if i % 10 == 0:
                logger.info(f"帧 {i}: 膝关节角度 = {angle:.2f}°")
                logger.info(f"  髋关节坐标: [{hip[0]:.3f}, {hip[1]:.3f}]")
                logger.info(f"  膝关节坐标: [{knee[0]:.3f}, {knee[1]:.3f}]")
                logger.info(f"  踝关节坐标: [{ankle[0]:.3f}, {ankle[1]:.3f}]")
            
        # 计算髋关节角度
        self.hip_angles = []
        logger.info("\n计算髋关节角度:")
        for i, kp in enumerate(self.all_keypoints):
            knee = kp[self.front_indices[1]][:-1]  # 膝关节
            hip = kp[self.front_indices[0]][:-1]   # 髋关节
            shoulder = kp[self.front_indices[3]][:-1]  # 肩关节
            
            angle = calculate_joint_angle(knee, hip, shoulder)
            # 确保角度在合理范围内
            angle = min(max(angle, 0), 180)
            self.hip_angles.append(angle)
            
            # 每10帧输出一次角度值
            if i % 10 == 0:
                logger.info(f"帧 {i}: 髋关节角度 = {angle:.2f}°")
                logger.info(f"  膝关节坐标: [{knee[0]:.3f}, {knee[1]:.3f}]")
                logger.info(f"  髋关节坐标: [{hip[0]:.3f}, {hip[1]:.3f}]")
                logger.info(f"  肩关节坐标: [{shoulder[0]:.3f}, {shoulder[1]:.3f}]")

        # 获取各个角度的平均值
        knee_angle_lowest = self.get_knee_angle_at_lowest_pedal_points_avg()
        knee_angle_highest = self.get_knee_angle_at_highest_pedal_points_avg()
        shoulder_angle = self.get_shoulder_angle_avg()
        elbow_angle = self.get_elbow_angle_avg()
        hip_angle_lowest = self.get_hip_angle_at_lowest_pedal_points_avg()
        hip_angle_highest = self.get_hip_angle_at_highest_pedal_points_avg()

        # 记录角度计算的中间值用于调试
        logger.info("\n角度统计信息:")
        # 只输出前15帧的角度数据
        sample_size = min(15, len(self.knee_angles))
        sample_indices = np.linspace(0, len(self.knee_angles)-1, sample_size, dtype=int)
        logger.info(f"膝关节角度数组(15帧采样): {[f'{self.knee_angles[i]:.1f}°' for i in sample_indices]}")
        logger.info(f"髋关节角度数组(15帧采样): {[f'{self.hip_angles[i]:.1f}°' for i in sample_indices]}")
        logger.info(f"膝关节角度范围: {min(self.knee_angles):.2f}° - {max(self.knee_angles):.2f}°")
        logger.info(f"髋关节角度范围: {min(self.hip_angles):.2f}° - {max(self.hip_angles):.2f}°")
        logger.info(f"最低点膝关节角度: {knee_angle_lowest:.2f}°")
        logger.info(f"最高点膝关节角度: {knee_angle_highest:.2f}°")
        logger.info(f"肩关节角度: {shoulder_angle:.2f}°")
        logger.info(f"肘关节角度: {elbow_angle:.2f}°")

        # 创建测量结果字典
        measurements = {
            'knee_angle_lowest': int(knee_angle_lowest),
            'knee_angle_highest': int(knee_angle_highest),
            'shoulder_angle': int(shoulder_angle),
            'elbow_angle': int(elbow_angle),
            'hip_angle_lowest': int(hip_angle_lowest),
            'hip_angle_highest': int(hip_angle_highest),
        }

        return measurements

    #获取膝盖最大角度的平均数
    def get_knee_angle_at_lowest_pedal_points_avg(self):
        if self.knee_angles is None or len(self.knee_angles) == 0:
            logger.error("膝盖角度数据为空")
            return 0
            
        try:
            # 使用中值滤波平滑数据
            smoothed_angles = np.array(self.knee_angles)
            window_size = 5
            smoothed_angles = np.pad(smoothed_angles, (window_size//2, window_size//2), mode='edge')
            smoothed_angles = np.array([np.median(smoothed_angles[i:i+window_size]) 
                                      for i in range(len(self.knee_angles))])
            
            logger.info("\n计算最低点膝关节角度:")
            # 只输出15帧采样数据
            sample_size = min(15, len(self.knee_angles))
            sample_indices = np.linspace(0, len(self.knee_angles)-1, sample_size, dtype=int)
            logger.info(f"原始角度数组(15帧采样): {[f'{self.knee_angles[i]:.1f}°' for i in sample_indices]}")
            logger.info(f"平滑后角度数组(15帧采样): {[f'{smoothed_angles[i]:.1f}°' for i in sample_indices]}")
            
            # 直接返回平滑后数据的最大值
            max_angle = np.max(smoothed_angles)
            logger.info(f"最大角度: {max_angle:.2f}°")
            return max_angle
            
        except Exception as e:
            logger.error(f"计算最低点膝盖角度时出错: {str(e)}")
            return 0

    #获取膝盖最小角度的平均数
    def get_knee_angle_at_highest_pedal_points_avg(self):
        if self.knee_angles is None or len(self.knee_angles) == 0:
            logger.error("膝盖角度数据为空")
            return 0
            
        try:
            # 使用中值滤波平滑数据
            smoothed_angles = np.array(self.knee_angles)
            window_size = 5
            smoothed_angles = np.pad(smoothed_angles, (window_size//2, window_size//2), mode='edge')
            smoothed_angles = np.array([np.median(smoothed_angles[i:i+window_size]) 
                                      for i in range(len(self.knee_angles))])
            
            logger.info("\n计算最高点膝关节角度:")
            # 只输出15帧采样数据
            sample_size = min(15, len(self.knee_angles))
            sample_indices = np.linspace(0, len(self.knee_angles)-1, sample_size, dtype=int)
            logger.info(f"原始角度数组(15帧采样): {[f'{self.knee_angles[i]:.1f}°' for i in sample_indices]}")
            logger.info(f"平滑后角度数组(15帧采样): {[f'{smoothed_angles[i]:.1f}°' for i in sample_indices]}")
            
            # 直接返回平滑后数据的最小值
            min_angle = np.min(smoothed_angles)
            logger.info(f"最小角度: {min_angle:.2f}°")
            return min_angle
            
        except Exception as e:
            logger.error(f"计算最高点膝盖角度时出错: {str(e)}")
            return 0

    #获取肩膀角度的平均数
    def get_shoulder_angle_avg(self):
        if self.front_indices is None:
            raise ValueError('front_indices未初始化')
        if self.all_keypoints is None:
            raise ValueError('all_keypoints未初始化')
            
        try:
            shoulder_index = self.front_indices[3]  # 肩膀索引
            elbow_index = self.front_indices[4]    # 手肘索引
            hip_index = self.front_indices[0]      # 髋关节索引

            # 计算所有帧的肩膀角度
            shoulder_angles = []
            for kp in self.all_keypoints:
                # 获取三个点的坐标
                elbow = kp[elbow_index][:-1]    # [y, x]
                shoulder = kp[shoulder_index][:-1]  # [y, x]
                hip = kp[hip_index][:-1]        # [y, x]
                
                # 计算角度：手肘-肩膀-髋部的夹角，考虑宽高比
                v1 = np.array([(elbow[1] - shoulder[1]) * self.original_aspect_ratio, elbow[0] - shoulder[0]])
                v2 = np.array([(hip[1] - shoulder[1]) * self.original_aspect_ratio, hip[0] - shoulder[0]])
                
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms == 0:
                    continue
                    
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                # 确保角度在合理范围内
                angle = min(max(angle, 0), 180)
                shoulder_angles.append(angle)

            # 移除异常值
            if shoulder_angles:
                mean_angle = np.mean(shoulder_angles)
                std_angle = np.std(shoulder_angles)
                valid_angles = [
                    angle for angle in shoulder_angles
                    if abs(angle - mean_angle) <= 2 * std_angle
                ]
                
                if valid_angles:
                    return np.mean(valid_angles)
                    
            return 0
            
        except Exception as e:
            logger.error(f"计算肩膀角度时出错: {str(e)}")
            return 0

    #获取手肘角度的平均数
    def get_elbow_angle_avg(self):
        if self.front_indices is None:
            raise ValueError('front_indices未初始化')
        if self.all_keypoints is None:
            raise ValueError('all_keypoints未初始化')
            
        try:
            shoulder_index = self.front_indices[3]  # 肩膀索引
            elbow_index = self.front_indices[4]    # 手肘索引
            wrist_index = self.front_indices[5]    # 手腕索引

            # 计算所有帧的手肘角度
            elbow_angles = []
            for kp in self.all_keypoints:
                # 获取三个点的坐标
                shoulder = kp[shoulder_index][:-1]  # [y, x]
                elbow = kp[elbow_index][:-1]     # [y, x]
                wrist = kp[wrist_index][:-1]     # [y, x]
                
                # 计算角度，考虑宽高比
                v1 = np.array([(shoulder[1] - elbow[1]) * self.original_aspect_ratio, shoulder[0] - elbow[0]])
                v2 = np.array([(wrist[1] - elbow[1]) * self.original_aspect_ratio, wrist[0] - elbow[0]])
                
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms == 0:
                    continue
                    
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                # 确保角度在合理范围内
                angle = min(max(angle, 0), 180)
                elbow_angles.append(angle)

            # 移除异常值
            if elbow_angles:
                mean_angle = np.mean(elbow_angles)
                std_angle = np.std(elbow_angles)
                valid_angles = [
                    angle for angle in elbow_angles
                    if abs(angle - mean_angle) <= 2 * std_angle
                ]
                
                if valid_angles:
                    return np.mean(valid_angles)
                    
            return 0
            
        except Exception as e:
            logger.error(f"计算手肘角度时出错: {str(e)}")
            return 0

    #获取髋关节最低点角度的平均数
    def get_hip_angle_at_lowest_pedal_points_avg(self):
        if self.hip_angles is None or len(self.hip_angles) == 0:
            logger.error("髋关节角度数据为空")
            return 0
            
        try:
            # 使用中值滤波平滑数据
            smoothed_angles = np.array(self.hip_angles)
            window_size = 5
            smoothed_angles = np.pad(smoothed_angles, (window_size//2, window_size//2), mode='edge')
            smoothed_angles = np.array([np.median(smoothed_angles[i:i+window_size]) 
                                      for i in range(len(self.hip_angles))])
            
            logger.info("\n计算最低点髋关节角度:")
            # 只输出15帧采样数据
            sample_size = min(15, len(self.hip_angles))
            sample_indices = np.linspace(0, len(self.hip_angles)-1, sample_size, dtype=int)
            logger.info(f"原始角度数组(15帧采样): {[f'{self.hip_angles[i]:.1f}°' for i in sample_indices]}")
            logger.info(f"平滑后角度数组(15帧采样): {[f'{smoothed_angles[i]:.1f}°' for i in sample_indices]}")
            
            # 直接返回平滑后数据的最大值
            max_angle = np.max(smoothed_angles)
            logger.info(f"最大角度: {max_angle:.2f}°")
            return max_angle
            
        except Exception as e:
            logger.error(f"计算髋关节最低点角度时出错: {str(e)}")
            return 0

    #获取髋关节最高点角度的平均数
    def get_hip_angle_at_highest_pedal_points_avg(self):
        if self.hip_angles is None or len(self.hip_angles) == 0:
            logger.error("髋关节角度数据为空")
            return 0
            
        try:
            # 使用中值滤波平滑数据
            smoothed_angles = np.array(self.hip_angles)
            window_size = 5
            smoothed_angles = np.pad(smoothed_angles, (window_size//2, window_size//2), mode='edge')
            smoothed_angles = np.array([np.median(smoothed_angles[i:i+window_size]) 
                                      for i in range(len(self.hip_angles))])
            
            logger.info("\n计算最高点髋关节角度:")
            # 只输出15帧采样数据
            sample_size = min(15, len(self.hip_angles))
            sample_indices = np.linspace(0, len(self.hip_angles)-1, sample_size, dtype=int)
            logger.info(f"原始角度数组(15帧采样): {[f'{self.hip_angles[i]:.1f}°' for i in sample_indices]}")
            logger.info(f"平滑后角度数组(15帧采样): {[f'{smoothed_angles[i]:.1f}°' for i in sample_indices]}")
            
            # 直接返回平滑后数据的最小值
            min_angle = np.min(smoothed_angles)
            logger.info(f"最小角度: {min_angle:.2f}°")
            return min_angle
            
        except Exception as e:
            logger.error(f"计算髋关节最高点角度时出错: {str(e)}")
            return 0

    @property
    def test_pose_analyzer(self):
        """
        测试函数，用于验证姿态分析器的各项功能
        """
        # 测试初始化
        logger.info("1. 测试模型初始化...")
        try:
            logger.info("✓ 模型初始化成功")
        except Exception as e:
            logger.error(f"✗ 模型初始化失败: {str(e)}")

        # 测试视频预处理
        logger.info("\n2. 测试视频预处理...")
        try:
            # 使用当前目录下的测试视频
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_video_path = os.path.join(current_dir, "uploads", "raw.mp4")
            
            if not os.path.exists(test_video_path):
                logger.error(f"✗ 测试视频不存在: {test_video_path}")
                return
            
            frames, tensors = pre_process_video(test_video_path)
            
            # 验证预处理结果
            logger.info(f"  - 处理的视频帧数: {len(frames)}")
            logger.info(f"  - 张量形状: {tensors.shape}")
            logger.info(f"  - 张量数据类型: {tensors.dtype}")
            logger.info(f"  - 张量值范围: [{tf.reduce_min(tensors):.2f}, {tf.reduce_max(tensors):.2f}]")
            logger.info("✓ 视频预处理成功")
        except Exception as e:
            logger.error(f"✗ 视频预处理失败: {str(e)}")
            return None

        # 测试姿态检测
        logger.info("\n3. 测试姿态检测...")
        try:
            self.all_keypoints = get_keypoints_from_video(tensors, self.model, self.input_size)
            logger.info(f"  - 关键点数量: {len(self.all_keypoints)}")
            logger.info(f"  - 单帧关键点形状: {self.all_keypoints[0].shape}")
            logger.info("✓ 姿态检测成功")
        except Exception as e:
            logger.error(f"✗ 姿态检测失败: {str(e)}")
            return None

        # 输出预测结果gif图


        # 测试姿态分析
        logger.info("\n4. 测试姿态分析...")
        try:
            # 获取朝向和关键点索引
            facing_direction = find_camera_facing_side(self.all_keypoints[0])
            self.front_indices = get_front_keypoint_indices(facing_direction)
            logger.info(f"  - 检测到的朝向: {facing_direction}")
            logger.info(f"  - 关键点索引: {self.front_indices}")
            
            # 测试膝盖角度计算
            logger.info("\n  测试膝盖角度计算:")
            # 获取一些示例帧的膝盖角度
            hip_knee_ankle_indices = self.front_indices[:4]
            sample_angles = [get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices) for kp in self.all_keypoints[:5]]
            logger.info(f"  - 前5帧的膝盖角度: {[f'{angle:.2f}°' for angle in sample_angles]}")
            
            # 测试最低点检测
            self.lowest_pedal_point_indices = get_lowest_pedal_frames(self.all_keypoints, hip_knee_ankle_indices)
            logger.info(f"  - 检测到的最低点帧索引: {self.lowest_pedal_point_indices[:5]}")
            
            # 测试最高点检测
            self.highest_pedal_point_indices = get_highest_pedal_frames(self.all_keypoints, hip_knee_ankle_indices)
            logger.info(f"  - 检测到的最高点帧索引: {self.highest_pedal_point_indices[:5]}")
            
            # 获取完整结果
            result = self.get_pose()
            logger.info("\n姿态分析结果:")
            logger.info(f"  - 最低点膝盖角度: {result['knee_angle_lowest']:.2f}°")
            logger.info(f"  - 最高点膝盖角度: {result['knee_angle_highest']:.2f}°")
            logger.info(f"  - 肩膀角度: {result['shoulder_angle']:.2f}°")
            logger.info(f"  - 手肘角度: {result['elbow_angle']:.2f}°")
            logger.info(f"  - 最低点髋关节角度: {result['hip_angle_lowest']:.2f}°")
            logger.info(f"  - 最高点髋关节角度: {result['hip_angle_highest']:.2f}°")
            logger.info("✓ 姿态分析成功")

            # 创建保存结果的文件夹
            output_dir = os.path.join(current_dir, 'output_frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 提取并绘制最低点和最高点的帧
            concat_indices = np.concatenate([self.lowest_pedal_point_indices, self.highest_pedal_point_indices])
            
            # 创建一个列表来存储所有绘制的帧
            output_images = []
            
            # 处理所有帧并生成GIF
            logger.info("\n5. 生成姿态检测可视化...")
            try:
                # 首先处理关键帧（最低点和最高点）
                for idx in concat_indices:
                    if idx < len(frames):  # 确保索引在有效范围内
                        frame = frames[idx]  # 使用切片好的frames变量
                        # 在帧上绘制关键点和骨架
                        keypoints = self.all_keypoints[idx]
                        result_frame = draw_prediction_on_image(
                            frame, 
                            keypoints, 
                            crop_region=None,
                            output_image_height=300,
                            front_indices=self.front_indices
                        )
                        # 保存结果帧
                        output_path = os.path.join(output_dir, f'frame_{idx}.png')
                        cv2.imwrite(output_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
                        if idx % 10 == 0:  # 每10帧显示一次进度
                            logger.info(f'✓ 保存关键帧: {output_path}')
                            logger.info(f"绘制关键帧索引: {idx}")
                        # 添加到输出图像列表
                        output_images.append(result_frame)
                
                # 然后处理所有帧以生成完整的GIF（可选，取决于性能需求）
                logger.info("\n生成完整GIF动画...")
                all_output_images = []
                # 为了性能考虑，可以选择每隔几帧处理一次
                step = max(1, len(frames) // 50)  # 最多处理50帧
                for idx in tqdm(range(0, len(frames), step), desc="处理帧"):
                    if idx % 10 == 0:  # 每10帧显示一次进度
                        logger.info(f"处理帧 {idx}/{len(frames)}")
                    
                    frame = frames[idx]
                    keypoints = self.all_keypoints[idx]
                    result_frame = draw_prediction_on_image(
                        frame, 
                        keypoints, 
                        crop_region=None,
                        output_image_height=300,
                        front_indices=self.front_indices
                    )
                    all_output_images.append(result_frame)
                
                # 保存GIF
                gif_path = os.path.join(output_dir, 'pose_animation.gif')
                to_gif(all_output_images, fps=10, output_path=gif_path)
                logger.info(f"✓ 成功保存GIF动画: {gif_path}")
                
            except Exception as e:
                logger.error(f"✗ 生成可视化失败: {str(e)}")
                logger.error(traceback.format_exc())

            return result
        except Exception as e:
            logger.error(f"✗ 姿态分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    
# TODO 需要增加帧返回值
    def pose_analyzer(self, file: Union[str, bytes]) -> tuple[dict, list[np.ndarray], str]:
        """
        姿态分析器的主函数，接收文件路径或字节数据，处理后返回姿态分析结果。

        参数:
            file (str | bytes): 文件路径或字节数据。

        返回:
            tuple: (dict: 包含姿态分析结果的字典, list: 结果帧列表, str: GIF文件路径)
        """
        # 初始化模型
        logger.info("initializing model...")
        try:
            logger.info("model initialization successful")
        except Exception as e:
            logger.error(f"model initialization failed: {str(e)}")
            return {"error": f"模型初始化失败: {str(e)}"}, [], ""
        
        # 预处理视频
        logger.info("preprocessing video...")
        try:
            frames, tensors = pre_process_video(file)
            logger.info(f"frames processed: {len(frames)}")
            logger.info(f"tensor shape: {tensors.shape}")
            logger.info(f"tensor dtype: {tensors.dtype}")
            logger.info(f"tensor range: [{tf.reduce_min(tensors):.2f}, {tf.reduce_max(tensors):.2f}]")
            
            # 获取原始视频的宽高比
            self.original_aspect_ratio = 1.0  # 默认宽高比为1.0（正方形）
            if frames and len(frames) > 0:
                first_frame = frames[0]
                height, width = first_frame.shape[:2]
                
                # 检查是否有水平填充（黑色区域）
                horizontal_padding = False
                for i in range(width):
                    if np.all(first_frame[height//2, i] == 0):
                        horizontal_padding = True
                        break
                
                if horizontal_padding:
                    # 有水平填充，说明原始视频是竖屏
                    # 计算非黑色区域的宽度
                    non_black_width = 0
                    for i in range(width):
                        if not np.all(first_frame[height//2, i] == 0):
                            non_black_width += 1
                    if non_black_width > 0:
                        self.original_aspect_ratio = non_black_width / height
                else:
                    # 有垂直填充，说明原始视频是横屏
                    # 计算非黑色区域的高度
                    non_black_height = 0
                    for i in range(height):
                        if not np.all(first_frame[i, width//2] == 0):
                            non_black_height += 1
                    if non_black_height > 0:
                        self.original_aspect_ratio = width / non_black_height
                
                logger.info(f"原始视频宽高比: {self.original_aspect_ratio:.3f}")
            
            logger.info("video preprocessing successful")
        except Exception as e:
            logger.error(f"video preprocessing failed: {str(e)}")
            return {"error": f"视频预处理失败: {str(e)}"}, [], ""

        # 姿态检测
        logger.info("detecting pose...")
        try:
            self.all_keypoints = get_keypoints_from_video(tensors, self.model, self.input_size)
            logger.info(f"keypoints count: {len(self.all_keypoints)}")
            logger.info(f"single frame keypoints shape: {self.all_keypoints[0].shape}")
            logger.info("pose detection successful")
        except Exception as e:
            logger.error(f"pose detection failed: {str(e)}")
            return {"error": f"姿态检测失败: {str(e)}"}, [], ""

        # 姿态分析
        logger.info("analyzing pose...")
        try:
            # 获取朝向和关键点索引
            facing_direction = find_camera_facing_side(self.all_keypoints[0])
            self.front_indices = get_front_keypoint_indices(facing_direction)
            logger.info(f"detected facing direction: {facing_direction}")
            logger.info(f"keypoint indices: {self.front_indices}")

            # 检测最低点
            self.lowest_pedal_point_indices = get_lowest_pedal_frames(self.all_keypoints, self.front_indices)
            logger.info(f"detected lowest pedal frame indices: {self.lowest_pedal_point_indices[:5]}")
            
            # 检测最高点
            self.highest_pedal_point_indices = get_highest_pedal_frames(self.all_keypoints, self.front_indices)
            logger.info(f"detected highest pedal frame indices: {self.highest_pedal_point_indices[:5]}")

            # 获取完整结果
            result = self.get_pose()
            logger.info("pose analysis results:")
            logger.info(f"lowest point knee angle: {result['knee_angle_lowest']:.2f}°")
            logger.info(f"highest point knee angle: {result['knee_angle_highest']:.2f}°")
            logger.info(f"shoulder angle: {result['shoulder_angle']:.2f}°")
            logger.info(f"elbow angle: {result['elbow_angle']:.2f}°")
            logger.info(f"lowest point hip angle: {result['hip_angle_lowest']:.2f}°")
            logger.info(f"highest point hip angle: {result['hip_angle_highest']:.2f}°")
            logger.info("pose analysis successful")

            # 提取并绘制最低点和最高点的帧
            concat_indices = np.concatenate([self.lowest_pedal_point_indices, self.highest_pedal_point_indices])
            result_frames = []
            
            # 创建保存结果的文件夹
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(current_dir, 'output_frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 处理关键帧并生成GIF
            logger.info("generating pose detection visualization...")
            gif_path = ""
            try:
                # 处理关键帧（最低点和最高点）
                output_images = []
                for idx in tqdm(concat_indices, desc="processing key frames"):
                    if idx < len(frames):
                        frame = frames[idx]
                        keyPoints = self.all_keypoints[idx]
                        result_frame = draw_prediction_on_image(
                            frame, 
                            keyPoints,
                            crop_region=None,
                            output_image_height=300,
                            front_indices=self.front_indices
                        )
                        output_path = os.path.join(output_dir, f'frame_{idx}.png')
                        cv2.imwrite(output_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
                        
                        output_images.append(result_frame)
                        result_frames.append(result_frame)
                    else:
                        logger.warning(f"index out of range: {idx}")

                logger.info(f"output directory: {output_dir}")
                
                # 生成完整的GIF动画
                logger.info("generating complete GIF animation...")
                all_output_images = []
                step = max(1, len(frames) // 50)
                for idx in tqdm(range(0, len(frames), step), desc='processing frames'):
                    frame = frames[idx]
                    keyPoints = self.all_keypoints[idx]
                    result_frame = draw_prediction_on_image(
                        frame, 
                        keyPoints,
                        crop_region=None,
                        output_image_height=300,
                        front_indices=self.front_indices
                    )
                    all_output_images.append(result_frame)
                
                # 保存GIF
                gif_path = os.path.join(output_dir, 'pose_animation.gif')
                to_gif(all_output_images, fps=10, output_path=gif_path)
                logger.info(f"GIF animation saved successfully: {gif_path}")
                
            except Exception as e:
                logger.error(f"visualization generation failed: {str(e)}")
                logger.error(traceback.format_exc())

            return result, result_frames, gif_path
        except Exception as e:
            logger.error(f"pose analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"姿态分析失败: {str(e)}"}, [], ""

if __name__ == "__main__":
    analyzer = PoseAnalyzer()

