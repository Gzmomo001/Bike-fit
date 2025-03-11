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

from .model import load_model_from_tfhub, get_keypoints_from_video
from .preprocessing import pre_process_video
from .postprocessing import (find_camera_facing_side,
                          get_front_keypoint_indices,
                          get_lowest_pedal_frames,
                          get_highest_pedal_frames,
                          get_hip_knee_ankle_angle,
                          calculate_angle)

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


    def upload_video(self, file: str|bytes):
        # 预处理视频并确保tensors被赋值
        tensors = pre_process_video(file)

        # 确保tensors不为空
        if tensors is None:
            return {'error': '视频预处理失败，未生成张量'}

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
        facing_direction = find_camera_facing_side(self.all_keypoints[0])
        self.front_indices = get_front_keypoint_indices(facing_direction)
        hip_knee_ankle_indices = self.front_indices[:3]

        self.knee_angles = [
                    get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices)
                    for kp in self.all_keypoints
                ]
        self.hip_angles = [
                    calculate_angle(kp[self.front_indices[1]][0:-1], kp[self.front_indices[0]][0:-1], kp[self.front_indices[3]][0:-1])
                    for kp in self.all_keypoints
                ]

        # 获取膝盖最低点角度的平均数
        knee_angle_lowest = self.get_knee_angle_at_lowest_pedal_points_avg(
        )
        
        # 获取膝盖最高点角度的平均数
        knee_angle_highest = self.get_knee_angle_at_highest_pedal_points_avg(
        )
        
        # 获取肩膀角度的平均数
        shoulder_angle = self.get_shoulder_angle_avg()
        
        # 获取手肘角度的平均数
        elbow_angle = self.get_elbow_angle_avg()

        # 获取髋关节最低点和最高点的角度
        hip_angle_lowest = self.get_hip_angle_at_lowest_pedal_points_avg()
        hip_angle_highest = self.get_hip_angle_at_highest_pedal_points_avg()

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
        if self.lowest_pedal_point_indices is None:
            raise ValueError('lowest_pedal_point_indices未初始化')
        # 获取所有帧的膝盖角度

        largest_knee_angle = find_peaks(self.knee_angles, distance=5)[0]

        # 从所有踏板角度中找到最位置最低的帧的膝盖角度
        angles_at_lowest_pedal_points = [
            self.knee_angles[i] for i in largest_knee_angle
        ]

        # 取最小膝盖角度的平均数
        angle_avg = np.mean(angles_at_lowest_pedal_points)

        return angle_avg

    #获取膝盖最小角度的平均数
    def get_knee_angle_at_highest_pedal_points_avg(self):
        if self.highest_pedal_point_indices is None:
            raise ValueError('highest_pedal_point_indices未初始化')
        # 获取所有帧的膝盖角度

        least_knee_angle = find_peaks(-np.array(self.knee_angles), distance=5)[0]

        # 从所有踏板角度中找到最位置最高的帧的膝盖角度
        angles_at_highest_pedal_points = [
            self.knee_angles[i] for i in least_knee_angle
        ]

        # 取最小膝盖角度的平均数和方差
        angle_avg = np.mean(angles_at_highest_pedal_points)

        return angle_avg

    #获取肩膀角度的平均数
    def get_shoulder_angle_avg(self):
        if self.front_indices is None:
            raise ValueError('front_indices未初始化')
        if self.all_keypoints is None:
            raise ValueError('all_keypoints未初始化')
        shoulder_index = self.front_indices[3]  # 肩膀索引
        elbow_index = self.front_indices[4]  # 手肘索引
        hip_index = self.front_indices[0]  # 髋关节索引

        # 计算所有帧的肩膀角度
        shoulder_angles = []
        for kp in self.all_keypoints:
            # 获取三个点的坐标
            [shoulder_y, shoulder_x] = kp[shoulder_index][0:-1]
            [elbow_y, elbow_x] = kp[elbow_index][0:-1]
            [hip_y, hip_x] = kp[hip_index][0:-1]
            
            # 计算角度：手肘-肩膀-髋部的夹角
            angle = calculate_angle((elbow_y, elbow_x), (shoulder_y, shoulder_x), (hip_y, hip_x))
            shoulder_angles.append(angle)

        # 取所有肩膀角度的平均值
        if len(shoulder_angles) > 0:
            shoulder_angle_avg = np.mean(shoulder_angles)
        else:
            shoulder_angle_avg = 0

        return shoulder_angle_avg

    #获取手肘角度的平均数
    def get_elbow_angle_avg(self):
        if self.front_indices is None:
            raise ValueError('front_indices未初始化')
        if self.all_keypoints is None:
            raise ValueError('all_keypoints未初始化')
        shoulder_index = self.front_indices[3]  # 肩膀索引
        elbow_index = self.front_indices[4]  # 手肘索引
        wrist_index = self.front_indices[5]  # 手腕索引

        # 计算所有帧的手肘角度
        elbow_angles = []
        for kp in self.all_keypoints:
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

    #获取髋关节最低点角度的平均数
    def get_hip_angle_at_lowest_pedal_points_avg(self):

        
        largest_hip_indices = find_peaks(self.hip_angles)[0]

        largest_hip_angle = [self.hip_angles[i] for i in largest_hip_indices]

        hip_angle_avg = np.mean(largest_hip_angle)

        return hip_angle_avg

    #获取髋关节最高点角度的平均数
    def get_hip_angle_at_highest_pedal_points_avg(self):

        least_hip_indices = find_peaks(-np.array(self.hip_angles))[0]

        least_hip_angle = [self.hip_angles[i] for i in least_hip_indices]

        hip_angle_avg = np.mean(least_hip_angle)

        return hip_angle_avg

    @property
    def test_pose_analyzer(self):
        """
        测试函数，用于验证姿态分析器的各项功能
        """
        # 测试初始化
        print("1. 测试模型初始化...")
        try:
            print("✓ 模型初始化成功")
        except Exception as e:
            print(f"✗ 模型初始化失败: {str(e)}")

        # 测试视频预处理
        print("\n2. 测试视频预处理...")
        try:
            # 使用当前目录下的测试视频
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_video_path = os.path.join(current_dir, "uploads", "raw.mp4")
            
            if not os.path.exists(test_video_path):
                print(f"✗ 测试视频不存在: {test_video_path}")
                return
            
            frames, tensors = pre_process_video(test_video_path)
            
            # 验证预处理结果
            print(f"  - 处理的视频帧数: {len(frames)}")
            print(f"  - 张量形状: {tensors.shape}")
            print(f"  - 张量数据类型: {tensors.dtype}")
            print(f"  - 张量值范围: [{tf.reduce_min(tensors):.2f}, {tf.reduce_max(tensors):.2f}]")
            print("✓ 视频预处理成功")
        except Exception as e:
            print(f"✗ 视频预处理失败: {str(e)}")
            return None

        # 测试姿态检测
        print("\n3. 测试姿态检测...")
        try:
            self.all_keypoints = get_keypoints_from_video(tensors, self.model, self.input_size)
            print(f"  - 关键点数量: {len(self.all_keypoints)}")
            print(f"  - 单帧关键点形状: {self.all_keypoints[0].shape}")
            print("✓ 姿态检测成功")
        except Exception as e:
            print(f"✗ 姿态检测失败: {str(e)}")
            return None

        # 输出预测结果gif图


        # 测试姿态分析
        print("\n4. 测试姿态分析...")
        try:
            # 获取朝向和关键点索引
            facing_direction = find_camera_facing_side(self.all_keypoints[0])
            self.front_indices = get_front_keypoint_indices(facing_direction)
            print(f"  - 检测到的朝向: {facing_direction}")
            print(f"  - 关键点索引: {self.front_indices}")
            
            # 测试膝盖角度计算
            print("\n  测试膝盖角度计算:")
            # 获取一些示例帧的膝盖角度
            hip_knee_ankle_indices = self.front_indices[:4]
            sample_angles = [get_hip_knee_ankle_angle(kp, hip_knee_ankle_indices) for kp in self.all_keypoints[:5]]
            print(f"  - 前5帧的膝盖角度: {[f'{angle:.2f}°' for angle in sample_angles]}")
            
            # 测试最低点检测
            self.lowest_pedal_point_indices = get_lowest_pedal_frames(self.all_keypoints, hip_knee_ankle_indices)
            print(f"  - 检测到的最低点帧索引: {self.lowest_pedal_point_indices[:5]}")
            
            # 测试最高点检测
            self.highest_pedal_point_indices = get_highest_pedal_frames(self.all_keypoints, hip_knee_ankle_indices)
            print(f"  - 检测到的最高点帧索引: {self.highest_pedal_point_indices[:5]}")
            
            # 获取完整结果
            result = self.get_pose()
            print("\n姿态分析结果:")
            print(f"  - 最低点膝盖角度: {result['knee_angle_lowest']:.2f}°")
            print(f"  - 最高点膝盖角度: {result['knee_angle_highest']:.2f}°")
            print(f"  - 肩膀角度: {result['shoulder_angle']:.2f}°")
            print(f"  - 手肘角度: {result['elbow_angle']:.2f}°")
            print(f"  - 最低点髋关节角度: {result['hip_angle_lowest']:.2f}°")
            print(f"  - 最高点髋关节角度: {result['hip_angle_highest']:.2f}°")
            print("✓ 姿态分析成功")

            # 创建保存结果的文件夹
            output_dir = os.path.join(current_dir, 'output_frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 提取并绘制最低点和最高点的帧
            concat_indices = np.concatenate([self.lowest_pedal_point_indices, self.highest_pedal_point_indices])
            
            # 创建一个列表来存储所有绘制的帧
            output_images = []
            
            # 处理所有帧并生成GIF
            print("\n5. 生成姿态检测可视化...")
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
                            print(f'✓ 保存关键帧: {output_path}')
                            print(f"绘制关键帧索引: {idx}")
                        # 添加到输出图像列表
                        output_images.append(result_frame)
                
                # 然后处理所有帧以生成完整的GIF（可选，取决于性能需求）
                print("\n生成完整GIF动画...")
                all_output_images = []
                # 为了性能考虑，可以选择每隔几帧处理一次
                step = max(1, len(frames) // 50)  # 最多处理50帧
                for idx in tqdm(range(0, len(frames), step), desc="处理帧"):
                    if idx % 10 == 0:  # 每10帧显示一次进度
                        print(f"处理帧 {idx}/{len(frames)}")
                    
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
                print(f"✓ 成功保存GIF动画: {gif_path}")
                
            except Exception as e:
                print(f"✗ 生成可视化失败: {str(e)}")
                traceback.print_exc()

            return result
        except Exception as e:
            print(f"✗ 姿态分析失败: {str(e)}")
            print(traceback.format_exc())
            return None

    
# TODO 需要增加帧返回值
    def pose_analyzer(self, file: Union[str, bytes]) -> tuple[dict,list[np.ndarray]]:
        """
        姿态分析器的主函数，接收文件路径或字节数据，处理后返回姿态分析结果。

        参数:
            file (str | bytes): 文件路径或字节数据。

        返回:
            dict: 包含姿态分析结果的字典。
        """
        # 初始化模型
        print("1. 初始化模型...")
        try:
            print("✓ 模型初始化成功")
        except Exception as e:
            return {"error": f"模型初始化失败: {str(e)}"},[]
        
        # 预处理视频
        print("\n2. 预处理视频...")
        try:
            frames, tensors = pre_process_video(file)
            print(f"  - 处理的视频帧数: {len(frames)}")
            print(f"  - 张量形状: {tensors.shape}")
            print(f"  - 张量数据类型: {tensors.dtype}")
            print(f"  - 张量值范围: [{tf.reduce_min(tensors):.2f}, {tf.reduce_max(tensors):.2f}]")
            print("✓ 视频预处理成功")
        except Exception as e:
            return {"error": f"视频预处理失败: {str(e)}"},[]

        # 姿态检测
        print("\n3. 姿态检测...")
        try:
            self.all_keypoints = get_keypoints_from_video(tensors, self.model, self.input_size)
            print(f"  - 关键点数量: {len(self.all_keypoints)}")
            print(f"  - 单帧关键点形状: {self.all_keypoints[0].shape}")
            print("✓ 姿态检测成功")
        except Exception as e:
            return {"error": f"姿态检测失败: {str(e)}"},[]

        # 姿态分析
        print("\n4. 姿态分析...")
        try:
            # 获取朝向和关键点索引
            facing_direction = find_camera_facing_side(self.all_keypoints[0])
            self.front_indices = get_front_keypoint_indices(facing_direction)
            print(f"  - 检测到的朝向: {facing_direction}")
            print(f"  - 关键点索引: {self.front_indices}")

            # 测试最低点检测
            self.lowest_pedal_point_indices = get_lowest_pedal_frames(self.all_keypoints, self.front_indices)
            print(f"  - 检测到的最低点帧索引: {self.lowest_pedal_point_indices[:5]}")
            
            # 测试最高点检测
            self.highest_pedal_point_indices = get_highest_pedal_frames(self.all_keypoints, self.front_indices)
            print(f"  - 检测到的最高点帧索引: {self.highest_pedal_point_indices[:5]}")

            # 获取完整结果
            result = self.get_pose()
            print("\n姿态分析结果:")
            print(f"  - 最低点膝盖角度: {result['knee_angle_lowest']:.2f}°")
            print(f"  - 最高点膝盖角度: {result['knee_angle_highest']:.2f}°")
            print(f"  - 肩膀角度: {result['shoulder_angle']:.2f}°")
            print(f"  - 手肘角度: {result['elbow_angle']:.2f}°")
            print(f"  - 最低点髋关节角度: {result['hip_angle_lowest']:.2f}°")
            print(f"  - 最高点髋关节角度: {result['hip_angle_highest']:.2f}°")
            print("✓ 姿态分析成功")

            # 提取并绘制最低点和最高点的帧
            # TODO 返回帧保存在result_frames中
            concat_indices = np.concatenate([self.lowest_pedal_point_indices, self.highest_pedal_point_indices])
            result_frames = []
            
            # 创建保存结果的文件夹
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(current_dir, 'output_frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 处理关键帧并生成GIF
            print("\n5. 生成姿态检测可视化...")
            try:
                # 处理关键帧（最低点和最高点）
                output_images = []
                for idx in tqdm(concat_indices, desc="绘制关键帧"):
                    if idx < len(frames):  # 确保索引在有效范围内
                        frame = frames[idx]  # 使用切片好的frames变量
                        # 在帧上绘制关键点和骨架
                        keyPoints = self.all_keypoints[idx]
                        result_frame = draw_prediction_on_image(
                            frame, 
                            keyPoints,
                            crop_region=None,
                            output_image_height=300,
                            front_indices=self.front_indices
                        )
                        # 保存结果帧
                        output_path = os.path.join(output_dir, f'frame_{idx}.png')
                        cv2.imwrite(output_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
                        
                        # 添加到输出图像列表和结果帧列表
                        output_images.append(result_frame)
                        result_frames.append(result_frame)
                    else:
                        print(f"✗ 索引超出范围: {idx}")

                print(f"保存目录：{output_dir}")
                
                # 生成完整的GIF动画
                print("\n生成完整GIF动画...")
                all_output_images = []
                # 为了性能考虑，可以选择每隔几帧处理一次
                step = max(1, len(frames) // 50)  # 最多处理50帧
                for idx in tqdm(range(0, len(frames), step), desc='处理关键帧'):
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
                print(f"✓ 成功保存GIF动画: {gif_path}")
                
            except Exception as e:
                print(f"✗ 生成可视化失败: {str(e)}")
                traceback.print_exc()

            # 返回结果
            return result, result_frames
        except Exception as e:
            return {"error": f"姿态分析失败: {str(e)}"},[]

if __name__ == "__main__":
    analyzer = PoseAnalyzer()

