from flask import Flask, request, jsonify
import os
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import kagglehub
import traceback
from scipy.signal import find_peaks
from typing import Union

from .model import load_model_from_tfhub, get_keypoints_from_video
from .preprocessing import pre_process_video
from .postprocessing import (find_camera_facing_side,
                          get_front_keypoint_indices,
                          get_lowest_pedal_frames,
                          get_highest_pedal_frames,
                          get_hip_knee_ankle_angle,
                          calculate_angle)

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
            hip_knee_ankle_indices
        )
        
        # 获取膝盖最高点角度的平均数
        knee_angle_highest = self.get_knee_angle_at_highest_pedal_points_avg(
            hip_knee_ankle_indices
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
            'knee_angle_lowest': knee_angle_lowest,
            'knee_angle_highest': knee_angle_highest,
            'shoulder_angle': shoulder_angle,
            'elbow_angle': elbow_angle,
            'hip_angle_lowest': hip_angle_lowest,
            'hip_angle_highest': hip_angle_highest,
        }

        return measurements

    #获取膝盖最大角度的平均数
    def get_knee_angle_at_lowest_pedal_points_avg(self, hip_knee_ankle_indices):
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
    def get_knee_angle_at_highest_pedal_points_avg(self, hip_knee_ankle_indices):
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
    
    # # 定义draw_pose函数，用于在帧上绘制关键点和角度
    # def draw_pose(self, frame, keypoints):
    #     if self.all_keypoints is None:
    #         raise ValueError('all_keypoints未初始化')
    #     if self.front_indices is None:
    #         raise ValueError('front_indices未初始化')
    #     # 确保input_size有效
    #     if self.input_size is None:
    #         raise ValueError('input_size未初始化')
    #     # 计算缩放比例
    #     scale_x = self.input_size
    #     scale_y = self.input_size

    #     # 在这里实现绘制逻辑，例如使用cv2.circle绘制关键点
    #     for idx in self.front_indices:
    #         if idx < len(keypoints):  # 确保索引在有效范围内
    #             kp = keypoints[idx]
    #             # 还原坐标
    #             x = int(kp[0] * scale_x)
    #             y = int(kp[1] * scale_y)
    #             print(f"绘制关键点坐标: ({x}, {y})")
    #             cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 绘制绿色圆点
    #     return frame

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
            return

        # 测试视频预处理
        print("\n2. 测试视频预处理...")
        try:
            # 使用当前目录下的测试视频
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            uploads_dir = os.path.join(current_dir, "uploads")

            if not os.path.exists(uploads_dir):
                print(f"✗ 上传目录不存在: {uploads_dir}")
                return

            # 遍历uploads目录中的所有文件
            for filename in os.listdir(uploads_dir):
                video_path = os.path.join(uploads_dir, filename)
                if not os.path.exists(video_path):
                    print(f"✗ 测试视频不存在: {video_path}")
                    continue
                frames, tensors = pre_process_video(video_path)

           # test_video_path = os.path.join(current_dir, "uploads", "raw.mp4")
            
            #if not os.path.exists(test_video_path):
               # print(f"✗ 测试视频不存在: {test_video_path}")
               # return
            
            #frames, tensors = pre_process_video(test_video_path)
            
            # 验证预处理结果
                print(f"  - 处理的视频帧数: {len(frames)}")
                print(f"  - 张量形状: {tensors.shape}")
                print(f"  - 张量数据类型: {tensors.dtype}")
                print(f"  - 张量值范围: [{tf.reduce_min(tensors):.2f}, {tf.reduce_max(tensors):.2f}]")
                print("✓ 视频预处理成功")
        #except Exception as e:
          #  print(f"✗ 视频预处理失败: {str(e)}")
          #  return None

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
                    for idx in concat_indices:
                        print(f"绘制帧索引: {idx}")

                        if idx < len(frames):  # 确保索引在有效范围内
                            frame = frames[idx]  # 使用切片好的frames变量
                            # 在帧上绘制front_indices中的关键点
                            # keypoints = self.all_keypoints[idx]
                            # result_frame = self.draw_pose(frame,keypoints)
                            result_frame = frame
                            # 保存结果帧
                            output_path = os.path.join(output_dir, f'frame_{idx}.png')
                            cv2.imwrite(output_path, result_frame)
                            print(f'✓ 保存帧: {output_path}')
                            # # 显示结果帧
                            # cv2.imshow('Pose Result', result_frame)
                            # cv2.waitKey(100)  # 显示每帧100毫秒
                        else:
                            print(f"✗ 索引超出范围: {idx}")

                    print(result)
                    continue
                except Exception as e:
                    print(f"✗ 姿态分析失败: {str(e)}")
                    print(traceback.format_exc())
                    return None
        except Exception as e:
            print(f"✗ 视频预处理失败: {str(e)}")
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
            for idx in concat_indices:
                print(f"绘制帧索引: {idx}")

                if idx < len(frames):  # 确保索引在有效范围内
                    result_frame = frames[idx]
                    result_frames.append(result_frame)
                    print(f'✓ 保存帧')
                else:
                    print(f"✗ 索引超出范围: {idx}")

            # 返回结果
            return result,result_frames
        except Exception as e:
            return {"error": f"姿态分析失败: {str(e)}"},[]

if __name__ == "__main__":
    analyzer = PoseAnalyzer()
    analyzer.test_pose_analyzer()

