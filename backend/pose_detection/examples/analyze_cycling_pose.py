#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
骑行姿态分析示例脚本

该脚本演示如何使用pose_detection包分析骑行视频并可视化结果。
"""

import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt

# 添加上层目录到路径，以便能够导入pose_detection包
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pose_detection import PoseAnalyzer

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='分析骑行视频并输出姿态评估结果')
    parser.add_argument('video_path', type=str, help='骑行视频的路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出结果的目录')
    parser.add_argument('--save_frames', action='store_true', help='是否保存关键帧图片')
    parser.add_argument('--debug', action='store_true', help='是否打印调试信息')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 打印欢迎信息
    print("=" * 80)
    print("自行车骑行姿态分析示例")
    print("=" * 80)
    print(f"视频路径: {args.video_path}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 80)
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 找不到视频文件 {args.video_path}")
        return 1
        
    try:
        # 初始化姿态分析器
        print("正在初始化姿态分析器...")
        analyzer = PoseAnalyzer()
        
        # 分析视频
        print("正在分析视频，请稍候...")
        results, frames, gif_path = analyzer.pose_analyzer(args.video_path)
        
        # 输出分析结果
        print("\n分析结果:")
        print("-" * 40)
        if "error" in results:
            print(f"错误: {results['error']}")
            return 1
            
        print(f"膝关节角度 (最低点): {results['knee_angle_lowest']}°")
        print(f"膝关节角度 (最高点): {results['knee_angle_highest']}°")
        print(f"髋关节角度 (最低点): {results['hip_angle_lowest']}°")
        print(f"髋关节角度 (最高点): {results['hip_angle_highest']}°")
        print(f"肩膀角度: {results['shoulder_angle']}°")
        print(f"肘关节角度: {results['elbow_angle']}°")
        print("-" * 40)
        
        # 如果有GIF动画结果，复制到输出目录
        if gif_path and os.path.exists(gif_path):
            output_gif_path = os.path.join(args.output_dir, 'pose_animation.gif')
            import shutil
            shutil.copy(gif_path, output_gif_path)
            print(f"动画已保存到: {output_gif_path}")
        
        # 如果需要保存关键帧图片
        if args.save_frames and frames and len(frames) > 0:
            print("\n保存关键帧...")
            for i, frame in enumerate(frames):
                output_frame_path = os.path.join(args.output_dir, f'frame_{i}.png')
                cv2.imwrite(output_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"保存帧 {i}: {output_frame_path}")
        
        # 可视化角度比较
        print("\n生成角度比较图...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        angles = [
            results['knee_angle_lowest'], 
            results['knee_angle_highest'],
            results['hip_angle_lowest'],
            results['hip_angle_highest'],
            results['shoulder_angle'],
            results['elbow_angle']
        ]
        
        angle_names = [
            '膝关节(最低点)', 
            '膝关节(最高点)',
            '髋关节(最低点)',
            '髋关节(最高点)',
            '肩膀角度',
            '肘关节角度'
        ]
        
        # 理想角度范围，用于比较
        ideal_ranges = [
            (150, 155),  # 膝关节最低点
            (65, 70),    # 膝关节最高点
            (None, None),  # 髋关节最低点(因人而异)
            (None, None),  # 髋关节最高点(因人而异)
            (85, 95),    # 肩膀角度
            (160, 170)   # 肘关节角度
        ]
        
        bars = ax.bar(angle_names, angles, color='royalblue')
        
        # 在柱状图上添加理想范围标记
        for i, (low, high) in enumerate(ideal_ranges):
            if low is not None and high is not None:
                ax.plot([i-0.4, i+0.4], [low, low], 'r--')
                ax.plot([i-0.4, i+0.4], [high, high], 'r--')
        
        # 在柱状图顶部显示具体角度值
        for bar, angle in zip(bars, angles):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{angle}°', ha='center', va='bottom')
        
        ax.set_ylabel('角度 (度)')
        ax.set_title('骑行姿态角度分析')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存角度比较图
        angle_chart_path = os.path.join(args.output_dir, 'angle_chart.png')
        plt.savefig(angle_chart_path)
        print(f"角度比较图已保存到: {angle_chart_path}")
        
        print("\n分析完成！")
        return 0
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 