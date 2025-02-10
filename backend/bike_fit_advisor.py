from typing import Dict, Any
import os
from pose_analyzer import upload_video
from local_llm import LocalLLMProcessor

class BikeFitAdvisor:
    def __init__(self):
        # 定义理想的角度范围
        self.ideal_ranges = {
            'knee_angle_lowest': (65, 75),  # 最低点膝关节角度理想范围
            'knee_angle_highest': (140, 150),  # 最高点膝关节角度理想范围
            'hip_angle': (45, 90),  # 髋关节角度理想范围
            'shoulder_angle': (20, 45),  # 肩膀角度理想范围
            'elbow_angle': (150, 165),  # 手肘角度理想范围
        }

    def analyze_pose(self,measurements):
        """分析姿态并提供建议"""
        llm = LocalLLMProcessor()
        prompt = f"""
        As a professional bike fitting expert, please analyze the following cycling posture data:
        {measurements}
        Please provide professional advice.
        """
        return llm.generate_response(prompt)
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """分析视频并提供建议"""
        try:
            # 确保视频路径是绝对路径
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise FileNotFoundError(f"视频文件不存在: {abs_video_path}")
                
            # 获取姿态数据
            pose_data = upload_video(abs_video_path)
            
            # 提取需要分析的测量数据
            measurements = {
                'knee_angle_lowest': pose_data['knee_angle_lowest'],
                'knee_angle_highest': pose_data['knee_angle_highest'],
                'shoulder_angle': pose_data['shoulder_angle'],
                'elbow_angle': pose_data['elbow_angle'],
                'hip_angle': pose_data['hip_angle']
            }
            
            # 获取详细分析
            analysis = self.analyze_pose(measurements)
            
            # 返回完整结果
            return {
                **measurements,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'error': f"视频分析过程中出现错误: {str(e)}"
            }

def main():
    """主函数，用于测试"""
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "uploads", "raw.mp4")
    
    advisor = BikeFitAdvisor()
    
    print(f"\n分析视频: {video_path}")
    result = advisor.analyze_video(video_path)
    
    print("\n详细分析:")
    print(result)

if __name__ == "__main__":
    main() 