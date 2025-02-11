from typing import Dict, Any
import os
from pose_detection import upload_video
from local_model import LocalModelProcessor
from api_model import APIModelProcessor

class BikeFitAdvisor:
    def __init__(self, use_api: bool = False, api_key: str = None):
        """初始化自行车适配顾问
        
        Args:
            use_api: 是否使用在线API
            api_key: API密钥（仅在use_api=True时需要）
        """
        # 定义理想的角度范围
        self.ideal_ranges = {
            'knee_angle_lowest': (65, 75),  # 最低点膝关节角度理想范围
            'knee_angle_highest': (140, 150),  # 最高点膝关节角度理想范围
            'hip_angle': (45, 90),  # 髋关节角度理想范围
            'shoulder_angle': (20, 45),  # 肩膀角度理想范围
            'elbow_angle': (150, 165),  # 手肘角度理想范围
        }
        
        # 初始化模型处理器
        if use_api:
            if not api_key:
                raise ValueError("使用API模式时必须提供API密钥")
            self.model = APIModelProcessor(api_key)
            print("使用在线API模式")
        else:
            self.model = LocalModelProcessor()
            print("使用本地模型模式")

    def analyze_pose(self, measurements):
        """分析姿态并提供建议"""
        prompt = f"""
        As a professional bike fitting expert, please analyze the following cycling posture data:
        {measurements}
        Please provide professional advice.
        """
        return self.model.generate_response(prompt)

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

def test_advisor():
    """测试自行车适配顾问"""
    # 测试API模式
    api_key = "sk-4f1bb64e2d5b4099b69ceaf8ab0d8d72"
    advisor_api = BikeFitAdvisor(use_api=True, api_key=api_key)
    
    # 测试本地模型模式
    # advisor_local = BikeFitAdvisor(use_api=False)
    
    # 使用示例数据进行测试
    test_measurements = {
        'knee_angle_lowest': 70,
        'knee_angle_highest': 145,
        'shoulder_angle': 30,
        'elbow_angle': 160,
        'hip_angle': 60
    }
    
    print("\nTesting API mode:")
    analysis_api = advisor_api.analyze_pose(test_measurements)
    print("API Analysis:", analysis_api)
    
    # print("\nTesting local model mode:")
    # analysis_local = advisor_local.analyze_pose(test_measurements)
    # print("Local Analysis:", analysis_local)

if __name__ == "__main__":
    test_advisor() 