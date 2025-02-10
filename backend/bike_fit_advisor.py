from typing import Dict, Any
import os
from pose_analyzer import upload_video
from local_llm import LocalLLMProcessor

class BikeFitAdvisor:
    def __init__(self):
        # 初始化本地LLM处理器
        self.llm = LocalLLMProcessor()
        
        # 定义理想的角度范围
        self.ideal_ranges = {
            'knee_angle_lowest': (65, 75),  # 最低点膝关节角度理想范围
            'knee_angle_highest': (140, 150),  # 最高点膝关节角度理想范围
            'hip_angle': (45, 90),  # 髋关节角度理想范围
            'shoulder_angle': (20, 45),  # 肩膀角度理想范围
            'elbow_angle': (150, 165),  # 手肘角度理想范围
        }

    def _generate_prompt(self, measurements: Dict[str, float]) -> str:
        """生成用于LLM的提示"""
        prompt = """作为一个专业的自行车适配专家，请根据以下测量数据分析骑行姿势并给出改善建议：

测量数据：
- 踩踏最低点膝关节角度: {knee_angle_lowest:.1f}° (理想范围: 65-75°)
- 踩踏最高点膝关节角度: {knee_angle_highest:.1f}° (理想范围: 140-150°)
- 髋关节角度: {hip_angle:.1f}° (理想范围: 45-90°)
- 肩膀角度: {shoulder_angle:.1f}° (理想范围: 20-45°)
- 手肘角度: {elbow_angle:.1f}° (理想范围: 150-165°)

请分析以上数据，并：
1. 评估每个关节角度是否在理想范围内
2. 解释每个角度对骑行的影响
3. 给出具体的调整建议（例如：调整座高、车把高度等）
4. 说明这些调整如何改善骑行体验和效率

请用中文回答，以易于理解的方式组织回答。重点关注需要调整的部分，并给出明确的改进建议。"""

        return prompt.format(**measurements)

    def get_quick_summary(self, measurements: Dict[str, float]) -> Dict[str, str]:
        """生成每个角度的快速评估"""
        summary = {}
        for angle_name, value in measurements.items():
            if angle_name in self.ideal_ranges and isinstance(value, (int, float)):
                min_val, max_val = self.ideal_ranges[angle_name]
                if value < min_val:
                    summary[angle_name] = "过小"
                elif value > max_val:
                    summary[angle_name] = "过大"
                else:
                    summary[angle_name] = "适中"
        return summary

    def analyze_pose(self, measurements: Dict[str, float]) -> str:
        """分析骑行姿势并生成建议"""
        try:
            # 生成提示
            prompt = self._generate_prompt(measurements)
            
            # 使用本地LLM生成回复
            response = self.llm.generate_response(prompt)
            
            return response
            
        except Exception as e:
            return f"分析过程中出现错误: {str(e)}"

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """分析视频并提供建议"""
        try:
            # 获取姿态数据
            pose_data = upload_video(video_path)
            
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
            
            # 获取快速总结
            summary = self.get_quick_summary(measurements)
            
            # 返回完整结果
            return {
                **pose_data,  # 包含原始姿态数据
                'analysis': analysis,  # 添加详细分析
                'summary': summary  # 添加快速总结
            }
            
        except Exception as e:
            return {
                'error': f"视频分析过程中出现错误: {str(e)}"
            }

def main():
    """主函数，用于测试"""
        
    video_path = "./uploads/raw.mp4"
    advisor = BikeFitAdvisor()
    
    print(f"\n分析视频: {video_path}")
    result = advisor.analyze_video(video_path)
    
    if 'error' in result:
        print(f"\n错误: {result['error']}")
    else:
        print("\n测量结果:")
        print(f"膝盖最低点角度: {result['knee_angle_lowest']:.2f}°")
        print(f"膝盖最高点角度: {result['knee_angle_highest']:.2f}°")
        print(f"肩膀角度: {result['shoulder_angle']:.2f}°")
        print(f"手肘角度: {result['elbow_angle']:.2f}°")
        print(f"髋关节角度: {result['hip_angle']:.2f}°")
        
        print("\n快速总结:")
        for key, value in result['summary'].items():
            print(f"{key}: {value}")
            
        print("\n详细分析:")
        print(result['analysis'])

if __name__ == "__main__":
    main() 