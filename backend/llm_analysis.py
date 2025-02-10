from typing import Dict, Any
import os
from openai import OpenAI

class BikeFitAnalyzer:
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 定义理想的角度范围
        self.ideal_ranges = {
            'knee_angle_lowest': (65, 75),  # 最低点膝关节角度理想范围
            'knee_angle_highest': (140, 150),  # 最高点膝关节角度理想范围
            'hip_angle': (45, 90),  # 髋关节角度理想范围
            'shoulder_angle': (20, 45),  # 肩膀角度理想范围
            'elbow_angle': (150, 165),  # 手肘角度理想范围
        }

    def _generate_prompt(self, measurements: Dict[str, Any]) -> str:
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

请用中文回答，以易于理解的方式组织回答。"""

        return prompt.format(**measurements)

    def analyze_pose(self, measurements: Dict[str, Any]) -> str:
        """分析骑行姿势并生成建议"""
        try:
            # 生成提示
            prompt = self._generate_prompt(measurements)
            
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",  # 或者使用其他可用的模型
                messages=[
                    {"role": "system", "content": "你是一个专业的自行车适配专家，擅长分析骑行姿势并提供改善建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 返回分析结果
            return response.choices[0].message.content
            
        except Exception as e:
            return f"分析过程中出现错误: {str(e)}"

    def get_quick_summary(self, measurements: Dict[str, Any]) -> Dict[str, str]:
        """生成每个角度的快速评估"""
        summary = {}
        for angle_name, value in measurements.items():
            if angle_name in self.ideal_ranges:
                min_val, max_val = self.ideal_ranges[angle_name]
                if value < min_val:
                    summary[angle_name] = "过小"
                elif value > max_val:
                    summary[angle_name] = "过大"
                else:
                    summary[angle_name] = "适中"
        return summary 