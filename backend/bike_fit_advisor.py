import json
from typing import Dict, Any
import os

from sympy.polys.polyconfig import query

from pose_detection.pose_analyzer import PoseAnalyzer
# from local_model import LocalModelProcessor
# from api_model import APIModelProcessor

# from local_rag.chat import get_model_response

class BikeFitAdvisor:
    """
        TODO: 决定是否由用户信息（身高，体重 等信息动态生成 ideal_ranges）
    """
    def __init__(self, use_api: bool = False, api_key: str = None,measurements = None):
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
        self.measurements = measurements
        #
        # # 初始化模型处理器
        # if use_api:
        #     if not api_key:
        #         raise ValueError("使用API模式时必须提供API密钥")
        #     self.model = APIModelProcessor(api_key)
        #     print("使用在线API模式")
        # else:
        #     self.model = LocalModelProcessor()
        #     print("使用本地模型模式")

    def generate_prompt(self,source=None):
        """分析姿态并提供建议"""
        query_prompt = f"用户query的内容如下：{source}" if source is not None else ""
        prompt = f"""
        你是一个专业的数据分析助手，现在需要根据用户查询和骑手的测量信息生成一个准确的分析报告。

        {query_prompt}
        
        骑手的测量信息如下：
        {self.measurements}
        
        请根据上述信息，生成一个详细的分析报告，包括但不限于以下内容：
        1. 骑手的健康状况评估；
        2. 骑行效率分析；
        3. 基于数据的改进建议。
        
        要求：
        - 报告内容清晰、逻辑连贯；
        - 使用专业术语进行描述；
        - 提供数据支持的结论。
        """
        prompt_en = f"""
        As a professional bicycle fitter, please make an analysis based on the following riding position data: 
        - measurements: {self.measurements} 
        Please provide professional advice based on this information, including but not limited to: 
        1. Evaluation of the current state of riding posture. 
        2. Any possible problems and their impact on riding efficiency or rider health. 
        3. Suggestions for improvement, such as adjusting seat height, handle position and other specific measures. 
        4. Recommend products or training methods that will help improve the riding experience, if applicable. 
        Make sure your advice is both scientific and practical to help the rider improve comfort and riding performance.
        """
        return prompt


    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """分析视频并提供建议"""
        try:
            # 确保视频路径是绝对路径
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise FileNotFoundError(f"视频文件不存在: {abs_video_path}")
                
            # 获取姿态数据
            pose_data,result_frame = PoseAnalyzer().pose_analyzer(abs_video_path)
            
            # 提取需要分析的测量数据
            measurements = {
                'knee_angle_lowest': pose_data['knee_angle_lowest'],
                'knee_angle_highest': pose_data['knee_angle_highest'],
                'shoulder_angle': pose_data['shoulder_angle'],
                'elbow_angle': pose_data['elbow_angle'],
                'hip_angle': pose_data['hip_angle']
            }
            
            # 获取详细分析
            analysis = self.generate_query_prompt(measurements)
            
            # 返回完整结果
            return {
                **measurements,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'error': f"视频分析过程中出现错误: {str(e)}"
            }


    def stream_advisor(self, measurements={
        'knee_angle_lowest': 70,
        'knee_angle_highest': 145,
        'shoulder_angle': 30,
        'elbow_angle': 160,
        'hip_angle': 60
    }):
        """测试自行车适配顾问，使用stream返回"""

        test_prompt = self.generate_query_prompt(measurements)

        
        yield {"type": "info", "message": measurements}

        full_reasoning = ""
        full_content = ""
        is_answering = False


        for reasoning, content in self.model.generate_response(test_prompt):
            if reasoning:
                full_reasoning += reasoning
                yield {"type": "reasoning", "message": reasoning}

            if content:
                if not is_answering:
                    is_answering = True
                full_content += content
                yield {"type": "response", "message": content}

            

        yield {"type": "done", "message": "Analysis complete."}
        print("Analysis complete.")
    #
    # def get_streaming_advice_based_on_rag(self, measurements):
    #     """使用RAG模型进行自行车适配顾问"""
    #     # Create a mock history with just the current question
    #     history = [[measurements, None]]
    #
    #     # Use local_rag's get_model_response
    #     for response in get_model_response(
    #         {'text': measurements, 'files': []},
    #         history,
    #         model='qwen-max',
    #         temperature=0.7,
    #         max_tokens=1024,
    #         history_round=1,
    #         db_name='bike-fit',
    #         similarity_threshold=0.2,
    #         chunk_cnt=5
    #     ):
    #         yield json.dumps({"type": "response", "message": response[1]}) + "\n"

if __name__ == "__main__":
    test_video_path = os.getenv("test_video_path")
    analyzer = PoseAnalyzer()
    result,result_frames = analyzer.pose_analyzer(test_video_path)
    print(result)
    test = BikeFitAdvisor(use_api=True, api_key=os.getenv("DASHSCOPE_API_KEY"))
    test.test_advisor(measurements=result) 