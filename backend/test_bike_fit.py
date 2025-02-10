import os
import unittest
from pose_analyzer import upload_video
from bike_fit_advisor import BikeFitAdvisor

class TestBikeFit(unittest.TestCase):
    def setUp(self):
        """测试开始前的设置"""
        self.test_video_dir = "./uploads"
        # 确保测试视频目录存在
        if not os.path.exists(self.test_video_dir):
            os.makedirs(self.test_video_dir)

    def test_video_analysis(self):
        """测试视频分析和姿势建议功能"""
        # 获取uploads目录中的第一个视频文件
        video_files = [f for f in os.listdir(self.test_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            self.skipTest("没有找到测试视频文件")
        
        test_video_path = os.path.join(self.test_video_dir, video_files[0])
        print(f"\n测试视频文件: {test_video_path}")

        try:
            # 创建顾问对象并分析视频
            advisor = BikeFitAdvisor()
            result = advisor.analyze_video(test_video_path)
            
            # 验证返回结果包含所有必要的键
            expected_keys = [
                'knee_angle_lowest',
                'knee_angle_highest',
                'shoulder_angle',
                'elbow_angle',
                'hip_angle',
                'knee_angles',
                'analysis',
                'summary'
            ]
            
            for key in expected_keys:
                self.assertIn(key, result, f"结果中缺少 {key}")
                
            # 验证数值是否在合理范围内
            self.assertGreaterEqual(result['knee_angle_lowest'], 0, "膝盖最低角度不应小于0度")
            self.assertLessEqual(result['knee_angle_lowest'], 180, "膝盖最低角度不应大于180度")
            
            self.assertGreaterEqual(result['knee_angle_highest'], 0, "膝盖最高角度不应小于0度")
            self.assertLessEqual(result['knee_angle_highest'], 180, "膝盖最高角度不应大于180度")
            
            self.assertGreaterEqual(result['shoulder_angle'], 0, "肩膀角度不应小于0度")
            self.assertLessEqual(result['shoulder_angle'], 180, "肩膀角度不应大于180度")
            
            self.assertGreaterEqual(result['elbow_angle'], 0, "手肘角度不应小于0度")
            self.assertLessEqual(result['elbow_angle'], 180, "手肘角度不应大于180度")

            self.assertGreaterEqual(result['hip_angle'], 0, "髋关节角度不应小于0度")
            self.assertLessEqual(result['hip_angle'], 180, "髋关节角度不应大于180度")
            
            # 验证分析结果
            self.assertIsInstance(result['analysis'], str, "分析结果应该是字符串")
            self.assertGreater(len(result['analysis']), 0, "分析结果不应为空")

            # 验证总结结果
            self.assertIsInstance(result['summary'], dict, "总结结果应该是字典")
            
            # 打印测试结果
            print("\n测量结果:")
            print(f"膝盖最低点角度: {result['knee_angle_lowest']:.2f}°")
            print(f"膝盖最高点角度: {result['knee_angle_highest']:.2f}°")
            print(f"肩膀角度: {result['shoulder_angle']:.2f}°")
            print(f"手肘角度: {result['elbow_angle']:.2f}°")
            print(f"髋关节角度: {result['hip_angle']:.2f}°")
            print(f"检测到的帧数: {len(result['knee_angles'])}")
            
            print("\n姿势分析:")
            print(result['analysis'])
            
            print("\n快速总结:")
            for key, value in result['summary'].items():
                print(f"{key}: {value}")
            
        except Exception as e:
            self.fail(f"测试失败，错误信息: {str(e)}")

if __name__ == '__main__':
    unittest.main() 