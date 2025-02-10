import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from main import get_knee_angle_at_lowest_pedal_points_avg

# 假设这些是被测试方法依赖的外部函数
@patch('main.get_hip_knee_ankle_angle')
@patch('main.get_lowest_pedal_frames')
@patch('main.filter_bad_knee_angles')
class TestGetKneeAngleAtLowestPedalPointsAvg(unittest.TestCase):

    def setUp(self):
        # 设置测试数据
        self.all_keypoints = [
            np.array(
                [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11],
                 [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19]]),
            np.array(
                [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12],
                 [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20]]),
            np.array(
                [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13],
                 [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21]])
        ]
        self.front_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.hip_knee_ankle_indices = self.front_indices[:4]

    def test_get_knee_angle_at_lowest_pedal_points_avg(self, mock_filter, mock_lowest_pedal_frames, mock_get_angle):
        # 模拟行为
        mock_get_angle.side_effect = lambda kp, indices: (0, 10)  # 假设角度为10
        mock_lowest_pedal_frames.return_value = [0, 1, 2]  # 假设最低踏板帧为0, 1, 2
        mock_filter.return_value = ([10, 10, 10], [0, 1, 2])  # 假设过滤后角度为10

        result = get_knee_angle_at_lowest_pedal_points_avg(self.all_keypoints, self.front_indices)

        # 验证结果
        self.assertAlmostEqual(result, 10.0, places=2)

        # 验证模拟调用
        mock_get_angle.assert_called_with(self.all_keypoints[2], self.hip_knee_ankle_indices)
        mock_lowest_pedal_frames.assert_called_with(self.all_keypoints, self.hip_knee_ankle_indices)
        mock_filter.assert_called_with([10, 10, 10], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
