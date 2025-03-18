# 姿态检测示例图片

此目录用于存放骑行姿态检测的示例图片和动画，用于README和文档展示。

## 示例图片列表

应包含以下示例图片：

1. `pose_detection_example.gif` - 骑行姿态检测的GIF动画示例
2. `knee_angle_example.png` - 膝关节角度测量示例
3. `hip_angle_example.png` - 髋关节角度测量示例
4. `shoulder_elbow_example.png` - 肩部和肘部角度测量示例

## 如何生成示例图片

可以通过以下方式生成示例图片：

1. 使用`PoseAnalyzer`类的`pose_analyzer`方法处理骑行视频
2. 从返回的`frames`和`gif_path`获取示例图片
3. 使用以下代码片段保存图片：

```python
import cv2
from pose_detection import PoseAnalyzer

# 初始化姿态分析器
analyzer = PoseAnalyzer()

# 分析视频文件
results, frames, gif_path = analyzer.pose_analyzer("example_cycling_video.mp4")

# GIF动画已自动保存到gif_path

# 保存关键帧作为示例图片
if frames and len(frames) > 0:
    # 保存膝关节角度示例
    cv2.imwrite("docs/images/knee_angle_example.png", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    
    # 保存髋关节角度示例
    if len(frames) > 1:
        cv2.imwrite("docs/images/hip_angle_example.png", cv2.cvtColor(frames[1], cv2.COLOR_RGB2BGR))
    
    # 保存肩部和肘部角度示例
    if len(frames) > 2:
        cv2.imwrite("docs/images/shoulder_elbow_example.png", cv2.cvtColor(frames[2], cv2.COLOR_RGB2BGR))
```

## 注意事项

1. 示例图片应保持良好的视觉效果，清晰展示骨架和关节角度
2. 图片分辨率建议保持在800×600以上
3. GIF动画帧率建议设置为10-15fps，文件大小控制在2MB以内
4. 图片命名应遵循上述规范，以便文档引用 