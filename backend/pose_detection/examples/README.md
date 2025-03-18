# 骑行姿态分析示例

本目录包含使用`pose_detection`包进行骑行姿态分析的示例代码。

## 示例文件

- `analyze_cycling_pose.py`: 完整的骑行姿态分析示例脚本，包括命令行参数处理、结果可视化等

## 如何运行示例

### 基本用法

```bash
# 确保在项目根目录下运行，或者已将项目添加到Python路径
python backend/pose_detection/examples/analyze_cycling_pose.py 你的视频文件路径
```

### 高级选项

```bash
# 指定输出目录
python backend/pose_detection/examples/analyze_cycling_pose.py 视频文件路径 --output_dir 结果保存目录

# 保存分析过程中的关键帧
python backend/pose_detection/examples/analyze_cycling_pose.py 视频文件路径 --save_frames

# 开启调试信息打印
python backend/pose_detection/examples/analyze_cycling_pose.py 视频文件路径 --debug
```

### 示例输出

运行示例后，将在指定的输出目录（默认为`./output`）生成以下文件：

1. `pose_animation.gif`: 骑行姿态动画，展示关键点和骨架
2. `angle_chart.png`: 各关节角度与理想范围的对比图
3. 如果使用了`--save_frames`选项，还会生成各个关键帧的图片文件

## 结果解读

示例程序会输出以下角度测量结果：

- 膝关节角度（最低点和最高点）
- 髋关节角度（最低点和最高点）
- 肩膀角度
- 肘关节角度

在`angle_chart.png`中，蓝色柱状图表示实际测量值，红色虚线表示理想范围（若适用）。

## 注意事项

1. 确保视频中骑行者呈侧面姿态，且清晰可见
2. 视频应包含至少3-5个完整的踏蹬周期
3. 处理高分辨率视频可能需要较长时间，请耐心等待

## 进阶使用

该示例可以扩展用于以下场景：

- 骑行技术分析和改进
- 自行车尺寸调整辅助
- 骑行效率评估
- 伤害风险评估

可以通过修改源代码进一步定制分析过程和结果展示。 