# 自行车骑行姿态检测包

## 简介

该姿态检测包是一个专为自行车骑行姿态评估设计的工具，基于人工智能视觉技术，能够从视频中自动识别骑行者的身体姿态，分析关键关节角度，帮助用户优化骑行姿势，提高效率并减少伤害风险。

<!-- 
图片引用说明：
1. 使用PoseAnalyzer处理示例视频后，将生成的GIF放在此处
2. 示例图片路径：../docs/images/pose_detection_example.gif
-->
![姿态检测示例（占位图）](../docs/images/pose_detection_example.gif)

> 注：运行示例代码后，将会在`docs/images`目录中生成可视化示例图片。请参考`docs/images/README.md`了解如何生成示例图片。

## 核心功能

- **智能关键点识别**：自动检测人体17个关键点位置
- **骑行姿态分析**：计算并评估关键关节角度
  - 膝关节角度（踏板最高点和最低点）
  - 髋关节角度（踏板最高点和最低点）
  - 肩膀角度
  - 肘关节角度
- **可视化结果**：生成带骨架标注的视频，直观展示骑行姿态

## 使用方法

### 快速上手

```python
from pose_detection import PoseAnalyzer

# 初始化分析器
analyzer = PoseAnalyzer()

# 分析骑行视频
results, frames, gif_path = analyzer.pose_analyzer("骑行视频.mp4")

# 查看分析结果
print(f"膝关节角度（最低点）: {results['knee_angle_lowest']}°")
print(f"膝关节角度（最高点）: {results['knee_angle_highest']}°")
print(f"髋关节角度（最低点）: {results['hip_angle_lowest']}°")
print(f"髋关节角度（最高点）: {results['hip_angle_highest']}°")
print(f"肩膀角度: {results['shoulder_angle']}°")
print(f"肘关节角度: {results['elbow_angle']}°")
```

### 关节角度说明

| 角度类型 | 说明 | 理想范围 |
|---------|------|---------|
| 膝关节最低点 | 踏板最低位置时大腿与小腿角度 | 150-155° |
| 膝关节最高点 | 踏板最高位置时大腿与小腿角度 | 65-70° |
| 髋关节角度 | 躯干与大腿的夹角 | 视骑行风格而定 |
| 肩膀角度 | 躯干与上臂的夹角 | 约90° |
| 肘关节角度 | 上臂与前臂的夹角 | 160-170° |

### 关键点索引说明

MoveNet模型检测的17个关键点及其在骑行姿态分析中的应用：

```mermaid
graph TD
    subgraph 人体关键点索引
    0[0: 鼻子]
    1[1: 左眼]
    2[2: 右眼]
    3[3: 左耳]
    4[4: 右耳]
    5[5: 左肩] --- 11[11: 左髋]
    6[6: 右肩] --- 12[12: 右髋]
    7[7: 左肘]
    8[8: 右肘]
    9[9: 左手腕]
    10[10: 右手腕]
    11 --- 13[13: 左膝]
    12 --- 14[14: 右膝]
    13 --- 15[15: 左踝]
    14 --- 16[16: 右踝]
    end
    
    subgraph 骑行姿态分析关键点
    style 11 fill:#f9f,stroke:#333,stroke-width:2px
    style 13 fill:#f9f,stroke:#333,stroke-width:2px
    style 15 fill:#f9f,stroke:#333,stroke-width:2px
    style 5 fill:#bbf,stroke:#333,stroke-width:2px
    style 7 fill:#bbf,stroke:#333,stroke-width:2px
    style 9 fill:#bbf,stroke:#333,stroke-width:2px
    
    11 -.->|髋关节| H1[髋部角度计算]
    13 -.->|膝关节| K1[膝部角度计算]
    15 -.->|踝关节| K1
    5 -.->|肩关节| S1[肩部角度计算]
    7 -.->|肘关节| E1[肘部角度计算]
    9 -.->|手腕| E1
    end
```

系统会根据相机拍摄角度自动判断使用左侧或右侧的关键点进行计算。例如，如果骑行者右侧面对相机，则使用右侧关键点（6, 8, 10, 12, 14, 16）进行分析；反之亦然。

## 技术特点

- **高精度姿态检测**：采用Google MoveNet-Thunder模型
- **智能关键帧识别**：自动定位踏板最高点和最低点
- **自适应视频处理**：支持不同尺寸和方向的视频输入
- **考虑宽高比的角度计算**：确保测量精度

## 算法流程

下面的流程图展示了骑行姿态分析的完整处理过程：

```mermaid
flowchart TD
    A[输入骑行视频] --> B[视频预处理]
    B --> C[加载MoveNet模型]
    C --> D[逐帧姿态检测]
    D --> E[获取关键点坐标]
    
    subgraph 关键帧检测
    E --> F1[识别相机朝向]
    F1 --> F2[计算踏板最高点]
    F1 --> F3[计算踏板最低点]
    end
    
    subgraph 角度计算
    F2 --> G1[计算膝关节角度]
    F3 --> G1
    F2 --> G2[计算髋关节角度]
    F3 --> G2
    E --> G3[计算肩膀角度]
    E --> G4[计算肘关节角度]
    end
    
    subgraph 结果处理
    G1 --> H1[平均值计算]
    G2 --> H1
    G3 --> H1
    G4 --> H1
    H1 --> H2[异常值过滤]
    H2 --> H3[生成分析结果]
    end
    
    subgraph 可视化
    E --> I1[生成骨架标注]
    I1 --> I2[合成GIF动画]
    H3 --> I3[生成角度图表]
    end
    
    H3 --> J[返回分析结果]
    I2 --> J
    I3 --> J
```

### 数据流程图（输入输出）

下面的流程图详细展示了每个处理阶段的输入和输出数据：

```mermaid
flowchart TD
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:1px
    classDef process fill:#fff9c4,stroke:#f57f17,stroke-width:1px
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    
    %% 1. 视频预处理
    VideoFile[/"骑行视频文件\n(MP4, AVI, MOV)"/]:::input --> PreProcess["视频预处理\n- 解码视频帧\n- 调整大小为正方形\n- 归一化像素值"]:::process
    PreProcess --> Frames["帧序列张量\n[帧数, 256, 256, 3]"]:::output
    PreProcess --> OrigFrames["原始解码帧"]:::output
    
    %% 2. 姿态检测
    Frames --> MoveNet["MoveNet-Thunder\n姿态检测模型"]:::process
    MoveNet --> Keypoints["关键点坐标数组\n[帧数, 17, 3]\n(y坐标, x坐标, 置信度)"]:::output
    
    %% 3. 关键帧检测
    Keypoints --> KeyFrameDetection["关键帧检测\n- 识别相机朝向\n- 计算踝关节轨迹"]:::process
    KeyFrameDetection --> Direction["朝向标识\n(left/right)"]:::output
    KeyFrameDetection --> HighestPts["最高点帧索引"]:::output
    KeyFrameDetection --> LowestPts["最低点帧索引"]:::output
    KeyFrameDetection --> FrontIndices["前侧关键点索引"]:::output
    
    %% 4. 角度计算
    Keypoints --> AngleCalc["角度计算\n- 考虑视频宽高比\n- 矢量计算夹角"]:::process
    Direction --> AngleCalc
    HighestPts --> AngleCalc
    LowestPts --> AngleCalc
    FrontIndices --> AngleCalc
    AngleCalc --> KneeAngles["膝关节角度序列"]:::output
    AngleCalc --> HipAngles["髋关节角度序列"]:::output
    AngleCalc --> ShoulderAngles["肩膀角度序列"]:::output
    AngleCalc --> ElbowAngles["肘关节角度序列"]:::output
    
    %% 5. 结果处理
    KneeAngles --> ResultProcessing["结果处理\n- 中值滤波平滑\n- 异常值过滤\n- 关键角度提取"]:::process
    HipAngles --> ResultProcessing
    ShoulderAngles --> ResultProcessing
    ElbowAngles --> ResultProcessing
    ResultProcessing --> Results["角度结果字典\n{\n  knee_angle_lowest,\n  knee_angle_highest,\n  hip_angle_lowest,\n  hip_angle_highest,\n  shoulder_angle,\n  elbow_angle\n}"]:::output
    
    %% 6. 可视化
    OrigFrames --> Visualization["可视化处理\n- 骨架绘制\n- GIF生成\n- 角度图表"]:::process
    Keypoints --> Visualization
    FrontIndices --> Visualization
    Results --> Visualization
    Visualization --> AnnotatedFrames["带骨架标注的帧"]:::output
    Visualization --> GifAnimation["骑行姿态GIF动画"]:::output
    Visualization --> AngleChart["角度对比图表"]:::output
    
    %% 7. 最终输出
    Results --> FinalOutput["最终返回值\n(results, frames, gif_path)"]:::output
    AnnotatedFrames --> FinalOutput
    GifAnimation --> FinalOutput
```

## 录制视频建议

为获得最佳检测效果，请遵循以下录制建议：

- **拍摄角度**：侧面视角，确保能清晰看到骑行者的侧面轮廓
- **光线条件**：充足自然光，避免逆光
- **服装选择**：穿着与背景对比明显的服装
- **视频质量**：分辨率不低于720p
- **骑行姿势**：保持自然骑行状态，完成至少3-5个完整踏蹬周期

## 角度测量示意图

以下图表展示了系统计算的关键角度测量方式：

```mermaid
graph TD
    subgraph 最低点膝关节角度
    direction LR
    A1[髋关节] --- B1[膝关节]
    B1 --- C1[踝关节]
    
    style B1 fill:#f66,stroke:#333,stroke-width:2px
    
    %% 添加角度弧线
    D1(("150-155°"))
    end
    
    subgraph 最高点膝关节角度
    direction LR
    A2[髋关节] --- B2[膝关节]
    B2 --- C2[踝关节]
    
    style B2 fill:#f66,stroke:#333,stroke-width:2px
    
    %% 添加角度弧线
    D2(("65-70°"))
    end
    
    subgraph 髋关节角度
    direction LR
    A3[肩关节] --- B3[髋关节]
    B3 --- C3[膝关节]
    
    style B3 fill:#f66,stroke:#333,stroke-width:2px
    end
    
    subgraph 肩膀角度
    direction LR
    A4[肘关节] --- B4[肩关节]
    B4 --- C4[髋关节]
    
    style B4 fill:#f66,stroke:#333,stroke-width:2px
    
    %% 添加角度弧线
    D4(("约90°"))
    end
    
    subgraph 肘关节角度
    direction LR
    A5[肩关节] --- B5[肘关节]
    B5 --- C5[手腕]
    
    style B5 fill:#f66,stroke:#333,stroke-width:2px
    
    %% 添加角度弧线
    D5(("160-170°"))
    end
```

### 宽高比校正说明

由于视频在预处理时会被调整为正方形（MoveNet要求），这会导致原始视频的宽高比发生变化，从而影响角度计算的准确性。为解决这个问题，系统会：

1. 在预处理阶段检测原始视频的宽高比（`original_aspect_ratio`）
2. 在角度计算时应用宽高比校正：

```mermaid
graph TB
    subgraph 宽高比校正过程
    A[获取原始坐标] --> B["坐标转换\n[y, x] → [x, y]"]
    B --> C["应用宽高比校正\nx' = x * original_aspect_ratio"]
    C --> D["计算矢量\nv = [x', y]"]
    D --> E["计算夹角\nangle = arccos(dot(v1, v2) / |v1|·|v2|)"]
    end
```

这种校正方法确保了即使原始视频是非正方形的（如横屏或竖屏视频），计算的关节角度也能准确反映真实的身体姿态。

## 常见问题

### Q: 为什么我的视频无法正确检测关键点？
A: 请检查视频拍摄角度是否为清晰的侧面视角，光线是否充足，以及骑行者是否清晰可见。

### Q: 测量的角度与专业自行车测量有差异，为什么？
A: 由于视频角度、衣物影响和AI模型的局限性，可能会有5-10°的误差，建议将结果作为参考，配合专业测量使用。

### Q: 如何判断我的骑行姿势是否合适？
A: 请参考理想角度范围，同时结合个人感受和专业建议。角度只是参考指标，最佳骑行姿势应适合个人身体条件和骑行风格。

### Q: 是否支持不同的骑行风格（如公路车、山地车）？
A: 是的，工具可用于各种骑行风格，但理想角度范围可能因骑行类型而异。

## 模块结构

```
pose_detection/
├── pose_analyzer.py     # 核心姿态分析类
├── model.py             # 模型加载和推理
├── preprocessing.py     # 视频预处理
├── postprocessing.py    # 数据后处理
├── cropping.py          # 图像裁剪功能
└── keypoints.py         # 关键点定义
```

## 安装依赖

```bash
# 使用conda创建环境
conda env create -f env-mac.yml
conda activate bike-fit

# 或使用pip安装核心依赖
pip install tensorflow tensorflow-hub numpy opencv-python matplotlib imageio scipy kagglehub
```

## 开发团队

该工具由自行车科学研究小组开发，致力于将人工智能技术应用于自行车运动科学研究。

## 许可证

MIT License 