# Bike-fitting

## 项目简介

Bike-fitting 是一个创新的自行车适配分析系统，旨在帮助骑行者获得最佳的骑行姿势和车架设置。本项目结合了计算机视觉和人工智能技术，通过分析骑行者的骑行姿态，提供专业的自行车调整建议。

主要功能包括：
- 实时姿态检测和分析
- 智能化的自行车设置建议
- 个性化的骑行姿势优化方案
- 详细的数据分析报告

本系统适用于：
- 专业的自行车店和适配中心
- 个人骑行爱好者
- 运动训练机构
- 自行车运动教练

通过使用本系统，用户可以：
- 提高骑行效率和舒适度
- 预防骑行相关的运动损伤
- 优化骑行表现
- 获得专业的自行车调校建议
  
### 项目demo演示

<video width="100%" controls>
  <source src="https://user-images.githubusercontent.com/your-user-id/bike-fitting/assets/demo.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>

演示视频展示了系统的主要功能：
- 实时姿态检测与分析
- 自动参数调整建议
- 数据可视化展示
- 报告生成功能

## 目录
- [Bike-fitting](#bike-fitting)
  - [项目简介](#项目简介)
    - [项目demo演示](#项目demo演示)
  - [目录](#目录)
  - [主要特性](#主要特性)
    - [智能姿态分析](#智能姿态分析)
    - [专业调整建议](#专业调整建议)
    - [实时监测系统](#实时监测系统)
    - [数据分析报告](#数据分析报告)
    - [用户友好界面](#用户友好界面)
    - [安全性能](#安全性能)
  - [环境要求](#环境要求)
    - [基础要求](#基础要求)
    - [macOS 环境](#macos-环境)
    - [Windows/Linux CUDA 环境](#windowslinux-cuda-环境)
    - [主要依赖](#主要依赖)
    - [网络要求](#网络要求)
  - [快速开始](#快速开始)
    - [安装步骤](#安装步骤)
    - [依赖说明](#依赖说明)
      - [核心依赖包](#核心依赖包)
      - [可选依赖](#可选依赖)
  - [使用指南](#使用指南)
    - [配置说明](#配置说明)
      - [系统配置](#系统配置)
      - [用户设置](#用户设置)
    - [基础用法](#基础用法)
    - [高级功能](#高级功能)
      - [专业模式](#专业模式)
      - [数据导出](#数据导出)
      - [自定义分析](#自定义分析)
      - [团队功能](#团队功能)
    - [注意事项](#注意事项)
  - [开发相关](#开发相关)
    - [项目结构](#项目结构)
      - [目录说明](#目录说明)
    - [测试方法](#测试方法)
    - [构建指南](#构建指南)
  - [贡献指南](#贡献指南)
  - [许可证](#许可证)
  - [致谢](#致谢)
  - [联系方式](#联系方式)

## 主要特性

### 智能姿态分析
- 基于 MediaPipe 的实时人体姿态识别
- 精确的关键点检测和跟踪
- 多角度姿势分析和评估
- 自动识别不良骑行姿势

### 专业调整建议
- 根据骑行者身体数据生成个性化建议
- 车架几何参数智能调整方案
- 详细的调整步骤指导
- 可视化的参数展示

### 实时监测系统
- 支持实时视频流分析
- 即时反馈骑行姿态
- 动态姿势纠正提示
- 连续数据记录和跟踪

### 数据分析报告
- 生成专业的适配报告
- 详细的测量数据统计
- 历史数据对比分析
- 可导出PDF格式报告

### 用户友好界面
- 直观的操作流程
- 响应式网页设计
- 多平台兼容支持
- 简洁清晰的数据展示

### 安全性能
- 用户数据加密存储
- 隐私信息保护
- 安全的数据传输
- 完整的备份机制

## 环境要求

### 基础要求
- Python 3.10 或更高版本
- Conda 包管理器

### macOS 环境
- macOS 操作系统（支持 Apple Silicon/Intel）
- 8GB 以上内存
- 10GB 以上可用磁盘空间

### Windows/Linux CUDA 环境
- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit 11.8
- 16GB 以上内存推荐
- 15GB 以上可用磁盘空间

### 主要依赖
- PyTorch 2.5+ (macOS) / 2.1+ (CUDA)
- TensorFlow 2.16+ (macOS) / 2.15+ (CUDA)
- OpenCV 4.11+ (macOS) / 4.8+ (CUDA)
- MediaPipe 0.10+
- FastAPI 0.115+
- Flask 3.0+

### 网络要求
- 稳定的互联网连接
- 如使用在线 API，需要相应的 API 密钥

## 快速开始

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/bike-fitting.git
cd bike-fitting
```

2. 创建并激活 Conda 环境

对于 macOS 用户：
```bash
conda env create -f environment-mac.yml
conda activate bike-fit
```

对于 CUDA 环境用户：
```bash
conda env create -f environment-cuda.yml
conda activate bike-fit-cuda
```

3. 安装项目依赖（可选）
```bash
cd backend
pip install -r requirements.txt
cd ../frontend
npm install
```

4. 配置环境变量（eg. api key）
```bash
cp .env.example .env
```
编辑 `.env` 文件，设置必要的环境变量。

5. 启动服务
```bash
python backend/server.py
```

6. 访问应用
打开浏览器访问 `demo.kittybob.com` 即可使用系统

### 依赖说明

#### 核心依赖包
- **PyTorch**: 深度学习框架，用于姿态检测模型
- **MediaPipe**: Google的机器学习解决方案库，用于人体姿态识别
- **OpenCV**: 计算机视觉库，用于图像处理
- **FastAPI**: 现代化的Web框架，用于后端API开发
- **React**: 前端框架，用于用户界面开发

#### 可选依赖
- **CUDA Toolkit**: 用于GPU加速（仅NVIDIA显卡用户需要）
- **TensorRT**: 用于模型推理优化（可选）

## 使用指南

### 配置说明

#### 系统配置

1. 环境要求
   - 光线充足均匀，避免逆光
   - 背景尽量简单，避免复杂背景干扰
   - 确保拍摄区域空间充足

2. 网络设置
   - 确保网络连接稳定
   - 如使用在线API，请在 `.env` 文件中正确配置API密钥

#### 用户设置
- 填写基本信息（身高、体重、骑行经验等）
- 设置偏好（计量单位、语言等）
- 配置通知提醒（可选）

### 基础用法

1.上传视频

2.等待分析完成

3.查看分析结果

### 高级功能

#### 专业模式
- 详细的角度测量
- 自定义参数设置
- 高级数据分析
- 历史数据对比

#### 数据导出
- 导出PDF报告
- 数据图表导出
- 视频记录保存
- 参数记录导出

#### 自定义分析
- 设置个性化标准
- 自定义评估指标
- 创建专属模板
- 批量数据处理

#### 团队功能
- 多用户管理
- 数据共享设置
- 团队报告生成
- 权限管理

### 注意事项
1. 安全提示
   - 确保自行车固定稳妥
   - 调整时注意安全
   - 遵医嘱进行调整

2. 使用建议
   - 定期进行检测
   - 保存调整记录
   - 循序渐进调整
   - 及时反馈问题

3. 维护保养
   - 定期检查设备
   - 及时更新系统
   - 备份重要数据

## 开发相关

### 项目结构

```
bike-fitting/
├── backend/                 # 后端服务
│   ├── server/             # 服务器核心代码
│   │   ├── __init__.py
│   │   ├── __main__.py    # 服务入口
│   │   └── config.py      # 服务配置
│   ├── pose_detection/     # 姿态检测模块
│   │   ├── preprocessing.py    # 数据预处理
│   │   ├── pose_analyzer.py    # 姿态分析
│   │   ├── postprocessing.py   # 后处理
│   │   └── cropping.py         # 图像裁剪
│   ├── api_model.py        # API模型接口
│   ├── local_model.py      # 本地模型实现
│   ├── bike_fit_advisor.py # 自行车适配建议
│   ├── app.py             # FastAPI应用
│   └── requirements.txt    # Python依赖
│
├── frontend/               # 前端应用
│   ├── src/               # 源代码
│   │   ├── components/    # React组件
│   │   ├── pages/        # 页面组件
│   │   ├── utils/        # 工具函数
│   │   └── styles/       # 样式文件
│   ├── public/           # 静态资源
│   └── package.json      # 前端依赖
│
├── data/                  # 数据目录
│   ├── models/           # 预训练模型
│   └── examples/         # 示例数据
│
├── docs/                  # 文档
│   ├── api/              # API文档
│   ├── deployment/       # 部署指南
│   └── development/      # 开发指南
│
├── tests/                # 测试
│   ├── unit/            # 单元测试
│   └── integration/     # 集成测试
│
├── scripts/              # 工具脚本
│   ├── setup.sh         # 环境配置脚本
│   └── deploy.sh        # 部署脚本
│
├── environment-mac.yml   # macOS环境配置
├── environment-cuda.yml  # CUDA环境配置
├── .env.example         # 环境变量示例
├── README.md            # 项目说明
└── LICENSE              # 许可证
```

#### 目录说明

1. **后端 (backend/)**
   - `server/`: 核心服务器代码
   - `pose_detection/`: 姿态检测和分析模块
   - `api_model.py`: 外部API调用接口
   - `local_model.py`: 本地模型实现
   - `bike_fit_advisor.py`: 自行车适配建议生成器

2. **前端 (frontend/)**
   - `src/components/`: 可复用的React组件
   - `src/pages/`: 页面级组件
   - `src/utils/`: 工具函数和辅助方法
   - `src/styles/`: 样式文件和主题

3. **数据 (data/)**
   - `models/`: 预训练的机器学习模型
   - `examples/`: 示例数据和测试用例

4. **文档 (docs/)**
   - `api/`: API接口文档
   - `deployment/`: 部署和运维指南
   - `development/`: 开发者指南

5. **测试 (tests/)**
   - `unit/`: 单元测试用例
   - `integration/`: 集成测试用例

6. **工具脚本 (scripts/)**
   - 包含各种自动化脚本
   - 环境配置和部署工具

### 测试方法

### 构建指南

## 贡献指南

## 许可证

## 致谢

## 联系方式





