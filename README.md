# Bike-fit 自行车体态拟合系统

## 项目简介
Bike-fit是一个创新的自行车体态拟合系统，旨在帮助骑行者获得最佳的骑行姿势和自行车设置。通过先进的计算机视觉技术和专业的骑行知识，为用户提供个性化的自行车调整建议。

## 系统架构
项目采用前后端分离的架构设计：
- 前端 (`frontend/`): 用户界面和交互层
- 后端 (`backend/`): 业务逻辑和数据处理层
- 演示模块 (`demo/`): 功能演示和测试用例
- 数据模型 (`thunder model/`): 系统数据模型设计

## 主要功能
- 骑行姿势分析
- 自行车参数测量
- 个性化调整建议
- 用户数据管理
- 专业报告生成

## 技术栈
### 前端
- HTML5 + CSS3
- JavaScript (ES6+)
- Bootstrap 用于响应式布局
- WebSocket 用于实时通信
- 文件上传和图像处理功能

### 后端
- Python 3.8+
- Flask Web框架
- OpenCV 用于图像处理
- MediaPipe 用于人体姿态识别
- NumPy 用于数学计算
- SQLite 数据库
- WebSocket 用于实时通信

## 环境要求
### 系统要求
- Python 3.8 或更高版本
- pip 包管理器
- 现代浏览器（Chrome、Firefox、Safari等）
- 摄像头（用于实时姿态分析）

### 依赖安装
1. 克隆项目
```bash
git clone [项目地址]
cd Bike-fit
```

2. 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

主要依赖包括：
- flask==2.0.1
- opencv-python==4.5.3.56
- mediapipe==0.8.7.3
- numpy==1.21.2
- websockets==10.0
- pillow==8.3.2

## 快速启动
### 前端启动
1. 进入前端目录
```bash
cd frontend
```

2. 使用浏览器打开main.html
```bash
# MacOS
open main.html
# 或者直接在浏览器中打开main.html文件
```

### 后端启动
1. 进入后端目录
```bash
cd backend
```

2. 启动Flask服务器
```bash
python main.py
```

服务器默认将在 http://localhost:5000 启动

### 验证安装
1. 打开浏览器访问 http://localhost:5000
2. 上传骑行图片或开启摄像头进行实时分析
3. 查看分析结果和调整建议

## 常见问题
1. 如果遇到依赖安装问题，请确保您的Python版本兼容
2. 确保摄像头权限已经开启
3. 如果出现图像处理错误，请检查图片格式是否支持

## 开发团队
[待补充开发团队信息]

## 许可证
[待补充许可证信息]

## 联系方式
如有任何问题或建议，请通过以下方式联系我们：
- 邮箱：[待补充]
- 项目地址：[待补充] 