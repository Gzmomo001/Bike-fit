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
- Web框架：
  * Flask==2.0.1
  * Werkzeug==2.0.1
  * Jinja2==3.0.1

- 图像处理：
  * opencv-python==4.5.3.56
  * mediapipe==0.8.7.3
  * Pillow==8.3.2

- 科学计算：
  * numpy==1.21.2
  * scipy==1.7.1

- 实时通信：
  * websockets==10.0
  * python-socketio==5.4.0

- 数据库：
  * SQLAlchemy==1.4.23

完整依赖列表请查看 `backend/requirements.txt`

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
# Linux
xdg-open main.html
# Windows
start main.html
# 或者直接在浏览器中拖拽main.html文件打开
```

3. 前端功能说明
- 首页提供两种模式：图片上传和实时摄像头分析
- 支持JPG、PNG、JPEG格式的图片
- 建议使用清晰的侧面骑行照片以获得最佳分析效果
- 实时分析需要允许浏览器访问摄像头

### 后端启动
1. 创建并激活虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# MacOS/Linux
source venv/bin/activate
```

2. 安装依赖
```bash
cd backend
# 如果requirements.txt不存在，先创建它
pip freeze > requirements.txt
# 安装依赖
pip install -r requirements.txt
```

3. 启动Flask服务器
```bash
# 开发模式启动（推荐）
export FLASK_ENV=development  # MacOS/Linux
set FLASK_ENV=development    # Windows
python main.py

# 或者直接启动
python main.py
```

服务器默认将在 http://localhost:5000 启动

### 验证安装
1. 确认服务启动
- 检查后端服务器是否正常运行（http://localhost:5000）
- 确认终端没有报错信息
- 检查所有依赖是否安装成功

2. 功能测试
- 上传图片测试：
  * 准备一张清晰的侧面骑行照片
  * 点击"上传图片"按钮
  * 等待分析结果（通常需要3-5秒）
  * 查看分析报告和建议

- 实时分析测试：
  * 点击"开启摄像头"按钮
  * 允许浏览器访问摄像头
  * 调整位置使自行车侧面完整显示
  * 查看实时分析结果

3. 常见问题处理
- 如果网页无法加载，检查浏览器控制台是否有错误信息
- 确保后端服务正常运行
- 检查防火墙是否允许本地服务器访问
- 确保Python版本兼容（推荐3.8+）

## 开发模式
如果您想要进行开发：
1. 创建新的git分支
```bash
git checkout -b feature/your-feature-name
```

2. 启用调试模式
```bash
export FLASK_DEBUG=1  # MacOS/Linux
set FLASK_DEBUG=1    # Windows
python main.py
```

3. 实时日志查看
```bash
# 在另一个终端窗口
tail -f backend/app.log
```

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
- 项目地址：https://github.com/Gzmomo001/Bike-fit 

## 后端代码详解
### 项目结构
```
backend/
├── __init__.py          # 包初始化文件
├── app.py              # Flask应用主配置
├── main.py             # 主程序入口
├── model.py            # 核心模型定义
├── preprocessing.py    # 图像预处理模块
├── postprocessing.py   # 结果后处理模块
├── cropping.py         # 图像裁剪模块
├── keypoints.py        # 关键点检测模块
├── unit_test.py        # 单元测试
└── requirements.txt    # 项目依赖
```

### 核心模块说明
#### 1. 主程序入口 (main.py)
- 初始化Flask应用和路由
- 处理HTTP请求和WebSocket连接
- 协调各个模块的工作流程
- 主要API端点：
  * `/upload`: 处理图片上传
  * `/analyze`: 姿态分析
  * `/realtime`: WebSocket实时分析
  * `/report`: 生成分析报告

#### 2. 应用配置 (app.py)
- Flask应用配置
- 数据库连接设置
- 跨域资源共享(CORS)配置
- 日志系统设置
- 错误处理机制

#### 3. 核心模型 (model.py)
- 自行车体态分析核心算法
- 使用MediaPipe进行人体姿态识别
- 计算关键角度和距离
- 生成调整建议
- 主要功能：
  * 姿态识别
  * 角度计算
  * 参数测量
  * 建议生成

#### 4. 图像处理
##### 预处理 (preprocessing.py)
- 图像格式转换
- 尺寸调整
- 色彩空间转换
- 图像增强
- 噪声处理

##### 后处理 (postprocessing.py)
- 分析结果优化
- 数据过滤和平滑
- 生成可视化结果
- 报告格式化
- 建议优化

##### 图像裁剪 (cropping.py)
- 自动检测ROI(感兴趣区域)
- 智能裁剪
- 边界处理
- 图像对齐

#### 5. 关键点检测 (keypoints.py)
- 人体关键点识别
- 自行车关键部件识别
- 坐标转换
- 特征点跟踪

#### 6. 测试模块 (unit_test.py)
- 单元测试用例
- 集成测试
- 性能测试
- 边界条件测试

### 数据流程
1. 图像输入
   - 上传图片或视频流输入
   - 格式验证和初始化

2. 预处理阶段
   - 图像标准化
   - 质量优化
   - 尺寸调整

3. 核心处理
   - 人体姿态识别
   - 自行车部件识别
   - 参数计算
   - 姿态分析

4. 后处理阶段
   - 结果优化
   - 数据验证
   - 建议生成
   - 报告整理

5. 输出结果
   - JSON格式数据
   - 可视化结果
   - 调整建议
   - 专业报告

### API接口说明
#### 1. 图片分析接口
```http
POST /api/analyze
Content-Type: multipart/form-data

参数：
- image: 图片文件
- options: 分析选项（可选）

返回：
{
    "status": "success",
    "data": {
        "angles": {...},
        "distances": {...},
        "suggestions": [...]
    }
}
```

#### 2. 实时分析接口
```websocket
WebSocket: ws://localhost:5000/realtime

消息格式：
{
    "type": "frame",
    "data": "base64编码的图像数据"
}

返回：
{
    "type": "analysis",
    "data": {
        "pose": {...},
        "suggestions": [...]
    }
}
```

### 错误处理
- 输入验证错误
- 处理超时
- 资源不可用
- 格式不支持
- 系统错误

每种错误都有对应的错误代码和处理机制，确保系统稳定性和用户体验。

### 性能优化
1. 图像处理优化
   - 使用OpenCV优化的算法
   - 图像缓存机制
   - 并行处理

2. 实时分析优化
   - 帧率控制
   - 数据压缩
   - 异步处理

3. 资源管理
   - 内存使用优化
   - CPU负载均衡
   - 并发连接管理 