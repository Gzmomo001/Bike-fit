# Bike-fit

基于计算机视觉的智能自行车适配系统

## 项目结构

```
├── backend/          # 后端服务
├── frontend/         # 前端应用
├── VectorStore/      # 向量存储
├── uploads/          # 上传文件存储
├── thunder model/    # 模型文件
└── environment-*.yml # 环境配置文件
```

## 技术栈

### 后端技术

#### 核心框架与运行时
- Python 3.10+
- FastAPI (异步Web框架)
- ASGI Server (uvicorn)
- WebSocket (实时通信)

#### 计算机视觉处理
- MoveNet (人体姿态估计)
- OpenCV (图像预处理)
- MediaPipe (辅助姿态检测)
- NumPy (数值计算)

#### AI 模型与向量处理
- LangChain (AI应用框架)
- OpenAI API (嵌入模型)
- FAISS (向量数据库)
- PyTorch (深度学习支持)

#### 数据处理与存储
- Pydantic (数据验证)
- Redis (缓存层)
- MongoDB (文档存储)
- Milvus/Pinecone (向量存储)

#### 工具与优化
- asyncio (异步IO)
- multiprocessing (并行处理)
- pytest (单元测试)
- logging (日志管理)

### 后端处理流程

#### 1. 视频处理流程 (`backend/pose_detection/preprocessing.py`)
```python
def pre_process_video(file: str | bytes) -> tuple:
    """视频预处理流程"""
    # 1. 视频质量压缩
    clip = reduce_video_quality(
        video_path=file,
        max_pixels=256,
        max_fps=30,
        max_duration=10
    )
    
    # 2. 视频帧处理
    frames = []
    tensors = []
    for frame in clip:
        # 调整帧大小和颜色空间
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为张量
        tensor = tf.convert_to_tensor(frame)
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.cast(tensor, dtype=tf.int32)
        
        frames.append(frame)
        tensors.append(tensor)
    
    return frames, tf.concat(tensors, axis=0)
```
#### 2. 模型加载与推理
```python
def get_keypoints_from_video(video_tensor, model, input_size):
    """
    从视频中提取关键点。

    参数：
    - video_tensor：一个包含视频数据的张量，其shape为(num_frames, height, width, channels)。
    - model：用于关键点检测的模型。
    - input_size：模型输入的尺寸。

    返回值：
    - all_keypoints_with_scores：一个列表，包含每一帧的关键点及其分数。
    """

    # 获取视频张量的形状信息
    num_frames, video_height, video_width, _ = video_tensor.shape

    # 初始化一个列表来存储所有帧的关键点及其分数
    all_keypoints_with_scores = []

    # 初始化裁剪区域，覆盖整个视频帧
    crop_region = init_crop_region(video_height, video_width)

    # 遍历每一帧
    for frame_idx in range(num_frames):
        # 对当前帧运行推理，获取关键点及其分数，并添加到列表中
        all_keypoints_with_scores.append(
            _run_inference(
                model,
                video_tensor[frame_idx, :, :, :],
                crop_region,
                crop_size=[input_size, input_size],
            )
        )

        # 根据当前帧的关键点，确定下一帧的裁剪区域。这个操作可以使模型的越来越聚焦，提高运算效率
        crop_region = determine_crop_region(
            all_keypoints_with_scores[frame_idx], video_height, video_width
        )

    # 返回所有帧的关键点及其分数
    return all_keypoints_with_scores
```

#### 3. 姿态检测流程 (`backend/pose_detection/pose_analyzer.py`)
```python
def get_pose(all_keypoints):
    """姿态分析主函数"""
    # 1. 确定拍摄朝向
    facing_direction = find_camera_facing_side(all_keypoints[0])
    front_indices = get_front_keypoint_indices(facing_direction)
    hip_knee_ankle_indices = front_indices[:4]

    # 2. 获取关键帧索引
    lowest_pedal_point_indices = get_lowest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )
    highest_pedal_point_indices = get_highest_pedal_frames(
        all_keypoints, hip_knee_ankle_indices
    )

    # 3. 计算各个角度
    knee_angle_lowest = get_knee_angle_at_lowest_pedal_points_avg(
        all_keypoints, hip_knee_ankle_indices, lowest_pedal_point_indices
    )
    knee_angle_highest = get_knee_angle_at_highest_pedal_points_avg(
        all_keypoints, hip_knee_ankle_indices, highest_pedal_point_indices
    )
    shoulder_angle = get_shoulder_angle_avg(all_keypoints, front_indices)
    elbow_angle = get_elbow_angle_avg(all_keypoints, front_indices)
    hip_angle_lowest = get_hip_angle_at_lowest_pedal_points_avg(
        all_keypoints, front_indices, lowest_pedal_point_indices
    )
    hip_angle_highest = get_hip_angle_at_highest_pedal_points_avg(
        all_keypoints, front_indices, highest_pedal_point_indices
    )

    # 4. 返回测量结果
    measurements = {
        'knee_angle_lowest': knee_angle_lowest,
        'knee_angle_highest': knee_angle_highest,
        'shoulder_angle': shoulder_angle,
        'elbow_angle': elbow_angle,
        'hip_angle_lowest': hip_angle_lowest,
        'hip_angle_highest': hip_angle_highest,
    }

    return measurements
```

#### 4. 生成prompt(`backend/bike_fit_advisor.py`)
```python
def generate_prompt(self, measurements):
        """分析姿态并提供建议"""
        prompt = f"""
        作为一位专业的自行车适配专家，请根据以下骑行姿势的数据进行分析：
        - 测量数据：{measurements}

        请基于这些信息提供专业建议，包括但不限于：
        1. 骑行姿势的当前状态评估。
        2. 任何可能存在的问题及其对骑行效率或骑手健康的影响。
        3. 改善建议，比如调整座椅高度、把手位置等具体措施。
        4. 如果适用的话，推荐一些有助于改善骑行体验的产品或训练方法。

        请确保你的建议既科学又实用，能够帮助骑手提高舒适度和骑行表现。
        """
        prompt_en = f"""
        As a professional bicycle fitter, please make an analysis based on the following riding position data: 
        - measurements: {measurements} 
        Please provide professional advice based on this information, including but not limited to: 
        1. Evaluation of the current state of riding posture. 
        2. Any possible problems and their impact on riding efficiency or rider health. 
        3. Suggestions for improvement, such as adjusting seat height, handle position and other specific measures. 
        4. Recommend products or training methods that will help improve the riding experience, if applicable. 
        Make sure your advice is both scientific and practical to help the rider improve comfort and riding performance.
        """
        return prompt
```

#### 5. 智能分析流程 (`backend/local_rag/chat.py`)
```python

def get_model_response(multi_modal_input,history,model,temperature,max_tokens,history_round,db_name,similarity_threshold,chunk_cnt):
    """
    基于本地知识库的多模态对话函数，支持文本和文件输入，使用检索增强生成（RAG）技术进行回答。

    参数:
        multi_modal_input (dict): 多模态输入，包含以下键：
            - text (str): 用户输入的文本
            - files (list): 用户上传的文件列表
        history (list): 对话历史记录，每个元素为 [user_input, assistant_response]
        model (str): 使用的大语言模型名称
        temperature (float): 模型输出的随机性，范围 0-1，越大越随机
        max_tokens (int): 模型回答的最大标记数
        history_round (int): 参考的历史对话轮数
        db_name (str): 知识库名称
        similarity_threshold (float): 相似度阈值，只使用相似度高于此阈值的知识片段
        chunk_cnt (int): 检索的知识片段数量

    返回:
        generator: 生成器对象，每次返回 (history, chunk_show)
            - history: 更新后的对话历史
            - chunk_show: 当前生成的回答片段

    工作流程:
        1. 处理用户输入，如有新文件则创建临时知识库
        2. 从知识库检索相关内容
        3. 使用DashScope进行重排序（rerank）
        4. 构建提示模板
        5. 调用大语言模型生成回答
        6. 流式返回生成的回答
    """
    # 获取用户最新的输入
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']

    # 处理临时知识库
    if os.path.exists(os.path.join("File",TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME

    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    
    try:
        # 初始化重排序器和加载知识库索引
        dashscope_rerank = DashScopeRerank(top_n=chunk_cnt,return_documents=True)
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH,db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")

        # 设置检索器并获取相关文档
        retriever_engine = index.as_retriever(
            similarity_top_k=20,
        )
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")

        # 尝试对检索结果进行重排序
        try:
            results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
            print(f"rerank成功，重排后的chunk为：{results}")
        except:
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")

        # 构建知识片段文本
        chunk_text = ""
        chunk_show = ""
        for i in range(len(results)):
            if results[i].score >= similarity_threshold:
                chunk_text = chunk_text + f"## {i+1}:\n {results[i].text}\n"
                chunk_show = chunk_show + f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score,2)}\n"
        print(f"已获取chunk：{chunk_text}")

        # 构建最终的提示模板
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt
        chunk_show = ""

    # 初始化对话历史
    history[-1][-1] = ""

    # 配置OpenAI客户端
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )                

    # 构建对话消息列表
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []
    history_round = min(len(history),history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages

    # 调用模型生成回答
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
        )

    # 流式返回生成的回答
    assistant_response = ""
    for chunk in completion:
        chunk_show =  chunk.choices[0].delta.content
        assistant_response += chunk_show
        history[-1][-1] = assistant_response
        yield history,chunk_show
```
#### 6. API接口实现 (`backend/server/__main__.py`)
```python
def analyze_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    result = pose_analyzer.pose_analyzer(video_bytes)
    def generate_streaming_response():
        yield json.dumps({"type": "info", "message": result}) + "\n"
        if if_useRAG:
            print("Using RAG")
            # Convert measurements to text format for RAG
            measurement_text = bike_advisor.generate_prompt(result)
            for chunk in bike_advisor.get_streaming_advice_based_on_rag(measurement_text):
                yield chunk
        else:
            pass


            
    return StreamingResponse(generate_streaming_response(), media_type="application/json")
```

### 前端技术
#### 核心框架
- React 18.0+ (前端核心框架)
- TypeScript 5.0+ (类型系统)
- Next.js 14+ (SSR框架，可选)

#### 状态管理
- Redux Toolkit (全局状态管理)
- React Query (服务端状态管理)
- Zustand (轻量级状态管理)

#### UI 框架和组件
- Ant Design (UI组件库)
- TailwindCSS (原子化CSS框架)
- Styled-components (CSS-in-JS方案)
- Material-UI (补充UI组件)
- React Icons (图标库)

#### 数据可视化
- ECharts (图表可视化)
- D3.js (自定义数据可视化)
- Three.js (3D渲染)
- React-Three-Fiber (React 3D渲染)

#### 开发工具
- Vite (构建工具)
- ESLint (代码检查)
- Prettier (代码格式化)
- Husky (Git Hooks)
- Jest (单元测试)
- React Testing Library (组件测试)
- Cypress (E2E测试)

#### 功能模块
- Axios (HTTP客户端)
- React Router (路由管理)
- React Hook Form (表单处理)
- Yup/Zod (数据验证)
- i18next (国际化)
- Day.js (时间处理)

#### WebRTC 相关
- MediaStream API (视频流处理)
- WebRTC API (实时通信)
- Socket.io-client (实时websocket通信)

#### 性能优化
- React.lazy (代码分割)
- Suspense (懒加载)
- Web Workers (后台计算)
- Service Workers (离线缓存)
- React.memo (组件缓存)

#### 开发规范
- TypeScript 严格模式
- ESLint Airbnb 规范
- Git Flow 工作流
- Conventional Commits
- JSDoc 文档规范

### 数据存储
- Vector Database
- File Storage System

## 环境配置

### macOS 环境
```bash
conda env create -f environment-mac.yml
conda activate bike-fit
```

### CUDA 环境 (Windows/Linux)
```bash
conda env create -f environment-cuda.yml
conda activate bike-fit-cuda
```

## 快速开始

1. 克隆项目
```bash
git clone [repository-url]
cd Bike-fit
```

2. 安装依赖
```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件配置必要的环境变量
```

4. 启动服务
```bash
# 启动后端
python backend/server.py

# 启动前端
cd frontend
npm start
```

## 主要功能

- 实时姿态检测与分析
- 智能自行车适配建议
- 数据可视化与报告生成
- 用户管理系统

## 部署

1. 安装依赖
```bash
conda env create -f environment-mac.yml
conda activate bike-fit
```

2. 启动服务
```bash
python backend/server.py
```

3. 访问地址
```bash
demo.kittybob.com
```

### 开发环境
- 本地开发服务器
- 测试数据库配置

### 生产环境
- 服务器要求
- 部署流程
- 监控方案

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 发起 Pull Request

## 许可证

[许可证类型]

## 联系方式

- 项目维护者：[联系信息]
- 问题反馈：[Issue 链接]





