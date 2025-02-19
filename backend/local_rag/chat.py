import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from create_kb import *
DB_PATH = "VectorStore"
TMP_NAME = "tmp_abcd"
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

uselocalmodel = True

if uselocalmodel:
    from langchain_community.embeddings import ModelScopeEmbeddings
    from llama_index.core.embeddings import LangchainEmbedding
    embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
    EMBED_MODEL = LangchainEmbedding(embeddings)

# 设置嵌入模型
Settings.embed_model = EMBED_MODEL

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