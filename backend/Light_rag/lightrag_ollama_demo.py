import asyncio
import nest_asyncio
import os
import inspect
import logging
import shutil
import hashlib
import json
import subprocess
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from openai import AsyncOpenAI

nest_asyncio.apply()

logging.basicConfig(
    format="%(levelname)s: Process %(process)d %(message)s",
    level=logging.INFO
)

async def openai_complete(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    
    model = kwargs.pop("model", os.getenv("API_MODEL", "qwen-plus"))
    
    # 过滤掉不支持的参数
    supported_params = {
        "model", "messages", "temperature", "top_p", "n", 
        "stream", "stop", "max_tokens", "presence_penalty", 
        "frequency_penalty", "logit_bias", "user"
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    
    client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    
    try:
        logging.info(f"Calling API with model: {model}")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **filtered_kwargs
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        raise

class LightRagManager:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.use_local_model = os.getenv("USE_LOCAL_OLLAMA", "true").lower() == "true"
        self.pdf_source_dir = os.getenv("PDF_SOURCE_DIR", "./pdfs")
        self.md_temp_dir = os.getenv("MD_TEMP_DIR", "./md_temp")
        self.file_hash_record_path = os.path.join(self.working_dir, "processed_files.json")
        self.processed_files = {}
        logging.info(f"模型类型: {'本地' if self.use_local_model else '在线'}")
        # 添加异步调用initialize_rag
        asyncio.run(self.initialize_rag())

    async def initialize_rag(self):
        if self.use_local_model:
            logging.info("初始化本地Ollama模型")
            llm_model_func = ollama_model_complete
            model_kwargs = {
                "host": "http://localhost:11434",
                "options": {"num_ctx": 32768},
            }
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:72b-instruct-q4_K_M")
            logging.info(f"本地模型名称: {model_name}")
        else:
            logging.info("初始化在线API模型")
            llm_model_func = openai_complete
            model_kwargs = {"model": os.getenv("API_MODEL", "qwen-plus")}
            model_name = model_kwargs["model"]
            logging.info(f"API模型名称: {model_name}")

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=model_name,
            llm_model_max_async=2,
            llm_model_max_token_size=32768,
            llm_model_kwargs=model_kwargs,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model=os.getenv("EMBED_MODEL"), host="http://localhost:11434"
                ),
            ),
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        
        # 加载已处理文件的记录
        await self.load_processed_files()
        
        # 检查向量数据库是否为空，如果为空则处理所有文件
        if await self.is_vector_db_empty():
            logging.info("向量数据库为空，开始处理所有文件...")
            await self.process_all_files_to_vector_db()
        else:
            # 如果数据库不为空，检查是否有新文件需要处理
            logging.info("向量数据库已存在，检查是否有新文件...")
            await self.process_new_files_to_vector_db()

    async def load_processed_files(self):
        """加载已处理文件的记录"""
        try:
            if os.path.exists(self.file_hash_record_path):
                with open(self.file_hash_record_path, 'r', encoding='utf-8') as f:
                    self.processed_files = json.load(f)
                logging.info(f"已加载处理过的文件记录，共 {len(self.processed_files)} 个文件")
            else:
                self.processed_files = {}
                logging.info("未找到已处理文件记录，将创建新记录")
        except Exception as e:
            logging.error(f"加载已处理文件记录时出错: {str(e)}")
            self.processed_files = {}

    async def save_processed_files(self):
        """保存已处理文件的记录"""
        try:
            with open(self.file_hash_record_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存处理过的文件记录，共 {len(self.processed_files)} 个文件")
        except Exception as e:
            logging.error(f"保存已处理文件记录时出错: {str(e)}")

    async def get_file_hash(self, file_path):
        """获取文件的哈希值，用于判断文件是否已处理或有更新"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logging.error(f"计算文件哈希值时出错 {file_path}: {str(e)}")
            return None

    async def is_vector_db_empty(self) -> bool:
        """检查向量数据库是否为空"""
        try:
            # 检查向量数据库文件是否存在
            vector_db_files = [
                os.path.join(self.working_dir, "vector_store.faiss"),
                os.path.join(self.working_dir, "vector_store_metadata.json")
            ]
            
            # 如果关键文件不存在或为空，则认为数据库为空
            for file_path in vector_db_files:
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    logging.info(f"向量数据库文件 {file_path} 不存在或为空")
                    return True
                    
            # 可以添加更多检查逻辑，例如查询数据库中的条目数量
            # 这里简化处理，仅检查文件存在性
            
            return False
        except Exception as e:
            logging.error(f"检查向量数据库时出错: {str(e)}")
            # 出错时默认为空，以便重新处理
            return True

    async def process_all_files_to_vector_db(self):
        """处理所有文件并添加到向量数据库"""
        try:
            # 检查源目录是否存在
            if not os.path.exists(self.pdf_source_dir):
                logging.error(f"源目录 {self.pdf_source_dir} 不存在")
                os.makedirs(self.pdf_source_dir, exist_ok=True)
                logging.info(f"已创建源目录 {self.pdf_source_dir}")
                return
                
            # 创建临时MD目录
            if os.path.exists(self.md_temp_dir):
                shutil.rmtree(self.md_temp_dir)
            os.makedirs(self.md_temp_dir, exist_ok=True)
            logging.info(f"已创建临时MD目录 {self.md_temp_dir}")
            
            # 处理所有文件
            await self.process_source_directory()
            
            # 保存处理记录
            await self.save_processed_files()
            
        except Exception as e:
            logging.error(f"处理所有文件时出错: {str(e)}")

    async def process_new_files_to_vector_db(self):
        """处理新文件并添加到向量数据库"""
        try:
            # 检查源目录是否存在
            if not os.path.exists(self.pdf_source_dir):
                logging.error(f"源目录 {self.pdf_source_dir} 不存在")
                os.makedirs(self.pdf_source_dir, exist_ok=True)
                logging.info(f"已创建源目录 {self.pdf_source_dir}")
                return
                
            # 确保临时MD目录存在
            os.makedirs(self.md_temp_dir, exist_ok=True)
            
            # 处理新文件
            has_new_files = await self.process_source_directory(only_new=True)
            
            if not has_new_files:
                logging.info("没有发现新文件，无需更新向量数据库")
            
            # 保存处理记录
            await self.save_processed_files()
            
        except Exception as e:
            logging.error(f"处理新文件时出错: {str(e)}")

    async def process_source_directory(self, only_new=False):
        """处理源目录中的所有文件"""
        has_new_files = False
        
        # 获取源目录中的所有文件
        all_files = []
        for root, _, files in os.walk(self.pdf_source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        logging.info(f"源目录中共有 {len(all_files)} 个文件")
        
        # 按文件类型分类
        pdf_files = []
        md_files = []
        office_files = []
        
        for file_path in all_files:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 计算文件哈希值
            file_hash = await self.get_file_hash(file_path)
            rel_path = os.path.relpath(file_path, self.pdf_source_dir)
            
            # 如果只处理新文件，且文件已处理且哈希值未变，则跳过
            if only_new and rel_path in self.processed_files and self.processed_files[rel_path] == file_hash:
                logging.info(f"文件 {rel_path} 已处理且未变更，跳过")
                continue
                
            has_new_files = True
            
            # 根据文件类型分类
            if file_ext == '.pdf':
                pdf_files.append((file_path, file_hash, rel_path))
            elif file_ext == '.md':
                md_files.append((file_path, file_hash, rel_path))
            elif file_ext in ['.doc', '.docx', '.ppt', '.pptx']:
                office_files.append((file_path, file_hash, rel_path))
        
        # 处理Office文件（转换为PDF）
        if office_files:
            logging.info(f"发现 {len(office_files)} 个Office文件，开始转换为PDF")
            pdf_from_office = await self.convert_office_to_pdf(office_files)
            pdf_files.extend(pdf_from_office)
        
        # 处理PDF文件（转换为MD）
        if pdf_files:
            logging.info(f"开始处理 {len(pdf_files)} 个PDF文件")
            await self.process_pdf_files(pdf_files)
        
        # 直接处理MD文件
        if md_files:
            logging.info(f"开始处理 {len(md_files)} 个MD文件")
            await self.process_md_files(md_files)
            
        return has_new_files

    async def convert_office_to_pdf(self, office_files):
        """将Office文件转换为PDF"""
        pdf_files = []
        pdf_temp_dir = os.path.join(self.md_temp_dir, "pdf_temp")
        os.makedirs(pdf_temp_dir, exist_ok=True)
        
        for file_path, file_hash, rel_path in office_files:
            try:
                file_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(file_name)[0]
                output_pdf = os.path.join(pdf_temp_dir, f"{name_without_ext}.pdf")
                
                logging.info(f"正在将 {file_name} 转换为PDF")
                
                # 使用LibreOffice或其他工具转换
                # 这里使用LibreOffice的命令行工具soffice
                # 确保系统中已安装LibreOffice
                try:
                    # 尝试使用LibreOffice转换
                    cmd = [
                        'soffice', 
                        '--headless', 
                        '--convert-to', 'pdf', 
                        '--outdir', pdf_temp_dir, 
                        file_path
                    ]
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logging.error(f"转换 {file_name} 失败: {stderr.decode('utf-8', errors='ignore')}")
                        continue
                        
                    logging.info(f"成功将 {file_name} 转换为PDF")
                    
                    # 计算生成的PDF文件哈希值
                    pdf_hash = await self.get_file_hash(output_pdf)
                    pdf_rel_path = f"converted_{rel_path}.pdf"
                    
                    pdf_files.append((output_pdf, pdf_hash, pdf_rel_path))
                    
                except Exception as e:
                    logging.error(f"转换 {file_name} 时出错: {str(e)}")
                    continue
                    
            except Exception as e:
                logging.error(f"处理Office文件 {file_path} 时出错: {str(e)}")
                continue
                
        return pdf_files

    async def process_pdf_files(self, pdf_files):
        """处理PDF文件并转换为MD"""
        local_image_dir = os.path.join(self.md_temp_dir, "images")
        local_md_dir = self.md_temp_dir
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        for file_path, file_hash, rel_path in pdf_files:
            try:
                file_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(file_name)[0]
                
                logging.info(f"正在处理PDF文件: {file_name}")

                # 读取PDF文件内容
                with open(file_path, 'rb') as f:
                    pdf_bytes = f.read()

                # 创建数据集实例
                ds = PymuDocDataset(pdf_bytes)

                # 推理
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    logging.info(f"使用OCR模式处理 {file_name}")
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    logging.info(f"使用文本模式处理 {file_name}")
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                md_content = pipe_result.get_markdown(image_dir)
                md_file_name = f"{name_without_ext}.md"

                # 保存Markdown文件
                pipe_result.dump_md(md_writer, md_file_name, image_dir)
                logging.info(f"PDF文件 {file_name} 已转换为MD: {md_file_name}")
                
                # 将MD内容添加到向量数据库
                self.rag.insert(md_content)
                logging.info(f"已将 {md_file_name} 的内容添加到向量数据库")
                
                # 更新处理记录
                self.processed_files[rel_path] = file_hash
                
            except Exception as e:
                logging.error(f"处理PDF文件 {file_path} 时出错: {str(e)}")

    async def process_md_files(self, md_files):
        """直接处理MD文件并添加到向量数据库"""
        for file_path, file_hash, rel_path in md_files:
            try:
                file_name = os.path.basename(file_path)
                logging.info(f"正在处理MD文件: {file_name}")
                
                # 读取MD文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # 将MD内容添加到向量数据库
                self.rag.insert(md_content)
                logging.info(f"已将 {file_name} 的内容添加到向量数据库")
                
                # 更新处理记录
                self.processed_files[rel_path] = file_hash
                
            except Exception as e:
                logging.error(f"处理MD文件 {file_path} 时出错: {str(e)}")

    async def async_convert_pdfs_to_md(self, input_folder, output_folder):
        """将PDF文件转换为MD（保留旧方法以兼容）"""
        local_image_dir = os.path.join(output_folder, "images")
        local_md_dir = output_folder
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        for pdf_file_name in os.listdir(input_folder):
            if pdf_file_name.endswith(".pdf"):
                name_without_suff = pdf_file_name.split(".")[0]
                logging.info(f"正在处理PDF文件: {pdf_file_name}")

                # read bytes
                reader1 = FileBasedDataReader(input_folder)
                pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

                # proc
                ## Create Dataset Instance
                ds = PymuDocDataset(pdf_bytes)

                ## inference
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                md_content = pipe_result.get_markdown(image_dir)

                ### dump markdown
                pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
                logging.info(f"PDF文件 {pdf_file_name} 已转换为MD")

    async def async_add_md_files_to_db(self, input_folder):
        """将MD文件添加到向量数据库（保留旧方法以兼容）"""
        md_files_count = 0
        for md_file_name in os.listdir(input_folder):
            if md_file_name.endswith(".md"):
                # read bytes
                reader = FileBasedDataReader(input_folder)
                md_bytes = reader.read(md_file_name)  # read the md content

                # decode bytes to string
                md_content = md_bytes.decode('utf-8')

                # insert into database
                self.rag.insert(md_content)
                logging.info(f"已添加 {md_file_name} 到数据库")
                md_files_count += 1
        
        logging.info(f"总共添加了 {md_files_count} 个MD文件到数据库")
        return None  # 显式返回None，避免await None的错误

    async def query_async(self, question, mode="hybrid", only_need_context=True, stream=False):
        """异步查询方法"""
        valid_modes = ["local", "global", "hybrid", "naive", "mix"]
        if mode not in valid_modes:
            raise ValueError(f"无效的模式: {mode}。必须是以下之一: {valid_modes}")
            
        # 初始化param变量，避免"param可能未绑定"的错误
        param = None
            
        # 使用字符串字面量而不是变量
        if mode == "local":
            param = QueryParam(mode="local", only_need_context=only_need_context, stream=stream)
        elif mode == "global":
            param = QueryParam(mode="global", only_need_context=only_need_context, stream=stream)
        elif mode == "hybrid":
            param = QueryParam(mode="hybrid", only_need_context=only_need_context, stream=stream)
        elif mode == "naive":
            param = QueryParam(mode="naive", only_need_context=only_need_context, stream=stream)
        elif mode == "mix":
            param = QueryParam(mode="mix", only_need_context=only_need_context, stream=stream)
        
        # 确保param已被初始化
        if param is None:
            # 默认使用hybrid模式
            param = QueryParam(mode="hybrid", only_need_context=only_need_context, stream=stream)
            
        return self.rag.query(question, param=param)

    def query(self, question, mode="hybrid", only_need_context=True, stream=False):
        """同步查询方法"""
        return asyncio.run(self.query_async(question, mode, only_need_context, stream))

    def visualize(self, query=None):
        import networkx as nx
        from pyvis.network import Network

        # Load the GraphML file
        G = nx.read_graphml(self.working_dir + "/graph_chunk_entity_relation.graphml")

        # Create a Pyvis network
        net = Network(notebook=True)

        # Convert NetworkX graph to Pyvis network
        net.from_nx(G)

        # Save and display the network
        net.show('knowledge_graph.html')


if __name__ == "__main__":
    input_source = "./source"
    db = "./lr_db"

    manager = LightRagManager(db)

    # 使用异步查询方法
    result = asyncio.run(manager.query_async("什么是bike fitting", stream=True))
    print(result)
    
    # 或使用同步查询方法
    # result = manager.query("什么是bike fitting?", stream=True, only_need_context=False)
    # print(result)
    
    # manager.visualize()