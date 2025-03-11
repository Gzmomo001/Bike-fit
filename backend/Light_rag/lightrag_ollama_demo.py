import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

class LightRagManager:
    def __init__(self, working_dir=WORKING_DIR):
        self.working_dir = working_dir
        # 添加异步调用initialize_rag
        asyncio.run(self.initialize_rag())


    async def initialize_rag(self):
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name="qwen2.5:32B",
            llm_model_max_async=2,
            llm_model_max_token_size=32768,
            llm_model_kwargs={
                "host": "http://localhost:11434",
                "options": {"num_ctx": 32768},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model="nomic-embed-text:latest", host="http://localhost:11434"
                ),
            ),
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    async def async_convert_pdfs_to_md(self, input_folder, output_folder):
        local_image_dir = os.path.join(output_folder, "images")
        local_md_dir = output_folder
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        for pdf_file_name in os.listdir(input_folder):
            if pdf_file_name.endswith(".pdf"):
                name_without_suff = pdf_file_name.split(".")[0]

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

    async def async_add_md_files_to_db(self, input_folder):
        for md_file_name in os.listdir(input_folder):
            if md_file_name.endswith(".md"):
                # read bytes
                reader = FileBasedDataReader(input_folder)
                md_bytes = reader.read(md_file_name)  # read the md content

                # decode bytes to string
                md_content = md_bytes.decode('utf-8')

                # insert into database
                self.rag.insert(md_content)
                print(f"Added {md_file_name} to database.")

    def query(self, question, mode="hybrid", only_need_context = True, stream=False):
        valid_modes = ["local", "global", "hybrid", "naive", "mix"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        param = QueryParam(mode=mode, only_need_context = only_need_context,stream=stream)
        return self.rag.query(question, param=param)

    def visualize(self,query):
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

    # asyncio.run(manager.async_convert_pdfs_to_md("./pdfs", "./md"))
    # asyncio.run(manager.async_add_md_files_to_db("./md"))
    # asyncio.run((manager.async_add_md_files_to_db(input_source)))
    asyncio.run(manager.query("什么是bike fitting",stream=True))

    # result = manager.query("什么是bike fitting?", stream=True, only_need_context = False)
    # print(result)
    # manager.visualize()