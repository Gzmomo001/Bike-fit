import os
import asyncio

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod




def convert_pdfs_to_md(input_folder, output_folder):
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

                ## pipeline
                pipe_result = infer_result.pipe_ocr_mode(image_writer)

            else:
                infer_result = ds.apply(doc_analyze, ocr=False)

                ## pipeline
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            ### get markdown content
            md_content = pipe_result.get_markdown(image_dir)

            ### dump markdown
            pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)


async def async_convert_pdfs_to_md(input_folder, output_folder):
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

if __name__ == "__main__":
    asyncio.run(async_convert_pdfs_to_md("./pdfs", "./md"))
