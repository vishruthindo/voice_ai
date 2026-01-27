from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# import pickle
import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
# import os, io
import numpy as np
# import cv2
# import re
from typing import List, Dict, Union, Optional
#from django.conf import settings
import pandas as pd
from langchain_core.documents import Document
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import shutil, logging, re, os, io, pickle, pytesseract, cv2, faiss
from io import BytesIO
# import logging
# from google.cloud import vision
from typing import List, Dict, Union, Optional, Any, Tuple
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
    PyPDFLoader,
    UnstructuredImageLoader,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
 
 
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pydantic import BaseModel
 
from google.cloud import vision
from multiprocessing import Pool, cpu_count
import tempfile, cv2
from dotenv import load_dotenv
import os
 
load_dotenv()
tesseract_path = os.getenv("TESSERACT_FILE", r"C:\Users\mohan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path
 
 
 
 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
def delete_file_embeddings(embeddings_file: str, fileid: Optional[Union[str, List[str]]] = None,
                      tenantid: Optional[Union[str, List[str]]] = None,
                      departmentid: Optional[Union[str, List[str]]] = None,
                      filename: Optional[str] = None,
                      default: Optional[bool] = None) -> Dict:
    """
    Delete embeddings based on filtering criteria.
 
    Args:
        embeddings_file: Path to the embeddings file
        fileid: File ID(s) to delete (single or list)
        tenantid: Tenant ID(s) to delete (single or list)
        departmentid: Department ID(s) to delete (single or list)
        filename: Exact filename to delete
        default: Delete by default status (True/False/None for no filter)
    """
    print("...........still using old deleting embedding..................",flush=True)
    if not os.path.exists(embeddings_file):
        return False
 
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
 
    if not isinstance(data, list):
        return {
            "status": "error",
            "message": f"The embeddings file '{embeddings_file}' does not contain the expected list structure.",
            "file": embeddings_file
        }
 
    # Convert single filter values to lists
    fileids = [fileid] if fileid is not None and not isinstance(fileid, list) else fileid
    tenantids = [tenantid] if tenantid is not None and not isinstance(tenantid, list) else tenantid
    departmentids = [departmentid] if departmentid is not None and not isinstance(departmentid, list) else departmentid
 
    # Filter out entries to delete
    updated_data = [entry for entry in data if not (
        (fileids is None or entry["fileid"] in fileids) and \
        (tenantids is None or entry["tenantid"] in tenantids) and \
        (departmentids is None or entry["departmentid"] in departmentids) and \
        (filename is None or entry["filename"] == filename) and \
        (default is None or entry["default"] == default)
    )]
 
    # Save the updated data
    with open(embeddings_file, 'wb') as f:
        pickle.dump(updated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    return True
 
 
 
 
 
# --- Data Models ---
class DocumentMetadata(BaseModel):
    fileid: str
    tenantid: str
    departmentid: str
    filename: str
    default: bool = False
    sourcetype: str = "pdf"
 
class DocumentWithEmbedding(BaseModel):
    text: str
    embedding: List[float]
    metadata: DocumentMetadata
 
 
 
# --- PDF Processing ---
class PaddleOCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
 
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and images from PDF using hybrid approach"""
        print("🚀 Initializing PDF processing...")
        doc = fitz.open(pdf_path)
        output_content = []
 
        for page_num, page in enumerate(doc):
            output_content.append(f"\n\n=== Page {page_num + 1} ===")
 
            # Extract Normal Text
            output_content.append("\n--- Normal Text ---")
            normal_text = page.get_text("text").strip()
            if normal_text:
                output_content.append(normal_text)
            else:
                output_content.append("(No selectable text found)")
 
            # Extract Images + OCR
            output_content.append("\n--- OCR on Images (as Table if possible) ---")
            image_content = self._process_images(page)
            output_content.append(image_content)
 
        print("\n✅ Hybrid text + OCR extraction complete.")
        return "\n".join(output_content)
 
    def _process_images(self, page) -> str:
        """Process images in a PDF page using OCR"""
        image_list = page.get_images(full=True)
        image_contents = []
 
        if not image_list:
            return "(No images found)"
 
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_np = np.array(image_pil)
 
                # Run OCR
                result = self.ocr.ocr(image_np, cls=True)
                ocr_content = self._format_ocr_results(result)
                image_contents.append(ocr_content)
 
            except Exception as e:
                image_contents.append(f"(Error processing image {img_index + 1}: {e})")
 
        return "\n".join(image_contents)
 
    def _format_ocr_results(self, result) -> str:
        """Format OCR results as string"""
        if not result:
            return "(No OCR result found in image)"
 
        table_rows = []
        for line in result:
            if line:
                row_text = []
                for entry in line:
                    if entry and len(entry) == 2:
                        box, (text, confidence) = entry
                        row_text.append(text.strip())
                if row_text:
                    table_rows.append(" | ".join(row_text))
 
        if table_rows:
            return "\n".join(table_rows)
        return "(No structured rows found)"
#-----------GOCR
class GoogleVisionOCRProcessor():
    def extract_text_from_pdf(self,pdf_path):
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=300)
        client = vision.ImageAnnotatorClient()
        results = []
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)}")
            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='PNG')
            image = vision.Image(content=img_byte_arr.getvalue())
            # Japanese language hint
            image_context = vision.ImageContext(language_hints=["ja", "en", "hi", "de", "ar", "it", "kn", "/*ko", "ms", "ru", "sr", "es", "th", "fr", "nl", "pt", "ta", "te"])
            response = client.document_text_detection(image=image, image_context=image_context)
            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")
            text = response.full_text_annotation.text
            results.append(text)
        return "\n\n".join(results)
   
#-- Tesseract
class TesseractOCRProcessor():
    def __init__(self):
        self.nthreads = max(1, min(4, cpu_count() - 1))
        self.temp_dir = None
 
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
 
    def binarize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary
 
    def find_contours_text(self, img, kernel_size, sort_key=None):
        binary = self.binarize(img)
        kernel = np.ones(kernel_size, dtype=np.uint8)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        ctrs, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(ctrs, key=sort_key if sort_key else lambda ctr: cv2.boundingRect(ctr)[0])
 
    def get_rect(self, img, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        return img[y:y+h, x:x+w]
 
    def find_sections(self, img):
        img_h, img_w = img.shape[:2]
        offsets = (2, 20)
        img = img[offsets[0]:-offsets[0], offsets[1]:-offsets[1]]
 
        sort_key = lambda ctr: cv2.boundingRect(ctr)[1]
        ctrs = self.find_contours_text(img, (10, 100), sort_key=sort_key)
 
        sections = []
        for ctr in ctrs:
            _, _, w, h = cv2.boundingRect(ctr)
            if h / img_h > 0.1:
                sections.append(self.get_rect(img, ctr))
        return sections
 
    def sort_contours(self, ctrs):
        threshold = 10
        ctrs = sorted(ctrs, key=lambda ctr: -cv2.boundingRect(ctr)[0])
 
        cols = []
        col = []
        prev = cv2.boundingRect(ctrs[0])[0]
 
        for i, ctr in enumerate(ctrs):
            x, _, _, _ = cv2.boundingRect(ctr)
            if abs(x - prev) >= threshold:
                cols.append(col)
                col = []
            col.append(i)
            prev = x
 
        if col:
            cols.append(col)
 
        sorted_ctrs = []
        for col in cols:
            col_ctrs = [ctrs[i] for i in col]
            sorted_col_ctrs = sorted(col_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
            sorted_ctrs.append(sorted_col_ctrs)
 
        return [item for sublist in sorted_ctrs for item in sublist]
 
    def find_text_in_section(self, section):
        text_width = 35
        ctrs = self.find_contours_text(section, (text_width // 2, 10))
        ctrs = self.sort_contours(ctrs)
 
        text_blocks = []
        for ctr in ctrs:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 30:
                text_blocks.append(self.get_rect(section, ctr))
        return text_blocks
    def process_image(self, img_path: str) -> str:
        print("🚀 Initializing PDF processing using Tesseract OCR engine...")
        def ocr_image(img, lang='jpn+jpn_vert+eng+chi_sim+chi_tra', psm=3):
            config = f'--psm {psm} --oem 3'
            return pytesseract.image_to_string(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                lang=lang,
                config=config
            ).strip()
 
        if img_path.lower().endswith('.pdf'):
            images = convert_from_path(img_path, dpi=400)  # use high DPI for better accuracy
            if not images:
                return ""
 
            all_text = []
            for img in images:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                full_text = ocr_image(img_cv, psm=3)
 
                if full_text and len(full_text.splitlines()) > 3:
                    all_text.append(full_text)
                else:
                    sections = self.find_sections(img_cv)
                    section_texts = []
                    for section in sections:
                        blocks = self.find_text_in_section(section)
                        for block in blocks:
                            text = ocr_image(block, lang='jpn+jpn_vert+eng+chi_sim+chi_tra', psm=6)  # block-level OCR
                            if text:
                                section_texts.append(text)
                    if section_texts:
                        all_text.append("\n".join(section_texts))
 
            return "\n\n".join(all_text)
        else:
            img = cv2.imread(img_path)
            if img is None:
                return ""
 
            full_text = ocr_image(img, psm=3)
            if full_text and len(full_text.splitlines()) > 3:
                return full_text
 
            sections = self.find_sections(img)
            section_texts = []
            for section in sections:
                blocks = self.find_text_in_section(section)
                for block in blocks:
                    text = ocr_image(block, lang='jpn+jpn_vert+eng+chi_sim+chi_tra', psm=6)
                    if text:
                        section_texts.append(text)
 
            return "\n".join(section_texts)
 
# --- Text Processing ---
class TextProcessor:
    def preprocess_text(self, documents: List[str]) -> List[str]:
        """Split and clean text documents into chunks"""
        processed_docs = []
        for doc in documents:
            cleaned_text = "\n".join([line.strip() for line in doc.strip().split("\n") if line.strip()])
 
            # Split based on headers or bullets/steps
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header")])
            chunks = header_splitter.split_text(cleaned_text)
 
            for chunk in chunks:
                recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
                processed_docs.extend(recursive_splitter.split_text(chunk.page_content))
 
        return processed_docs
 
# --- Embeddings Management ---
 
class EmbeddingsManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
 
    def create_and_save_embeddings(
        self,
        text_content: str,
        output_pkl: str,
        metadata: DocumentMetadata,
        update: bool = False
    ) -> None:
        """Create and save embeddings with metadata"""
 
        # Preprocess OCR text into chunks
        text_processor = TextProcessor()
        new_documents = text_processor.preprocess_text([text_content])
 
        # Generate embeddings
        new_embeddings = self.embeddings.embed_documents(new_documents)
 
        # Prepare documents with metadata
        new_docs_with_embeddings = [
            {
                "text": text,
                "embedding": embedding,
                "metadata": metadata.dict()
            }
            for text, embedding in zip(new_documents, new_embeddings)
        ]
 
        # Handle existing file
        existing_data = {'documents': []}
        if update and Path(output_pkl).exists():
            existing_data = self._load_existing_embeddings(output_pkl)
 
        # Combine old and new data
        combined_documents = existing_data['documents'] + new_docs_with_embeddings
 
        # Save to PKL file
        self._save_embeddings(output_pkl, combined_documents)
 
        print(f"\n✅ Saved {len(new_docs_with_embeddings)} new documents (total: {len(combined_documents)})")
        print(f"📁 Embeddings saved to: {output_pkl}")
 
    def _load_existing_embeddings(self, pkl_file: str) -> Dict:
        """Load existing embeddings from file"""
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Loaded existing data with {len(data['documents'])} documents")
            return data
        except Exception as e:
            print(f"⚠️ Error loading existing file: {e}. Creating new file.")
            return {'documents': []}
 
    def _save_embeddings(self, output_pkl: str, documents: List[Dict]) -> None:
        """Save embeddings to file"""
        with open(output_pkl, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'version': '2.0'  # Version with metadata support
            }, f)
 
# --- Main Pipeline ---
def process_pdf_to_embeddings(pdf_path: str,embedding_file_path:str, metadata: DocumentMetadata, metadata_text, ocr_type) -> None:
    """Complete pipeline from PDF to embeddings"""
    # Step 1: Extract text from PDF based on cofigured OCR
    if ocr_type == 'PADDLE':
        ocr_processor = PaddleOCRProcessor()
        extracted_text = ocr_processor.extract_text_from_pdf(pdf_path)
 
    elif ocr_type == 'GOOGLE_VISION':
        ocr_processor = GoogleVisionOCRProcessor()
        extracted_text = ocr_processor.extract_text_from_pdf(pdf_path)
       
    elif ocr_type == 'TESSERACT':
        with TesseractOCRProcessor() as extractor:
            extracted_text = extractor.process_image(pdf_path)
    # Step 1: Extract text from PDF
    # pdf_processor = PDFProcessor()
    # extracted_text = pdf_processor.extract_text_from_pdf(pdf_path, metadata)
 
    # Step 2: Create and save embeddings
    embeddings_manager = EmbeddingsManager()
    if os.path.exists(embedding_file_path):
        embeddings_manager.create_and_save_embeddings(
            extracted_text,
            embedding_file_path,
            metadata,
            update=True
        )
    else:
        embeddings_manager.create_and_save_embeddings(
            extracted_text,
            embedding_file_path,
            metadata,
            update=False
        )
 
 
# --- Example Usage ---
if __name__ == "__main__":
    # Example metadata for the document
    document_metadata = DocumentMetadata(
        fileid="indo1456",
        tenantid="1456",
        departmentid="IEEE",
        filename="botinformatio4n.pdf",
        default=True,
        sourcetype="pdf"
    )
    # ✅ Define paths here
    # pdf_path = r"D:\Indo_01\vb2\SourceBytes_Pricing_Model.pdf"
    pdf_path = r"vishruth_new_4.pdf"
    embedding_file_path = r"embeddings_data.pkl"
    ocr_type = "PADDLE"  # Choose from 'PADDLE', 'GOOGLE_VISION', or 'TESSERACT'
    metadata_text = "some info about file5"
    process_pdf_to_embeddings(pdf_path, embedding_file_path, document_metadata, metadata_text, ocr_type)