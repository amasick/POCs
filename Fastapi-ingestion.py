import os
import re
import uuid
import json
import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticChunker,
)
from langchain_openai import OpenAIEmbeddings


# ============================================
# ✅ CLASS 1 — DOCUMENT LOADER
# ============================================

class DocumentLoader:

    def load(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext == ".docx":
            return self.load_docx(file_path)
        elif ext == ".txt":
            return self.load_txt(file_path)
        elif ext == ".csv":
            return self.load_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            return self.load_excel(file_path)
        else:
            raise Exception(f"Unsupported file format: {ext}")

    def load_pdf(self, path):
        loader = PyPDFLoader(path)
        return loader.load()

    def load_docx(self, path):
        loader = Docx2txtLoader(path)
        return loader.load()

    def load_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text)]

    def load_csv(self, path):
        df = pd.read_csv(path)
        text = df.to_string()
        return [Document(page_content=text)]

    def load_excel(self, path):
        df = pd.read_excel(path)
        text = df.to_string()
        return [Document(page_content=text)]


# ============================================
# ✅ CLASS 2 — TEXT CLEANER / EXTRACTOR
# ============================================

class TextExtractor:

    def clean(self, docs):
        cleaned_docs = []

        for doc in docs:
            text = doc.page_content

            text = re.sub(r"\s+", " ", text)   # remove extra whitespace
            text = text.strip()

            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata
                )
            )

        return cleaned_docs


# ============================================
# ✅ CLASS 3 — CHUNKER
# ============================================

class Chunker:

    def __init__(self):
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    def char_split(self, docs, chunk_size=800, chunk_overlap=100):
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    def recursive_split(self, docs, chunk_size=900, chunk_overlap=150):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    def semantic_split(self, docs, min_chunk_size=200, max_chunk_size=800):
        splitter = SemanticChunker(
            embeddings=self.embedder,
            breakpoint_threshold_type="percent",
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        return splitter.split_documents(docs)


# ============================================
# ✅ FASTAPI APP + ENDPOINT
# ============================================

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload any document → Load → Clean → Chunk → return JSONL file.
    """

    # Step 1 — Save file
    file_id = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}-{file.filename}")

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # Step 2 — Initialize pipeline components
    loader = DocumentLoader()
    extractor = TextExtractor()
    chunker = Chunker()

    # Step 3 — Pipeline: Load → Clean → Chunk
    docs = loader.load(saved_path)
    docs = extractor.clean(docs)

    # ✅ Change here if you want char or semantic split
    chunks = chunker.recursive_split(docs)

    # Step 4 — Write JSONL output
    jsonl_path = os.path.join(OUTPUT_DIR, f"{file_id}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as out_file:
        for idx, chunk in enumerate(chunks):
            record = {
                "id": idx,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            out_file.write(json.dumps(record) + "\n")

    # Step 5 — Return JSONL file
    return FileResponse(
        jsonl_path,
        media_type="application/jsonl",
        filename=f"{file.filename}.jsonl"
    )
