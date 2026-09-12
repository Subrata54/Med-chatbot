import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

from typing import List

load_dotenv()


# -----------------------------
# PDF LOADING
# -----------------------------

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# -----------------------------
# KEEP ONLY REQUIRED METADATA
# -----------------------------

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


# -----------------------------
# TEXT SPLITTING
# -----------------------------

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# -----------------------------
# HUGGING FACE REMOTE EMBEDDINGS
# -----------------------------

class HuggingFaceRemoteEmbeddings(Embeddings):

    def __init__(self):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN is missing. Add HF_TOKEN to your .env file "
                "and Vercel Environment Variables."
            )

        self.client = InferenceClient(
            api_key=token,
            provider="hf-inference"
        )

        self.model = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.feature_extraction(
            texts,
            model=self.model
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.client.feature_extraction(
            text,
            model=self.model
        )

        return embedding[0].tolist()


# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def download_hugging_face_embeddings():
    return HuggingFaceRemoteEmbeddings()