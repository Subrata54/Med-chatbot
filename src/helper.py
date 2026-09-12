import os
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.embeddings import Embeddings


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

        # IMPORTANT:
        # This model produces 384-dimensional embeddings.
        self.model = "sentence-transformers/all-MiniLM-L6-v2"

    def _convert_embedding(self, embedding):
        """
        Convert Hugging Face output into a single
        384-dimensional vector.
        """

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # If HF returns [[...384 values...]]
        if (
            isinstance(embedding, list)
            and len(embedding) == 1
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]

        # Safety check
        if not isinstance(embedding, list):
            raise ValueError(
                f"Invalid embedding returned by Hugging Face: {type(embedding)}"
            )

        if len(embedding) != 384:
            raise ValueError(
                f"Embedding dimension is {len(embedding)}, "
                f"but Pinecone index requires 384."
            )

        return [float(x) for x in embedding]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        results = []

        for text in texts:
            embedding = self.client.feature_extraction(
                text,
                model=self.model
            )

            vector = self._convert_embedding(embedding)

            results.append(vector)

        return results

    def embed_query(self, text: str) -> List[float]:

        embedding = self.client.feature_extraction(
            text,
            model=self.model
        )

        # IMPORTANT:
        # Do NOT use embedding[0].
        # That was causing the wrong dimension.
        return self._convert_embedding(embedding)


# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def download_hugging_face_embeddings():
    return HuggingFaceRemoteEmbeddings()