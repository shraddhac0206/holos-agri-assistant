from __future__ import annotations

import os
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import settings


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self.model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.encode([text], normalize_embeddings=True)[0])


def get_embedding_model() -> Embeddings:
    if settings.model_provider == "openai":
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return LocalSentenceTransformerEmbeddings()


def build_vector_store_from_files(docs_dir: str, store: str = "chroma"):
    texts, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if not fn.lower().endswith((".txt", ".md", ".pdf")):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            for chunk in splitter.split_text(content):
                texts.append(chunk)
                metadatas.append({"source": path})

    embeddings = get_embedding_model()
    vector_dir = settings.vector_dir
    os.makedirs(vector_dir, exist_ok=True)

    if store == "faiss":
        vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        vs.save_local(os.path.join(vector_dir, "faiss"))
    else:
        persist_dir = os.path.join(vector_dir, "chroma")
        vs = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_dir)
        vs.persist()


def load_vector_store(store: str = "chroma"):
    embeddings = get_embedding_model()
    vector_dir = settings.vector_dir
    if store == "faiss":
        return FAISS.load_local(os.path.join(vector_dir, "faiss"), embeddings, allow_dangerous_deserialization=True)
    return Chroma(persist_directory=os.path.join(vector_dir, "chroma"), embedding_function=embeddings)


def retrieve(query: str, k: int = 4) -> List[Document]:
    vs = load_vector_store(settings.rag_store)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


