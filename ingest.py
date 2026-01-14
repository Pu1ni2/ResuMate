import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_documents(resume_path: str, bio_path: str):
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Missing resume PDF: {resume_path}")
    if not os.path.exists(bio_path):
        raise FileNotFoundError(f"Missing bio TXT: {bio_path}")

    pdf_docs = PyPDFLoader(resume_path).load()
    bio_docs = TextLoader(bio_path, encoding="utf-8").load()

    
    for d in pdf_docs:
        d.metadata["source"] = "resume.pdf"
    for d in bio_docs:
        d.metadata["source"] = "details.txt"

    return pdf_docs + bio_docs


def chunk_documents(docs, chunk_size: int = 900, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_chroma(
    chunks,
    persist_dir: str,
    collection_name: str = "about_me",
    embedding_model: str = "text-embedding-3-small",
    reset: bool = False,
):
    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)

    # Data is persisted automatically because persist_directory is set.

    return vectordb


def main():
    load_dotenv()  # reads .env if present
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Set it in your environment or a .env file."
        )

    resume_path = os.getenv("RESUME_PATH", "resume.pdf")
    bio_path = os.getenv("BIO_PATH", "details.txt")
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "about_me")

    reset = os.getenv("RESET_DB", "false").lower() in {"1", "true", "yes"}

    print(f"Loading: {resume_path} and {bio_path}")
    docs = load_documents(resume_path, bio_path)

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print(f"Building Chroma DB at: {persist_dir} (collection: {collection_name})")
    build_chroma(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        reset=reset,
    )

    print(" Ingestion complete.")


if __name__ == "__main__":
    main()
