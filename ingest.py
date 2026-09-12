"""
Ingestion pipeline for the RAG chatbot.

Responsibilities:
1. Load PDF documents from the data/ directory (or a single given path)
2. Split them into overlapping chunks
3. Generate embeddings for each chunk
4. Persist everything into a local Chroma vector store

Run standalone:
    python ingest.py
This will (re)build the vector store from every PDF in DATA_DIR.
"""

import os
import glob
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(data_dir="C:/Users/LENOVO/Downloads/"):

    """Load all PDFs from a directory. Each page becomes a LangChain Document
    with metadata (source filename, page number) attached automatically."""
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}/'. Add some PDFs there first."
        )

    all_docs = []
    for path in pdf_paths:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_paths)} PDF(s).")
    return all_docs


def split_documents(documents):
    """Split documents into overlapping chunks so retrieval can pull focused,
    relevant snippets instead of entire pages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")
    return chunks


def build_vector_store(chunks, persist_directory: str = config.VECTOR_STORE_DIR, reset: bool = True):
    """Embed chunks and persist them into a local Chroma vector store."""
    if reset and os.path.exists(persist_directory):
        print(f"Removing existing vector store at '{persist_directory}'...")
        shutil.rmtree(persist_directory)

    embeddings = NVIDIAEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.NVIDIA_API_KEY,
    )

    print("Generating embeddings and writing to Chroma (this may take a moment)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"Vector store built and persisted at '{persist_directory}'.")
    return vector_store


def ingest_single_file(file_path: str, persist_directory: str = config.VECTOR_STORE_DIR):
    """Used by the Streamlit app when a user uploads a single new PDF at runtime.
    Adds it to the existing store instead of rebuilding everything from scratch."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = split_documents(docs)

    embeddings = NVIDIAEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.NVIDIA_API_KEY,
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} chunks from '{file_path}' to the vector store.")
    return vector_store


def main():
    if not config.NVIDIA_API_KEY:
        raise EnvironmentError(
            "NVIDIA_API_KEY not set. Copy .env.example to .env and add your key."
        )

    documents = load_documents()
    chunks = split_documents(documents)
    build_vector_store(chunks)
    print("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
