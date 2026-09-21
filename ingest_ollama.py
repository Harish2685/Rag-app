"""
Ingest PDFs into Chroma vector store using Ollama embeddings (FREE, LOCAL)

No API keys needed!
"""

import os
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(data_dir: str):
    """Load all PDF files from data directory."""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory '{data_dir}' not found. Creating it...")
        data_path.mkdir(exist_ok=True)
        logger.info(f"Please add your PDF files to '{data_dir}/'")
        return []
    
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in '{data_dir}/'")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF(s)")
    
    for pdf_path in pdf_files:
        logger.info(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        logger.info(f"  ✓ Loaded {len(docs)} pages")
        documents.extend(docs)
    
    return documents


def chunk_documents(documents, chunk_size: int = config_ollama.CHUNK_SIZE, chunk_overlap: int = config_ollama.CHUNK_OVERLAP):
    """Split documents into smaller chunks."""
    logger.info(f"\nChunking documents...")
    logger.info(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"  ✓ Created {len(chunks)} chunks")
    
    return chunks


def create_vectorstore(chunks, persist_directory: str = config_ollama.VECTOR_STORE_DIR):
    """Create Chroma vector store with Ollama embeddings."""
    logger.info(f"\nCreating vector store with Ollama embeddings...")
    logger.info(f"  Embedding model: {config_ollama.EMBEDDING_MODEL}")
    logger.info(f"  Ollama URL: {config_ollama.OLLAMA_BASE_URL}")
    logger.info(f"  Storage: {persist_directory}")
    
    try:
        # Initialize Ollama embeddings
        embeddings = OllamaEmbeddings(
            model=config_ollama.EMBEDDING_MODEL,
            base_url=config_ollama.OLLAMA_BASE_URL,
        )
        
        # Test embeddings
        logger.info("  Testing embeddings...")
        test_embed = embeddings.embed_query("test")
        logger.info(f"  ✓ Embeddings working ({len(test_embed)} dimensions)")
        
        # Create vector store
        logger.info("  Creating Chroma database...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        
        logger.info(f"  ✓ Vector store created successfully!")
        logger.info(f"  ✓ Total chunks embedded: {len(chunks)}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        logger.error("\n⚠️  Make sure Ollama is running:")
        logger.error(f"   1. ollama serve")
        logger.error(f"   2. ollama pull {config_ollama.EMBEDDING_MODEL}")
        raise


def main():
    """Main ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("PDF INGESTION PIPELINE (Ollama - FREE LOCAL EMBEDDINGS)")
    logger.info("=" * 60)
    
    # Step 1: Load PDFs
    logger.info(f"\nStep 1: Loading PDFs from '{config_ollama.DATA_DIR}/'...")
    documents = load_documents(config_ollama.DATA_DIR)
    
    if not documents:
        logger.warning("No documents loaded. Add PDFs to the 'data/' folder and try again.")
        return
    
    logger.info(f"✓ Loaded {len(documents)} total pages")
    
    # Step 2: Chunk documents
    logger.info(f"\nStep 2: Chunking documents...")
    chunks = chunk_documents(documents)
    
    # Step 3: Create vector store
    logger.info(f"\nStep 3: Creating vector store with Ollama embeddings...")
    vector_store = create_vectorstore(chunks)
    
    # Done!
    logger.info("\n" + "=" * 60)
    logger.info("✅ INGESTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n✓ Created vector store at: {config_ollama.VECTOR_STORE_DIR}/")
    logger.info(f"✓ Total documents: {len(documents)}")
    logger.info(f"✓ Total chunks: {len(chunks)}")
    logger.info(f"\nYou can now run: python rag_chain.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()