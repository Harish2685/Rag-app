"""
Central configuration for the RAG pipeline.
Keeping all tunables in one place makes the project easy to explain in interviews
and easy to experiment with (chunk size, k, model choice, etc.)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# --- Paths ---
DATA_DIR = "data"                  # folder where uploaded / source PDFs live
VECTOR_STORE_DIR = "chroma_db"     # persistent on-disk vector store

# --- Chunking ---
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # overlap between chunks to preserve context across boundaries

# --- Embeddings ---
EMBEDDING_MODEL ="nvidia/nemotron-3-embed-1b"  # cheap + good quality OpenAI embedding model

# --- Retrieval ---
TOP_K = 4                  # number of chunks retrieved per query
SIMILARITY_SCORE_THRESHOLD = 0.3   # below this, we tell the user "not found in docs" instead of guessing

# --- Generation ---
LLM_MODEL = "meta-llama/Llama-3.2-3b-instruct" # swap for "gpt-4o" for higher quality, or a local/open model
TEMPERATURE = 0.0          # deterministic, factual answers — important for a RAG QA system

# --- Prompt ---
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the
provided context extracted from the user's documents.

Rules:
- Only use information found in the context below to answer.
- If the answer is not present in the context, say "I couldn't find that in the provided documents."
  Do NOT use outside knowledge or make anything up.
- When possible, mention which source/page the information came from.
- Keep answers concise and directly relevant to the question.

Context:
{context}
"""
 