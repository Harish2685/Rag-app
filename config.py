"""
Central configuration for the RAG pipeline with OLLAMA (Free, Local Models)
No API costs - everything runs on your machine!
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- OLLAMA Configuration ---
# Make sure Ollama is running: ollama serve
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama port

# --- Paths ---
DATA_DIR = "data"                  # folder where uploaded / source PDFs live
VECTOR_STORE_DIR = "chroma_db"     # persistent on-disk vector store

# --- Chunking ---
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # overlap between chunks to preserve context across boundaries

# --- Embeddings Model ---
# Choose one (all free, run locally):
# - "nomic-embed-text" (Recommended - best quality, ~275MB)
# - "mxbai-embed-large" (Good quality, ~680MB)
# - "snowflake-arctic-embed" (Very good, ~650MB)
EMBEDDING_MODEL = "nomic-embed-text"

# --- Retrieval ---
TOP_K = 4                  # number of chunks retrieved per query
SIMILARITY_SCORE_THRESHOLD = 0.3   # below this, we tell the user "not found in docs"

# --- Generation LLM Model ---
# Choose one (all completely free, run locally):
# 
# Fast & Light (good for quick responses, ~4GB RAM):
# - "mistral" (7B, very fast, good quality)
# - "neural-chat" (7B, optimized for chat)
# - "phi" (2.7B, ultra-fast, basic quality)
#
# Balanced (good quality, ~8-12GB RAM):
# - "llama2" (7B/13B, reliable, good quality)
# - "dolphin-mixtral" (8x7B, very good quality)
#
# High Quality (slower, needs more RAM):
# - "neural-chat:13b" (13B, excellent quality)
# - "mistral:7b-instruct-v0.2" (newest mistral)
#
LLM_MODEL = "mistral"

TEMPERATURE = 0.0          # deterministic, factual answers

# --- Prompt ---
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the
provided context extracted from the user's documents.

Rules:
- Only use information found in the context below to answer.
- If the answer is not present in the context, say "I couldn't find that in the provided documents."
  Do NOT use outside knowledge or make anything up.
- When possible, mention which source/page the information came from.
- Keep answers concise and directly relevant to the question.
"""