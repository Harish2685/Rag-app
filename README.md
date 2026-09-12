# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in your own PDF documents — with source citations — instead of relying on an LLM's raw (and sometimes outdated or hallucinated) knowledge.

Upload any PDFs (manuals, papers, policy docs, reports), ask questions in plain English, and get answers backed by the exact source page and snippet they came from.



## 🧠 How it works


 PDF files                 Vector Store (Chroma)              LLM
┌───────────┐   chunk    ┌───────────────────────┐  top-k   ┌─────────────┐
│  data/*.pdf│──────────▶│  embeddings per chunk  │─────────▶│NVIDIA NIM model │──▶ Grounded answer
└───────────┘  + embed   └───────────────────────┘  chunks  └─────────────┘        + sources

## 🧩 Tech Stack

- LangChain — orchestration (loaders, splitters, prompt chains)
- ChromaDB — local vector database
- Streamlit— chat UI
- NVIDIA NIM API — embeddings (`nvidia/nemotron-3-embed-1b`) + generation (`meta-llama/Llama-3.2-3b-instruct`)



