<<<<<<< HEAD
# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **grounded in your own PDF documents** — with source citations — instead of relying on an LLM's raw (and sometimes outdated or hallucinated) knowledge.

Upload any PDFs (manuals, papers, policy docs, reports), ask questions in plain English, and get answers backed by the exact source page and snippet they came from.

---

## 🧠 How it works

```
 PDF files                 Vector Store (Chroma)              LLM
┌───────────┐   chunk    ┌───────────────────────┐  top-k   ┌─────────────┐
│  data/*.pdf│──────────▶│  embeddings per chunk  │─────────▶│  gpt-4o-mini │──▶ Grounded answer
└───────────┘  + embed   └───────────────────────┘  chunks  └─────────────┘        + sources
```

1. **Ingest** — PDFs are loaded, split into overlapping text chunks, and embedded using OpenAI embeddings.
2. **Store** — Chunks + embeddings are persisted in a local **ChromaDB** vector store.
3. **Retrieve** — When a user asks a question, it's embedded and compared against stored chunks via similarity search (top-k).
4. **Generate** — The retrieved chunks are injected into a prompt that instructs the LLM to answer *only* from that context.
5. **Cite** — The UI shows which document/page/snippet backed each answer, and a relevance score.
6. **Guardrail** — If no chunk is relevant enough (below a similarity threshold), the bot says so instead of guessing.

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Streamlit UI — chat interface, file upload, source display
├── ingest.py           # Document loading, chunking, embedding, vector store creation
├── rag_chain.py         # Core RAG logic: retrieval + prompt + generation (RAGPipeline class)
├── config.py            # All tunable settings (chunk size, top-k, models, prompt template)
├── requirements.txt      # Python dependencies
├── .env.example          # Template for your OpenAI API key
├── data/                 # Put source PDFs here for batch ingestion (optional)
└── chroma_db/            # Auto-created persistent vector store (git-ignored)
```

---

## 🚀 Setup

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd rag-pdf-chatbot
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
```bash
cp .env.example .env
# then edit .env and paste your key:
# OPENAI_API_KEY=sk-...
```

### 3. Run it

**Option A — Use the UI (recommended):**
```bash
streamlit run app.py
```
Then upload PDFs directly from the sidebar and start chatting.

**Option B — Batch-ingest PDFs from a folder first:**
```bash
# Drop PDFs into the data/ folder, then:
python ingest.py
# Then launch the chat UI:
streamlit run app.py
```

**Option C — Quick CLI test (no UI):**
```bash
python rag_chain.py
```

---

## ⚙️ Configuration

All key parameters live in `config.py`:

| Setting | Purpose | Default |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Controls how documents are split | 1000 / 150 |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `TOP_K` | Number of chunks retrieved per query | 4 |
| `SIMILARITY_SCORE_THRESHOLD` | Minimum relevance to trust a chunk | 0.3 |
| `LLM_MODEL` | Generation model | `gpt-4o-mini` |
| `TEMPERATURE` | Generation randomness | 0.0 (deterministic) |

---

## 🧩 Tech Stack

- **LangChain** — orchestration (loaders, splitters, prompt chains)
- **ChromaDB** — local vector database
- **OpenAI API** — embeddings (`text-embedding-3-small`) + generation (`gpt-4o-mini`)
- **Streamlit** — chat UI
- **PyPDF** — PDF parsing

---

## 🔮 Possible extensions (good talking points for interviews)

- Swap OpenAI embeddings for a free local model (`sentence-transformers/all-MiniLM-L6-v2`) to run fully offline
- Add conversational memory (multi-turn follow-up questions using chat history in the retriever)
- Support more file types (docx, txt, HTML) via `unstructured`
- Add a reranker (e.g., Cohere rerank or cross-encoder) after initial retrieval for higher precision
- Swap Chroma for a production vector DB (Pinecone, Weaviate, pgvector) for scale
- Add evaluation: track retrieval precision/recall and hallucination rate over a test question set
- Deploy via Docker + host on Streamlit Community Cloud / Render / AWS

---

## 📝 Notes

- The vector store persists to disk (`chroma_db/`), so you don't need to re-embed documents every run — only when you add new files.
- The system prompt explicitly forbids the LLM from answering outside the provided context, which is the core "anti-hallucination" mechanism of RAG.
- Relevance scores shown in the UI let you sanity-check *why* the bot answered the way it did — useful for debugging and for demoing to reviewers/interviewers.
=======
# Rag-app
Built a Retrieval-Augmented Generation (RAG) pipeline that grounds LLM responses in user-supplied documents, reducing hallucination risk through similarity-thresholded retrieval and citation-backed answers — using LangChain, ChromaDB, and NVIDIA NIM.
>>>>>>> 4cdb238d942a12602a7b3455af61f46045256824
