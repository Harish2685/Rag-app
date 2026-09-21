# RAG PDF Chatbot

A chatbot that answers questions about your PDF documents. Upload your files and ask questions — the bot finds answers in your documents and shows you exactly where it found them.



## How It Works

1. Upload PDF files
2. The system reads and splits them into chunks
3. Chunks are converted to embeddings and stored in a database
4. When you ask a question, the system finds relevant chunks and sends them to an LLM
5. The LLM answers based ONLY on those chunks
6. You get the answer + the exact source

## What You Need

- Python 3.8+
- Ollama (free, from https://ollama.ai)
- At least 4GB RAM

## Setup

### 1. Install Ollama
Download from https://ollama.ai and install.

### 2. Clone and Setup
```bash
cd rag-pdf-chatbot
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 3. Start Ollama
```bash
ollama serve
```

### 4. Download Models
In another terminal:
```bash
ollama pull nomic-embed-text
ollama pull mistral
```

### 5. Add Your PDFs
Create a `data/` folder and add your PDF files.

### 6. Prepare Data
```bash
python ingest.py
```

### 7. Run the App
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How to Use

1. Type your question in the chat
2. The bot searches your documents
3. You get an answer with sources

Example questions:
- "What does this document say about X?"
- "How do I do Y according to the manual?"
- "Find the section about Z"

## Tech Stack

- **LangChain** - connects everything together
- **ChromaDB** - stores document embeddings locally
- **Ollama** - runs AI models on your computer (free)
- **Streamlit** - simple web interface

## Models Used

- **Embedding**: `nomic-embed-text` (converts text to numbers for searching)
- **LLM**: `mistral` (answers questions)

Both run locally, no internet needed after first download.

## Change Models

Edit `config.py`:

```python
# Faster (2-5 seconds)
LLM_MODEL = "phi"

# Balanced (5-10 seconds)  
LLM_MODEL = "mistral"

# Better quality (slower)
LLM_MODEL = "neural-chat"
```

Download the model:
```bash
ollama pull mistral
```


## Common Issues

**"Connection refused"**
- Make sure `ollama serve` is running in another terminal

**"Out of memory"**
- Use a smaller model: `phi` or `neural-chat`

**Slow responses**
- Normal for first response. Subsequent ones are faster.
- Use a smaller model if too slow

**"Vector store not found"**
- Run `python ingest.py` to process your PDFs

## Tips

- The bot only uses information from your documents
- It won't make things up (unlike ChatGPT)
- Source citations show you exactly where answers come from
- Add more PDFs anytime and run `ingest.py` again

## No Costs

Everything runs on your machine. No API keys, no subscriptions, no fees.

---

Built with LangChain, ChromaDB, Ollama, and Streamlit.