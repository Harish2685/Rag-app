"""
Streamlit UI for the RAG PDF Chatbot.

Run with:
    streamlit run app.py

Features:
- Upload one or more PDFs, which get chunked + embedded on the fly
- Chat interface for asking questions
- Displays grounded answers with expandable source citations (file, page, snippet, score)
- Falls back gracefully with "not found" instead of hallucinating
"""

import os
import tempfile

import streamlit as st
import os
from pathlib import Path

import config
from rag_chain import RAGPipeline

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("📄 RAG PDF Chatbot")
st.caption("Upload documents, ask questions, get answers grounded in your own content — with citations.")

# --- Session state setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...]}
if "pipeline" not in st.session_state:
    with st.spinner("🔧 Initializing RAG Pipeline..."):
        try:
            st.session_state.pipeline = RAGPipeline()
            st.success("✅ Pipeline ready!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: upload + status ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check vector store
    if os.path.exists(config.VECTOR_STORE_DIR):
        st.success("✅ Vector store ready")
    else:
        st.error("❌ Vector store not found. Run ingest_ollama.py first")
    
    st.markdown("---")
    st.markdown(f"**Model:** {config.LLM_MODEL}")
    st.markdown(f"**Embedding:** {config.EMBEDDING_MODEL}")
    st.markdown(f"**Ollama URL:** {config.OLLAMA_BASE_URL}")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# --- Main chat interface ---
pipeline = get_pipeline()

if pipeline is None:
    st.info("👈 Upload and ingest at least one PDF from the sidebar to get started.")
else:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source']}** — page {s['page']} "
                            f"(relevance: {s['score']})\n\n> {s['snippet']}"
                        )

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                response = st.session_state.pipeline.ask(user_input)
                
                # Display answer
                st.markdown(response.answer)
                
                # Display sources
                if response.sources:
                    with st.expander("📚 Sources"):
                        for source in response.sources:
                            st.markdown(f"""
                            **{source['source']}** (Page {source['page']}, Relevance: {source['score']})
                            
                            {source['snippet']}
                            """)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })