"""
Streamlit UI for RAG Chatbot with Ollama (Free, Local Models)
"""

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

st.title("📚 RAG Chatbot - Ollama Edition")
st.markdown("Ask questions about your documents! (100% Free, runs locally)")

# Initialize session state
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

# Sidebar
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

# Chat display
st.markdown("### 💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("🤔 Ask a question about your documents...")

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