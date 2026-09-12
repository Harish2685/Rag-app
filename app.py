
import os
import tempfile

import streamlit as st

import config
from ingest import ingest_single_file
from rag_chain import RAGPipeline

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG PDF Chatbot")
st.caption("Upload documents, ask questions, get answers grounded in your own content — with citations.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


def get_pipeline():
    """Lazily (re)load the pipeline once documents exist."""
    if st.session_state.pipeline is None and os.path.exists(config.VECTOR_STORE_DIR):
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline



with st.sidebar:
    st.header("📁 Documents")

    if not config.NVIDIA_API_KEY:
        st.error("NVIDIA_API_KEY not set. Add it to your .env file.")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Ingest uploaded files"):
        with st.spinner("Chunking and embedding documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.ingested_files:
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                ingest_single_file(tmp_path)
                st.session_state.ingested_files.append(uploaded_file.name)
                os.remove(tmp_path)

            # Force pipeline reload to pick up new vector store content
            st.session_state.pipeline = RAGPipeline()
        st.success(f"Ingested {len(uploaded_files)} file(s).")

    if st.session_state.ingested_files:
        st.subheader("Indexed files")
        for f in st.session_state.ingested_files:
            st.write(f"✅ {f}")

    st.divider()
    st.subheader("⚙️ Settings")
    st.write(f"**Embedding model:** {config.EMBEDDING_MODEL}")
    st.write(f"**LLM:** {config.LLM_MODEL}")
    st.write(f"**Top-K retrieved chunks:** {config.TOP_K}")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()



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

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                response = pipeline.ask(question)
                st.markdown(response.answer)
                if response.sources:
                    with st.expander("📚 Sources"):
                        for s in response.sources:
                            st.markdown(
                                f"**{s['source']}** — page {s['page']} "
                                f"(relevance: {s['score']})\n\n> {s['snippet']}"
                            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })
