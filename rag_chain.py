"""
Core RAG logic with OLLAMA (Free, Local Models)

No API keys needed - everything runs locally on your machine!
Make sure Ollama is running: ollama serve
"""

import os
from dataclasses import dataclass, field
import logging

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured result so the UI can show the answer AND its sources separately."""
    answer: str
    sources: list = field(default_factory=list)
    found_relevant_context: bool = True


class RAGPipeline:
    def __init__(self, persist_directory: str = config.VECTOR_STORE_DIR):
        logger.info("🔧 Initializing RAG Pipeline with Ollama...")
        logger.info(f"   Embedding Model: {config.EMBEDDING_MODEL}")
        logger.info(f"   LLM Model: {config.LLM_MODEL}")
        logger.info(f"   Ollama URL: {config.OLLAMA_BASE_URL}")
        logger.info(f"   Vector Store: {persist_directory}\n")

        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"❌ No vector store found at '{persist_directory}'. "
                f"Run `python ingest_ollama.py` first to build it."
            )

        try:
            logger.info("📥 Initializing Ollama Embeddings...")
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
            test_embed = self.embeddings.embed_query("test")
            logger.info(f"✓ Embeddings initialized ({len(test_embed)} dimensions)")

            logger.info("📚 Loading Chroma vector store...")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            logger.info(f"✓ Vector store loaded")

            logger.info("🧠 Initializing Ollama LLM...")
            self.llm = Ollama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.TEMPERATURE,
                num_predict=512,
            )
            logger.info("✓ LLM initialized")

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", config.SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])
            logger.info("✓ RAG Pipeline ready!\n")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Pipeline: {str(e)}")
            logger.error("\n⚠️  Make sure Ollama is running:")
            logger.error("   1. Download from: https://ollama.ai")
            logger.error("   2. Run: ollama serve")
            logger.error("   3. Pull models:")
            logger.error(f"      ollama pull {config.EMBEDDING_MODEL}")
            logger.error(f"      ollama pull {config.LLM_MODEL}")
            raise

    def retrieve(self, question: str, k: int = config.TOP_K):
        """Retrieve relevant chunks from vector store."""
        results = self.vector_store.similarity_search_with_score(question, k=k)      
        return results

    def format_context(self, retrieved_docs):
        """Format retrieved documents into a readable context block."""
        context_blocks = []
        for doc, score in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            context_blocks.append(
                f"[Source: {os.path.basename(source)}, Page: {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_blocks)

    def ask(self, question: str) -> RAGResponse:
        """Answer a question using RAG."""
        logger.info(f"📝 Question: {question}")

        retrieved = self.retrieve(question)

        if not retrieved:
            logger.warning("⚠️  No documents retrieved")
            return RAGResponse(
                answer="I couldn't find that in the provided documents.",
                sources=[],
                found_relevant_context=False,
            )

        relevant = [
            (doc, score) for doc, score in retrieved 
            if score >= config.SIMILARITY_SCORE_THRESHOLD
        ]

        if not relevant:
            logger.warning(f"⚠️  No relevant chunks found")
            return RAGResponse(
                answer="I couldn't find that in the provided documents.",
                sources=[],
                found_relevant_context=False,
            )

        logger.info(f"✓ Retrieved {len(relevant)} relevant chunks")
        context = self.format_context(relevant)

        try:
            logger.info("🧠 Generating answer...")
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            logger.info("✓ Answer generated")
        except Exception as e:
            logger.error(f"❌ LLM call failed: {str(e)}")
            raise

        sources = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:250] + 
                          ("..." if len(doc.page_content) > 250 else ""),
                "score": round(score, 3),
            }
            for doc, score in relevant
        ]

        return RAGResponse(answer=answer, sources=sources, found_relevant_context=True)


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
        while True:
            q = input("\n🤔 Ask a question (or 'quit'): ").strip()
            if q.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not q:
                continue
            
            response = pipeline.ask(q)
            print(f"\n💬 Answer:\n{response.answer}")
            
            if response.sources:
                print("\n📚 Sources:")
                for s in response.sources:
                    print(f"  • {s['source']} (page {s['page']}, relevance: {s['score']})")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise