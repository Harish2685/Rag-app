"""
Core RAG logic: given a user question, retrieve relevant chunks from the
vector store and generate a grounded answer using an LLM.

Kept separate from the UI (app.py) so this module can also be unit-tested
or reused in a CLI / API context.
"""

import os
from dataclasses import dataclass, field

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config


@dataclass
class RAGResponse:
    """Structured result so the UI can show the answer AND its sources separately."""
    answer: str
    sources: list = field(default_factory=list)   # list of dicts: {source, page, snippet, score}
    found_relevant_context: bool = True


class RAGPipeline:
    def __init__(self, persist_directory: str = config.VECTOR_STORE_DIR):
        if not config.NVIDIA_API_KEY:
            raise EnvironmentError(
                "NVIDIA_API_KEY not set. Copy .env.example to .env and add your key."
            )

        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"No vector store found at '{persist_directory}'. "
                f"Run `python ingest.py` first to build it."
            )

        self.embeddings = NVIDIAEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.NVIDIA_API_KEY,
        )
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.llm = ChatNVIDIA(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.NVIDIA_API_KEY,
        )
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", config.SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
        

    def retrieve(self, question: str, k: int = config.TOP_K):
        results = self.vector_store.similarity_search_with_score(question, k=k)      
        return results  # list of (Document, score)

    def format_context(self, retrieved_docs):
        context_blocks = []
        for doc, score in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            context_blocks.append(f"[Source: {os.path.basename(source)}, Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(context_blocks)

    def ask(self, question: str) -> RAGResponse:
        retrieved = self.retrieve(question)

        if not retrieved:
            return RAGResponse(
                answer="I couldn't find that in the provided documents.",
                sources=[],
                found_relevant_context=False,
            )

        # Filter out weakly-relevant chunks so we don't feed noise to the LLM
        relevant = [(doc, score) for doc, score in retrieved if score >= config.SIMILARITY_SCORE_THRESHOLD]

        if not relevant:
            return RAGResponse(
                answer="I couldn't find that in the provided documents.",
                sources=[],
                found_relevant_context=False,
            )

        context = self.format_context(relevant)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        sources = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else ""),
                "score": round(score, 3),
            }
            for doc, score in relevant
        ]

        return RAGResponse(answer=answer, sources=sources, found_relevant_context=True)


if __name__ == "__main__":
    # Quick CLI smoke test
    pipeline = RAGPipeline()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in ("quit", "exit"):
            break
        response = pipeline.ask(q)
        print(f"\nAnswer: {response.answer}")
        if response.sources:
            print("\nSources:")
            for s in response.sources:
                print(f"  - {s['source']} (page {s['page']}, score {s['score']})")
