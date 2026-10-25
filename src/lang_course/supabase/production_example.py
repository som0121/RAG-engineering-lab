import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


@dataclass
class Config:
    database_url: str = field(
        default_factory=lambda: (
            os.getenv("SUPABASE_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
        )
    )

    collection_name: str = "production_documents_gemini"
    embedding_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-3.8-flash"
    temperature: float = 0.0
    default_k: int = 5
    min_similarity: float = 0.5


class RAGService:
    """RAG service using Gemini, LangChain, and PostgreSQL pgvector."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chain = None

    @property
    def vectorstore(self) -> PGVector:
        """Lazily initialize the vector store."""
        if self._vectorstore is None:
            if not self.config.database_url:
                raise ValueError("Database URL is not configured.")

            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY is not configured.")

            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
            )

            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True,
            )

        return self._vectorstore

    @property
    def chain(self):
        """Lazily initialize the RAG chain."""
        if self._chain is None:
            self._chain = self._create_chain()

        return self._chain

    def _create_chain(self):
        """Create the retrieval and generation chain."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k}
        )

        llm = ChatGoogleGenerativeAI(
            model=self.config.chat_model,
            temperature=self.config.temperature,
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant that answers questions using retrieved documents.

Use the context below to answer the question accurately.
Do not invent facts that are not supported by the context.

If the context does not contain enough information, say:
"I don't have enough information to answer that question."

Context:
{context}

Question:
{question}

Answer concisely and accurately.
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        if not documents:
            return []

        return self.vectorstore.add_documents(documents)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Search for relevant documents and return distance scores."""
        search_kwargs = {
            "k": k if k is not None else self.config.default_k
        }

        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )

    def ask(self, question: str) -> str:
        """Answer a question using retrieved context."""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """Return an answer and the documents retrieved for the question."""
        docs_with_scores = self.search(question)
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": float(score),
                }
                for doc, score in docs_with_scores
            ],
        }


def create_sample_documents() -> list[Document]:
    """Create sample documents for testing RAG."""
    return [
        Document(
            page_content=(
                "LangChain is a framework for developing applications "
                "powered by language models. It provides integrations "
                "for models, prompts, retrievers, and chains."
            ),
            metadata={
                "source": "langchain_intro",
                "topic": "langchain",
            },
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation, also called RAG, "
                "retrieves relevant information from a knowledge base "
                "and provides it to a language model as context "
                "when answering a question."
            ),
            metadata={
                "source": "rag_intro",
                "topic": "rag",
            },
        ),
        Document(
            page_content=(
                "Supabase provides a PostgreSQL database. The pgvector "
                "extension enables storing vector embeddings and "
                "performing similarity searches over those vectors."
            ),
            metadata={
                "source": "supabase_intro",
                "topic": "supabase",
            },
        ),
        Document(
            page_content=(
                "Gemini is Google's family of generative AI models. "
                "Gemini models can generate text, while embedding "
                "models convert text into numerical vectors."
            ),
            metadata={
                "source": "gemini_intro",
                "topic": "gemini",
            },
        ),
    ]


def main():
    print("=" * 60)
    print("Gemini RAG Service Test")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is missing from the environment.")
        return

    try:
        service = RAGService()

        print("\nStep 1: Initializing vector store...")
        _ = service.vectorstore
        print("Vector store initialized.")

        print("\nStep 2: Adding sample documents...")
        documents = create_sample_documents()
        ids = service.add_documents(documents)
        print(f"Added {len(ids)} documents.")

        print("\nStep 3: Testing similarity search...")
        query = "What is Retrieval-Augmented Generation?"
        results = service.search(query, k=3)

        if results:
            for index, (doc, distance) in enumerate(results, start=1):
                print(f"\nResult {index}")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                print(f"Distance: {distance:.4f}")
        else:
            print("No matching documents found.")

        print("\nStep 4: Testing RAG question answering...")
        question = "How does RAG work?"
        answer = service.ask(question)

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

        print("\nStep 5: Testing answer with sources...")
        result = service.ask_with_sources(
            "What does Supabase pgvector do?"
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for index, source in enumerate(result["sources"], start=1):
            print(f"\nSource {index}")
            print(f"Content: {source['content']}")
            print(f"Metadata: {source['metadata']}")
            print(f"Distance: {source['distance']:.4f}")

        print("\nRAG test completed.")

    except Exception as e:
        print(f"\nRAG test failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()