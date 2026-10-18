import tempfile

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from lang_course.common.llm import get_embeddings

COLLECTION_NAME = "test_collection"

DOCUMENTS = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later"},
]

SAMPLE_DOCS = [
    Document(
        page_content="Vector stores save embeddings so you can search text by meaning.",
        metadata={"source": "vector_stores.txt", "topic": "database"},
    ),
    Document(
        page_content="Chroma is an open-source vector database for AI applications.",
        metadata={"source": "chroma.txt", "topic": "database"},
    ),
    Document(
        page_content="Embeddings turn text into lists of numbers that capture meaning.",
        metadata={"source": "embeddings.txt", "topic": "embeddings"},
    ),
    Document(
        page_content="LangChain helps you build apps powered by language models.",
        metadata={"source": "langchain.txt", "topic": "framework"},
    ),
    Document(
        page_content="FAISS is a library from Meta for fast similarity search over dense vectors.",
        metadata={"source": "faiss.txt", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed cloud vector database built for production workloads.",
        metadata={"source": "pinecone.txt", "topic": "database"},
    ),
    Document(
        page_content="PostgreSQL can store and search vectors using the pgvector extension.",
        metadata={"source": "pgvector.txt", "topic": "database"},
    ),
    Document(
        page_content="OpenAI and Google offer embedding models that convert text into vectors.",
        metadata={"source": "embedding_models.txt", "topic": "embeddings"},
    ),
    Document(
        page_content="LlamaIndex is a framework for connecting your own data to language models.",
        metadata={"source": "llamaindex.txt", "topic": "framework"},
    ),
    Document(
        page_content="Large language models like Claude and GPT generate text from prompts.",
        metadata={"source": "llms.txt", "topic": "llm"},
    ),
]


def main() -> None:
    # In-memory client: data is gone when the script ends.
    # Use chromadb.PersistentClient(path="chroma_db") to keep it on disk.
    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Chroma expects lists, so add all documents in one call.
    collection.upsert(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
    )

    query_text = "Hello, World!"
    results = collection.query(query_texts=[query_text], n_results=3)

    # Results are lists of lists: one inner list per query text.
    print(f"Query: {query_text!r}\n")
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"{doc_id}  distance={distance:.4f}  {text}")


def similarity_search_with_scores() -> None:
    embeddings_model = get_embeddings()

    # Chroma keeps its files open, so on Windows the temp folder can't be deleted
    # right away. Ignore that cleanup error instead of crashing.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Create a vector store from the documents.
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        # Lower score = closer match (Chroma returns a distance).
        query = "Explain vector stores"
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"\nTop 3 results with scores for query {query!r}:")
        for i, (doc, score) in enumerate(results_with_scores, start=1):
            print(
                f"Result {i}: {doc.page_content} "
                f"(Score: {score:.4f}, Source: {doc.metadata.get('source')})"
            )


# metadata filtering

def metadata_filtering() -> None:
    embeddings_model = get_embeddings()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available?"

        # without metadata filtering
        results = vectorstore.similarity_search(query, k=5)
        print(f"\nResults without metadata filtering for query '{query}':")

        for i, doc in enumerate(results, start=1):
            print(f"Result {i}: {doc.page_content} (Source: {doc.metadata['source']})")

        # with metadata filtering
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=5, filter=filter_criteria
        )

        print(f"\nResults with metadata filtering for query '{query}':")

        for i, doc in enumerate(filtered_results, start=1):
            print(f"Result {i}: {doc.page_content} (Source: {doc.metadata['source']})")


if __name__ == "__main__":
    main()
    similarity_search_with_scores()
    metadata_filtering()
