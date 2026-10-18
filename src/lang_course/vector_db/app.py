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
        metadata={"source": "vector_stores.txt"},
    ),
    Document(
        page_content="Chroma is an open-source vector database for AI applications.",
        metadata={"source": "chroma.txt"},
    ),
    Document(
        page_content="Embeddings turn text into lists of numbers that capture meaning.",
        metadata={"source": "embeddings.txt"},
    ),
    Document(
        page_content="LangChain helps you build apps powered by language models.",
        metadata={"source": "langchain.txt"},
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


if __name__ == "__main__":
    main()
    similarity_search_with_scores()
