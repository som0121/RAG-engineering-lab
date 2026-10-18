import chromadb

COLLECTION_NAME = "test_collection"

DOCUMENTS = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later"},
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


if __name__ == "__main__":
    main()
