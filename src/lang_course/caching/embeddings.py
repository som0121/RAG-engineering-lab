from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from dotenv import load_dotenv
import numpy as np
import os
import tempfile

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Gemini text embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


def basic_embeddings():

    # Single text
    text = "What is Machine Learning?"

    single_embedding = embeddings_model.embed_query(text)

    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():

    texts = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    # Embed multiple documents
    batch_embedding = embeddings_model.embed_documents(texts)

    for i, emb in enumerate(batch_embedding):

        print(f"Text {i + 1} - Vector dimensions: {len(emb)}")
        print(f"Text {i + 1} - First 5 values: {emb[:5]}")
        print(f"Text {i + 1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # Generate embeddings
    doc_vectors = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # Calculate cosine similarity
    def cosine_similarity(vec1, vec2):

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if denominator == 0:
            return 0.0

        return np.dot(vec1, vec2) / denominator

    similarities = [
        cosine_similarity(query_vector, doc_vector)
        for doc_vector in doc_vectors
    ]

    # Rank documents by similarity
    ranked_docs = sorted(
        zip(docs, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Query: {query}\n")
    print("Ranked by similarity:")

    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")


def embedding_caching():

    # Temporary cache directory
    with tempfile.TemporaryDirectory() as tempdir:

        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="gemini_embedding_001",
        )

        text = "What is Reinforcement Learning?"

        # First call: generates the embedding through Gemini
        print("First call (API if not cached):")

        vectors1 = cached_embeddings.embed_documents([text])

        print(f"Embedded {len(vectors1)} document(s)")

        # Second call: retrieves the embedding from cache
        print("\nSecond call (Cache):")

        vectors2 = cached_embeddings.embed_documents([text])

        print(f"Embedded {len(vectors2)} document(s)")

        # Compare the vectors
        print(
            f"\nSame vectors: "
            f"{np.allclose(vectors1[0], vectors2[0])}"
        )


if __name__ == "__main__":

    # Uncomment the function you want to test.

    basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()
