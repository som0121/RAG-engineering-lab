from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

def basic_embeddings():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    text = [
        "What is Machine learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embeddings = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

if __name__ == "__main__":
    basic_embeddings()
    batch_embeddings()
