from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List


def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + 1 + rrf_k))

            if key in doc_scores:
                old_score, _ = doc_scores[key]
                doc_scores[key] = (old_score + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    ranked_docs = sorted(
        doc_scores.values(),
        key=lambda item: item[0],
        reverse=True
    )

    return [doc for score, doc in ranked_docs[:k]]


documents = [
    Document(
        page_content="The UltraBook Pro X15 laptop has a 15-inch OLED display, 32GB RAM and an Intel Core i9 processor.",
        metadata={"id": 1, "type": "product", "category": "laptop", "price": 1899, "source": "catalog"},
    ),
    Document(
        page_content="The SwiftBook Air 13 is a lightweight laptop with 18 hours of battery life, ideal for students.",
        metadata={"id": 2, "type": "product", "category": "laptop", "price": 999, "source": "catalog"},
    ),
    Document(
        page_content="SoundMax ANC-700 wireless headphones offer active noise cancellation and 30-hour battery life.",
        metadata={"id": 3, "type": "product", "category": "audio", "price": 249, "source": "catalog"},
    ),
    Document(
        page_content="The PixelView 4K monitor (model PV-27U) has a 27-inch IPS panel with USB-C power delivery.",
        metadata={"id": 4, "type": "product", "category": "monitor", "price": 449, "source": "catalog"},
    ),
    Document(
        page_content="Error code E-1042 means the device firmware is outdated. Update the firmware from the settings menu.",
        metadata={"id": 5, "type": "support", "category": "troubleshooting", "error_code": "E-1042", "source": "help_center"},
    ),
    Document(
        page_content="If your computer will not turn on, hold the power button for 15 seconds and reconnect the charger.",
        metadata={"id": 6, "type": "support", "category": "troubleshooting", "source": "help_center"},
    ),
]


class HybridRetriever:
    def __init__(
        self,
        documents: List[Document],
        bm25_weight: float = 0.5,
        k: int = 4
    ):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        self.documents = list(documents)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.vector = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name="hybrid_search"
        )

        self.vector_retriever = self.vector.as_retriever(
            search_kwargs={"k": k}
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=k
        )

    def search(self, query: str) -> List[Document]:
        return hybrid_retrieve(
            query,
            retrievers=[
                self.bm25_retriever,
                self.vector_retriever
            ],
            weights=[
                self.bm25_weight,
                self.vector_weight
            ],
            k=self.k
        )

    def add_documents(self, documents: List[Document]):
        self.vector.add_documents(documents)
        self.documents.extend(documents)

        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=self.k
        )


if __name__ == "__main__":
    retriever = HybridRetriever(
        documents,
        bm25_weight=0.5,
        k=4
    )

    results = retriever.search("UltraBook Pro X15 specifications")

    for doc in results:
        print(doc.page_content[:100])
        print(doc.metadata)