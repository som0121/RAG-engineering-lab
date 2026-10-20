# Hybrid search = keyword search (BM25) + semantic search (embeddings)
# Dependencies: langchain-classic, langchain-openai, langchain-chroma, rank-bm25
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


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
    Document(
        page_content="Customers can return any item within 30 days of delivery for a full refund, no questions asked.",
        metadata={"id": 7, "type": "policy", "category": "returns", "source": "faq"},
    ),
    Document(
        page_content="Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
        metadata={"id": 8, "type": "policy", "category": "shipping", "source": "faq"},
    ),
    Document(
        page_content="All laptops include a 2-year limited warranty covering hardware defects and battery failure.",
        metadata={"id": 9, "type": "policy", "category": "warranty", "source": "faq"},
    ),
    Document(
        page_content="Review: The SoundMax ANC-700 blocks out airplane engine noise well, but the ear cushions get warm.",
        metadata={"id": 10, "type": "review", "category": "audio", "rating": 4, "source": "customer_reviews"},
    ),
]

print(f'Loaded {len(documents)} documents')

#Create embeddings and vector store 

embeddings = OpenAIEmbeddings(model= 'text-embedding-3-small')

vectorstore = Chroma.from_documents(
   documents,
   embeddings,
   collection_name= 'hybrid_test'
)

#Create vector retriver

vector_retriever = vectorstore.as_retriever(
   search_kwargs = {'k': 3} # return top 3
)

#BM25 works on the raw text

bm25_retriever = BM25Retriever.from_documents(
   documents,
   k = 3 # returns top 3
)

#Combine with EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
   retrievers=[bm25_retriever, vector_retriever],
   weights=[0.5,0.5] # Equal weight to both
)


print('BM25 retriever ready')

if __name__ == "__main__":
   pass