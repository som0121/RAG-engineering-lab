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

print("BM25 retriever ready")

#Hybrid retrieval using Reciprocal Rank Fusion (RPF)
#This is what EnsembleRetriever did internally 

def hybrid_retriever(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion"""

    doc_scores = {} #page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank+rrf_k))

            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)

            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse= True)
    return [doc for _, doc in sorted_docs[:k]]
#Combine with EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
   retrievers=[bm25_retriever, vector_retriever],
   weights=[0.5,0.5] # Equal weight to both
)

print('Hybrid retriever ready')

def test_query(query, name, retriever):

    '''Test a query and show results'''

    results = retriever.invoke(query)
    print(f'\\n{name} - Query:\"{query}\"')

    for i,doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'{i+1}.{preview}')

    return results

    #Test queries designed to challenge vector search

test_queries = [

        'SKU-7742X specifications', #Exact product code
        'E_CONN_REFUSED error', #Error code
        'How do I authenticate',#Semantic question
        'WCAG compliance',  #Acronym
        'router configuration'  #General semantic
    ]

for query in test_queries:
    print('=' *60)

    #Vector only 

    vector_results = test_query(query, 'VECTOR', vector_retriever)

    #BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    #Hybrid
    hybrid_results = test_query(query, 'HYBRID', EnsembleRetriever)
    
    
if __name__ == "__main__":
   pass