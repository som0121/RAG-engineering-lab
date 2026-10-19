"""
Text Splitters and Chunking Strategies
Optimizing document chunks for RAG
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

load_dotenv()

# Sample documents for testing

SAMPLE_TEXT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map
inputs to known outputs, then predicts outputs for new, unseen inputs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers
structure in the data on its own, such as groups or reduced dimensions.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Medical diagnosis
"""

SAMPLE_CODE = '''
def quicksort(arr):
    """Sort a list using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """Return the index of target in a sorted list, or -1 if missing."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


class Stack:
    """A simple last-in, first-out stack."""

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
'''


def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators= ["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes; {[len(c) for c in chunks]}")
    print(f"\n First chunk preview:\n {chunks [0][:200]}...")



def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text

    # Without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

    # With overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    chunks_no_overlap = no_overlap.split_text(text)
    chunks_with_overlap = with_overlap.split_text(text)

    print("Without overlap:")
    print(f" Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    print("\nWith overlap:")
    print(f" Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunks_with_overlap[1][:20]}...")

    # Real-world example of why overlap matters
    api_text = (
        "The API key expires after 24 hours. "
        "You must refresh it using the /token endpoint. "
        "Send a POST request with your refresh token to get a new key."
    )

    # Without overlap, this might split as:
    # Chunk 1: "The API key expires after 24 hours."
    # Chunk 2: "You must refresh it using the /token endpoint."
    #
    # User asks: "How do I handle API key expiration?"
    # Chunk 1 matches "expiration" but misses the solution.
    # Chunk 2 has the solution but doesn't mention "expiration".
    #
    # With overlap, chunk 2 would include: "...expires after 24 hours. You must refresh it..."

    
if __name__ == "__main__":
    print("=== Recursive Character Text Splitter ===")
    recursive_splitter()
    overlap_importance()