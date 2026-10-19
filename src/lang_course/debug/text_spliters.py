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

    if __name__ == "main":
        print("=== Recursive Character Text Spliter ===")
        recursive_splitter()