
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Sample document with distinct topics

document = '''

# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type = client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.

## Error Handling
Our API uses standard HTTP status codes to indicate the success or failure of a request.
A 2xx status code indicates success, a 4xx status code indicates a client error, and a 5xx status code indicates a server error.
Error responses include a message describing the issue and may include an error code.
Check the response body to identify the cause of an error.
For temporary failures, retry the request with an appropriate delay.
Ensure that your request parameters and authentication credentials are valid before retrying.

## Rate Limiting
Our API enforces rate limits to ensure fair usage and maintain service availability.
If you exceed the allowed request limit, the API returns HTTP 429 (Too Many Requests).
When available, check the rate-limit headers in the response for your remaining request quota and reset time.
If a request is rate-limited, wait before sending additional requests.
Use exponential backoff when retrying requests to avoid repeated rate-limit errors.

'''


# Step 1: Initial chunking using RecursiveCharacterTextSpliter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
     separators=["\n\n", "\n", ".", " "]
)

recursive_chunks = recursive_splitter.split_text(document)
print("Recursive Character Splitter Chunks")

print(f"Recursive chunks: {len(recursive_chunks)}")

for i,chunk in enumerate(recursive_chunks):
    print(f"\\n--- Chunk{i+1} ({len(chunk)} chars)----")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

Semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type= 'percentile',
    breakpoint_threshold_amount=90 #Split at 90th percentile dissimilarity
)

semantic_chunks = Semantic_chunker.split_text(document)

print(f"Semantic chunks: {len(semantic_chunks)}")

for i, chunk in enumerate(semantic_chunks):
    print(f"\n---- Chunk{i+1} ({len(chunk)}chars) ---")
    print(chunk[:100]+ "..." if len(chunk) > 100 else chunk)

def print_chunks(chunks, method_name):
    """Print chunks with topic detection and clear formatting."""

    print(f"\n{'=' * 60}")
    print(f" {method_name}")
    print(f" Total chunks: {len(chunks)}")
    print(f"{'=' * 60}")

    for i, chunk in enumerate(chunks):
        topics = detect_topics(chunk)
        topic_label = f" | Topics: {', '.join(topics)}" if topics else ""

        print(f"\n--- Chunk {i + 1} ({len(chunk)} chars){topic_label} ---")

        lines = chunk.strip().split("\n")
        preview = "\n".join(lines[:3])

        if len(lines) > 3:
            preview += "\n..."

        print(preview)
