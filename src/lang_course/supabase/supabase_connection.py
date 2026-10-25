
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore):
    from langchain_core.documents import Document

    test_doc = Document(
        page_content="This is a test document to verify Supabase",
        metadata={"test": True}
    )

    ids = []

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Added test document: {ids[0]}")

        results = vectorstore.similarity_search("test documents", k=1)

        if results:
            print(f"Search works: {results[0].page_content}")
        else:
            print("No search results found.")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        if ids:
            vectorstore.delete(ids=ids)
            print("Cleanup complete")


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print("\nConnecting to Supabase...")
    else:
        print("\nSUPABASE_DATABASE_URL not set.")
        print("Using DATABASE_URL or local PostgreSQL fallback.")

    try:
        vectorstore = connect_to_supabase()
        success = verify_connection(vectorstore)

        if success:
            print("\nConnection test completed successfully.")
        else:
            print("\nConnection test failed.")

    except Exception as e:
        print(f"\nFailed to initialize vector store: {e}")


if __name__ == "__main__":
    print("supabase_connection.py is executing")
    main()