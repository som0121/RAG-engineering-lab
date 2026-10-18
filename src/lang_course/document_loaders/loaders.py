
import os
import tempfile
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing langchain_community, which reads USER_AGENT on import.
load_dotenv()

# langchain-community is being sunset, but TextLoader has no replacement package yet.
# Hide only its sunset warning so the output stays clean.
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning,
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader  # noqa: E402

# Project root (Lang-course/), so the PDF path works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PDF = PROJECT_ROOT / "Docs" / "RAG.pdf"


def load_text_file() -> None:
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\n This file is used for demonstration")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def pdf_loader(pdf_path: str | Path) -> None:
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")

    for i, doc in enumerate(documents):
        print(f"Document {i + 1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    load_text_file()
    pdf_loader(SAMPLE_PDF)
