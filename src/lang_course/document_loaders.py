import os
import tempfile
import warnings

# langchain-community is being sunset, but TextLoader has no replacement package yet.
# Hide only its sunset warning so the output stays clean.
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning,
)

from langchain_community.document_loaders import(
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():

    #Create a temporary text file for demonstartion

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\n This file is used for demonstration")
        temp_file_path = temp_file.name

    try:
        #Load the text file using TextLoader 
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content [:100]}...")
        print(f"Metadata: {documents[0].metadata}")

        #Print the loaded documents

        #for doc in documents:
            #print(doc.page_content)
            #print(doc)

    finally:
        #Cleanup up the temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    load_text_file()

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded{len(documents)} documents(s) from PDF")

    for i, doc in enumerate(documents):

        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":


    pdf_loader("./Docs/RAG.pdf")