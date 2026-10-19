from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lang_course.common.llm import get_embeddings, get_model

# The text our RAG system answers questions from.
KNOWLEDGE_BASE = """
# LangChain

## What is LangChain?
LangChain is an open-source framework for building applications powered by
large language models (LLMs). It gives you standard building blocks such as
chat models, prompt templates, output parsers, document loaders, text
splitters, embeddings, vector stores and retrievers, so you can connect an LLM
to your own data and tools without writing everything from scratch.
LangChain is available for Python and JavaScript/TypeScript.

## Who created LangChain?
LangChain was created by Harrison Chase and released as an open-source project
in October 2022. The company behind it, LangChain Inc., was co-founded by
Harrison Chase and Ankush Gola.

## LangChain Expression Language (LCEL)
LCEL is the way to chain components together in LangChain using the pipe
operator (|). For example: prompt | llm | StrOutputParser(). Every component is
a "Runnable", so chains support invoke, batch and stream out of the box.

## What is LangGraph used for?
LangGraph is a library from the LangChain team for building stateful,
multi-step and multi-agent applications. You describe your app as a graph:
nodes are steps (such as calling an LLM or a tool) and edges decide what runs
next. LangGraph supports loops, branching, memory that persists between steps,
and human-in-the-loop approval, which makes it a good fit for AI agents.

## What is LangSmith?
LangSmith is a platform for tracing, debugging, testing and monitoring LLM
applications. It records every step of a chain or agent run so you can see
inputs, outputs, latency and token usage.

## Retrieval-Augmented Generation (RAG) in LangChain
RAG lets an LLM answer questions using your own documents. The steps are:
load documents, split them into chunks, embed the chunks, store them in a
vector store such as Chroma or FAISS, retrieve the most relevant chunks for a
question, and pass them to the LLM as context.
"""

SOURCE_NAME = "langchain_knowledge_base.md"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "context below. Answer in 1-3 sentences. If the context does not "
            "contain the answer, reply exactly: I don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def create_kb() -> Chroma:
    """Create a vector store from the knowledge base."""
    # Split on markdown headings first, so each chunk keeps one topic together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": SOURCE_NAME})
    chunks = splitter.split_documents([doc])

    # Number each chunk so answers can point to where they came from.
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = i

    # No persist_directory: the store lives in memory and disappears when the
    # script ends. This avoids leftover temp folders on Windows.
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="langchain_kb",
    )


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string, labelled by source."""
    return "\n\n".join(
        f"[{doc.metadata['source']} #{doc.metadata['chunk']}]\n{doc.page_content}"
        for doc in docs
    )


def demo_basic_rag(vector_store: Chroma) -> None:
    """Question in, answer out."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = get_model(temperature=0)  # 0 = stick to the facts, no creativity

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
        "What is the capital of France?",  # not in the knowledge base -> "I don't know"
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_sources(vector_store: Chroma) -> None:
    """Return the answer together with the chunks it was based on."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = get_model(temperature=0)

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 1: fetch docs and keep the question.
    # Step 2: add an "answer" key, built from those same docs.
    rag_chain = RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    ).assign(answer=answer_chain)

    print("RAG with Sources Demo:\n")
    question = "What is LCEL and how do you chain components?"
    result = rag_chain.invoke(question)

    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print("Sources:")
    for doc in result["docs"]:
        preview = doc.page_content.strip().splitlines()[0]
        print(f"  - {doc.metadata['source']} #{doc.metadata['chunk']}: {preview}")


if __name__ == "__main__":
    kb = create_kb()  # kb = knowledge base; build once, reuse in both demos
    demo_basic_rag(kb)
    demo_rag_with_sources(kb)
