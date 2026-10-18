# lang-course

LangChain course exercises, one folder per lesson.

## Folder structure

```
Lang-course/
├── Docs/                     sample files used by the lessons (RAG.pdf)
├── src/lang_course/
│   ├── __init__.py           marks lang_course as a package (required by the build tool)
│   ├── common/
│   │   └── llm.py            shared helpers: get_model(), get_embeddings()
│   ├── setup_check/
│   │   └── check_models.py   confirms your API keys work
│   ├── document_loaders/
│   │   └── loaders.py        load text files and PDFs
│   └── vector_db/
│       └── app.py            Chroma vector database basics
├── .env.example              copy to .env and fill in your keys
└── pyproject.toml
```

New lesson: add a folder under `src/lang_course/` with a `.py` file in it.
Import shared code with `from lang_course.common.llm import get_model`.

## Setup

```
uv sync
copy .env.example .env    # then fill in your keys
```

## Running a lesson

Run from the project root with `uv run` so the project's `.venv` is used
(plain `python` uses your system Python, which doesn't have the packages):

```
uv run python -m lang_course.setup_check.check_models
uv run python -m lang_course.document_loaders.loaders
uv run python -m lang_course.vector_db.app
```
