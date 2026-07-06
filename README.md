# RAG

A Retrieval-Augmented Generation (RAG) pipeline: it chunks `.txt` documents, embeds them with Gemini, stores the vectors in ChromaDB, and answers questions by retrieving relevant chunks and generating a grounded, cited answer with Gemini.

## How it works

1. `main.py` reads every `.txt` file in `data/`, skipping files that are already indexed
2. Each file is split into sentence-aware, overlapping chunks (`rag/chunker.py`)
3. Chunks are embedded in batches via the Gemini embedding model, using the `RETRIEVAL_DOCUMENT` task type (`rag/embedder.py`)
4. Chunks + embeddings + source metadata are upserted into a local ChromaDB collection with deterministic IDs, so re-running never duplicates data (`rag/vector_store.py`)
5. You ask a question in the terminal; it's embedded with the `RETRIEVAL_QUERY` task type, the top matching chunks are retrieved, and passed to Gemini to generate a cited answer that's grounded in the sources — the model is instructed to say so if the answer isn't in the retrieved context (`rag/generator.py`)

## Requirements

- Python 3.10+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

## Setup

```bash
git clone https://github.com/leman-cap13/RAG.git
cd RAG

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

Add your `.txt` documents to the `data/` folder, then run:

```bash
python main.py
```

On first run it will chunk, embed, and index all files in `data/`. You'll then be prompted to ask questions:

```
question (output: q): what does this document say about X?
```

Type `q` to quit.

## Project structure

```
.
├── data/               # source .txt documents to index
├── rag/
│   ├── chunker.py      # sentence-aware chunking with overlap
│   ├── embedder.py     # batched embeddings via Gemini (document/query task types)
│   ├── vector_store.py # ChromaDB storage/query with metadata + dedup
│   └── generator.py    # generates cited, grounded answers via Gemini
├── main.py             # ingestion + Q&A loop
└── requirements.txt
```
