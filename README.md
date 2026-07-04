# RAG

A simple Retrieval-Augmented Generation (RAG) pipeline: it chunks `.txt` documents, embeds them with Gemini, stores the vectors in ChromaDB, and answers questions by retrieving relevant chunks and generating a grounded answer with Gemini.

## How it works

1. `main.py` reads every `.txt` file in `data/`
2. Each file is split into chunks (`rag/chunker.py`)
3. Each chunk is embedded via the Gemini embedding model (`rag/embedder.py`)
4. Chunks + embeddings are stored in a local ChromaDB collection (`rag/vector_store.py`)
5. You ask a question in the terminal; the top matching chunks are retrieved and passed to Gemini to generate an answer (`rag/generator.py`)

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
│   ├── chunker.py      # splits text into fixed-size chunks
│   ├── embedder.py     # embeds text via Gemini
│   ├── vector_store.py # ChromaDB storage/query
│   └── generator.py    # generates answers via Gemini
├── main.py             # ingestion + Q&A loop
└── requirements.txt
```
