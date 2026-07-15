from pathlib import Path

from config import settings
from rag.chunker import chunk_text
from rag.embedder import embed_documents
from rag.vector_store import add_chunks, is_source_indexed

def index_data(data_dir: Path = None, reindex: bool = True):
    data_dir = Path(data_dir or settings.data_dir)

    if not data_dir.exists():
        print(f"Warning: data directory '{data_dir}' not found.")
        return
    
    results = []

    if reindex:
        for file in data_dir.glob("*.txt"):
            if is_source_indexed(file.name):
                print(f"{file.name}: already indexed, skipping")
                continue

            text = file.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            embeddings = embed_documents(chunks)
            add_chunks(chunks, embeddings, source=file.name)

            results.append((file.name, len(chunks)))
    


