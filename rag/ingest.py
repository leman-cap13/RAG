from pathlib import Path

from config import settings
from rag.chunker import chunk_text
from rag.embedder import embed_documents
from rag.vector_store import add_chunks, is_source_indexed

def index_all(data_dir=None):
    data_dir = Path(data_dir or settings.data_dir)
    results = []

    for file in data_dir.glob("*.txt"):
        if is_source_indexed(file.name):
            results.append({'file': file.name, 'status': 'skipped', 'chunks': 0})
            continue

        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        embeddings = embed_documents(chunks)
        add_chunks(chunks, embeddings, source=file.name)

        results.append({'file': file.name, 'status': 'indexed', 'chunks': len(chunks)})
    return results