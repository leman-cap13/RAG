import hashlib
import chromadb

from config import settings

client = chromadb.PersistentClient(path=settings.chrome_path)
collection = client.get_or_create_collection("documents")


def _chunk_id(source, index):
    return hashlib.sha256(f"{source}::{index}".encode("utf-8")).hexdigest()


def is_source_indexed(source):
    existing = collection.get(where={"source": source}, limit=1)
    return len(existing.get("ids") or []) > 0


def add_chunks(chunks, embeddings, source):
    ids = [_chunk_id(source, i) for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def list_sources():
    existing = collection.get(include = ["metadatas"])
    return sorted({m['source'] for m in existing.get("metadatas") or [] if m.get("source")})

def delete_source(source):
    matches = collection.get(where={"source": source})
    ids = matches.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


