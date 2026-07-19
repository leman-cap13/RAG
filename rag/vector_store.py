import hashlib
import logging
import time

import chromadb
from config import settings

from config import settings

logger = logging.getLogger(__name__)

client = chromadb.PersistentClient(path=settings.chroma_path)
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
    existing = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in existing.get("metadatas") or [] if m.get("source")})


def delete_source(source):
    matches = collection.get(where={"source": source})
    ids = matches.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def query(embedding, top_k=4):
    start = time.perf_counter()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    logger.debug(
        "vector_store_query",
        extra={"top_k": top_k, "results": len(documents), "duration_ms": duration_ms},
    )

    return [
        {
            "text": doc,
            "source": meta.get("source") if meta else None,
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]