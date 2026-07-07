import hashlib
from pathlib import Path

import chromadb
import numpy as np

DB_PATH = Path("chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("documents")


def _chunk_id(source, index):
    return hashlib.sha256(f"{source}::{index}".encode("utf-8")).hexdigest()


def _cosine_similarity(query_embedding, document_embedding):
    query_vector = np.asarray(query_embedding, dtype=np.float64)
    document_vector = np.asarray(document_embedding, dtype=np.float64)
    if query_vector.size == 0 or document_vector.size == 0:
        return 0.0

    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)
    if query_norm == 0 or document_norm == 0:
        return 0.0

    return float(np.dot(query_vector, document_vector) / (query_norm * document_norm))


def is_source_indexed(source):
    existing = collection.get(where={"source": source}, limit=1)
    return len(existing.get("ids") or []) > 0


def add_chunks(chunks, embeddings, source):
    ids = [_chunk_id(source, i) for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def query(embedding, top_k=4):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    embeddings = (results.get("embeddings") or [[]])[0]

    chunks = []
    for doc, meta, dist, doc_emb in zip(documents, metadatas, distances, embeddings):
        similarity = _cosine_similarity(embedding, doc_emb) if doc_emb is not None else 0.0
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source") if meta else None,
                "distance": dist,
                "similarity": similarity,
            }
        )

    return chunks



