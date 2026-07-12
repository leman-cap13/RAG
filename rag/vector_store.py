import hashlib
from pathlib import Path

import chromadb
import numpy as np

from config import settings

DB_PATH = settings.chroma_path
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("documents")


def _chunk_id(source, index):
    return hashlib.sha256(f"{source}::{index}".encode("utf-8")).hexdigest()

def generateUUID(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()

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
    existing = collection.get(where={"source": source})
    return bool(existing.get("ids"))


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
                "uuid": generateUUID(doc) 
            }
        )

    return chunks



def list_sources():
    existing = collection.get(include=["metadatas"])
    return sorted({meta.get("source") for meta in (existing.get("metadatas") or []) if meta.get("source")})

def delete_source(source):
    existing = collection.get(where={"source": source}, include=["ids"])
    ids_to_delete = existing.get("ids") or []
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete), ids_to_delete
