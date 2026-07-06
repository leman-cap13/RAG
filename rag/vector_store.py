import hashlib
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
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


def query(embedding, top_k=4):
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source") if meta else None,
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]



