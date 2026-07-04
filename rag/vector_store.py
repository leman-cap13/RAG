import uuid
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("documents")

def add_chunks(chunks, embeddings, source):
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)

def query(embedding, top_k=4):
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    documents = results.get("documents") or []
    return documents[0] if documents else []