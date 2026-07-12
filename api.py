from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.chunker import chunk_text
from rag.embedder import embed_query
from rag.generator import generate_answer
from rag.vector_store import  query, list_sources, delete_source
from config import settings
from rag.ingest import index_data

app = FastAPI(title="RAG", version="9.32.6 debug", description = "salam")

class AskRequest(BaseModel):
    question: str
    top_k: int = settings.top_k

class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float
    similarity: float
    uuid: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    context: list[SourceChunk]

class IndexResponse(BaseModel):
    file_name: str
    status: str
    chunks: int
@app.get("/")
def read_root():
    return {"message": "Welcome to my RAG API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/index", response_model=list[IndexResponse])
def index_endpoint():
    res = index_data()
    return res

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question
    top_k = request.top_k or settings.top_k

    if not question:
        raise HTTPException(status_code=400, detail="Where is your question?")

    qv = embed_query(question)
    context_chunks = query(qv, top_k=top_k)
    answer = generate_answer(question, context_chunks)

    sources = sorted({c["source"] for c in context_chunks if c.get("source")})
    context_response = [
        SourceChunk(
            text=c["text"],
            source=c.get("source", "unknown"),
            distance=c.get("distance", 0.0),
            similarity=c.get("similarity", 0.0),
            uuid=c.get("uuid", "")
        )
        for c in context_chunks
    ]

    return AskResponse(answer=answer, sources=sources, context=context_response)

@app.get('/sources', response_model=list[str])
def get_sources():
    return list_sources()

@app.delete('/sources/{source_name}', response_model=dict)
def delete_source_endpoint(source_name: str):
    deleted_count, deleted_ids = delete_source(source_name)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source_name}'")
    return {
             "deleted_chunks": deleted_count,
             "ids": deleted_ids
            }

@app.on_event("startup")
def startup_event():
    print("Setting the server up...")

@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    return HTTPException(status_code=999, detail=f"An unexpected error occurred: {str(exc)}")