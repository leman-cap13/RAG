from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from config import settings
from rag.embedder import embed_query
from rag.vector_store import query, list_sources, delete_source
from rag.ingest import index_all
from rag.generator import generate_answer

app = FastAPI(title="RAG API", version="1.0.0")

class AskRequest(BaseModel):
    question: str

    top_k: int = settings.top_k


class SourceChunk(BaseModel):
    text: str
    source: str | None
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    context: list[SourceChunk]


class IndexResult(BaseModel):
    file: str
    status: str
    chunks: int



@app.get("/health")

def health():
    return {"status": "ok"}



@app.post("/index", response_model=list[IndexResult])
def index():
    results = index_all()
    return results


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    qv = embed_query(request.question)
    context = query(qv, top_k=request.top_k)

    answer = generate_answer(request.question, context)

    sources = sorted({c["source"] for c in context if c.get("source")})

    response = AskResponse(
        answer=answer,
        sources=sources,
        context=[SourceChunk(**c) for c in context],
    )
    return response



@app.get("/sources", response_model=list[str])
def sources():
    return list_sources()


@app.delete("/sources/{filename}")

def delete(filename: str):
    deleted_count = delete_source(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"file": filename, "deleted_chunks": deleted_count}