from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from rag.cache import clear_cache, get_cached_answer, set_cached_answer
from rag.cache import get_cached_answer
from rag.embedder import embed_query
from rag.generator import generate_answer
from rag.ingest import index_all
from rag.vector_store import delete_source, list_sources, query

app = FastAPI(title='RAG API')


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


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/index', response_model=list[IndexResult])
def index():
    results = index_all()
    if any (r['status'] == 'indexed' for r in results):
        clear_cahce()
    return results

@app.post('/ask', response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question boş ola bilməz.")
    
    cached = get_cached_answer(req.question, req.top_k)
    if cached:
        return AskResponse(**cached)

    qv = embed_query(req.question)
    context = query(qv, top_k=req.top_k)
    answer = generate_answer(req.question, context)
    sources = sorted({c["source"] for c in context if c.get("source")})

    response = AskResponse(answer=answer, sources=sources, context=context)
    set_cached_answer(req.question, req.top_k, response.model_dump())
    return response


@app.get('/sources', response_model=list[str])
def sources():
    return list_sources()

@app.delete('/sources/{filename}')
def remove_source(filename: str):
    deleted = delete_source(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"'{filename}' indekslənməmişdir")
    clear_cache()
    return {'file': filename, 'deleted_chunks': deleted}

