import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from rag.chunker import chunk_text
from rag.embedder import embed_query
from rag.cache import get_cached_answer, set_cached_answer, clear_cache
from rag.generator import generate_answer
from rag.vector_store import  query, list_sources, delete_source
from config import settings
from rag.ingest import index_data
from logging_config import request_id_var, setup_logging
from rag.vector_store import delete_source, list_sources
from rabbit import rabbit_client



setup_logging(settings.log_level, settings.log_format)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rabbit_client.connect()
    yield
    await rabbit_client.close()


app = FastAPI(title="RAG", version="9.32.6 debug", description = "salam", lifespan=lifespan)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    logger.debug("request_started", extra={"method": request.method, "path": request.url.path})
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request_failed",
            extra={"method": request.method, "path": request.url.path, "duration_ms": duration_ms},
        )
        request_id_var.reset(token)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_finished",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    request_id_var.reset(token)
    return response

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
def index(reindex: bool) -> list[IndexResponse]:
    results = index_data(reindex=reindex)
    indexed = [r["file"] for r in results if r["status"] == "indexed"]
    logger.info("index_completed", extra={"files_indexed": len(indexed), "files_total": len(results)})
    if any(r["status"] == "indexed" for r in results):
        clear_cache()
    return results

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:

    if not request.question:
        raise HTTPException(status_code=400, detail="Where is your question?")
    logger.debug("ask_received", extra={"question": request.question, "top_k": request.top_k})
    cached = get_cached_answer(request.question, top_k=request.top_k)

    if cached:
        return AskResponse(
            **cached
        )
    
    depth = await rabbit_client.queue_depth()
    if depth >= settings.rabbitmq_max_queue_depth:
        logger.warning("ask_rejected_queue_full", extra={"queue_depth": depth})
        raise HTTPException(status_code=503, detail="Server is busy, bir try again later")
    
    try:
        result = await rabbit_client.call(
            {
                "question": request.question,
                "top_k": request.top_k,
            },
            correlation_id=request_id_var.get(),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out waiting for worker.")

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    set_cached_answer(request.question, request.top_k, result)

    return AskResponse(**result)

@app.get('/sources', response_model=list[str])
def get_sources():
    return list_sources()

@app.delete('/sources/{source_name}', response_model=dict)
def delete_source_endpoint(source_name: str):
    deleted_count, deleted_ids = delete_source(source_name)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source_name}'")
    logger.info("source_deleted", extra={"file": source_name, "deleted_chunks": deleted_count})
    return {
             "deleted_chunks": deleted_count,
             "ids": deleted_ids
            }


# @app.on_event("startup")
# def startup_event():
#     print("Setting the server up...")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")

    return JSONResponse(
        status_code=500,
        content={
            "detail": f"An unexpected error occurred: {exc}"
        },
    )