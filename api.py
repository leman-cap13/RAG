import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from config import settings
from logging_config import request_id_var, setup_logging
from rag.cache import clear_cache, get_cached_answer
from rag.ingest import index_all
from rag.vector_store import delete_source, list_sources
from rabbit import rabbit_client

setup_logging(settings.log_level, settings.log_format)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rabbit_client.connect()
    yield
    await rabbit_client.close()


app = FastAPI(title="RAG API", lifespan=lifespan)


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
    indexed = [r["file"] for r in results if r["status"] == "indexed"]
    logger.info("index_completed", extra={"files_indexed": len(indexed), "files_total": len(results)})
    if indexed:
        clear_cache()
    return results


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question boş ola bilməz")

    logger.debug("ask_received", extra={"question": req.question, "top_k": req.top_k})

    cached = get_cached_answer(req.question, req.top_k)
    if cached:
        return AskResponse(**cached)

    depth = await rabbit_client.queue_depth()
    if depth >= settings.rabbitmq_max_queue_depth:
        logger.warning("ask_rejected_queue_full", extra={"queue_depth": depth})
        raise HTTPException(status_code=503, detail="Server hazırda çox məşğuldur, bir az sonra yenidən cəhd edin")

    try:
        result = await rabbit_client.call(
            {"question": req.question, "top_k": req.top_k},
            correlation_id=request_id_var.get(),
        )
    except asyncio.TimeoutError:
        logger.error("ask_timeout", extra={"top_k": req.top_k})
        raise HTTPException(status_code=504, detail="Cavab vaxtında alınmadı, bir az sonra yenidən cəhd edin")

    if "error" in result:
        logger.error("ask_worker_error", extra={"error": result["error"]})
        raise HTTPException(status_code=502, detail="Sual emalı zamanı xəta baş verdi")

    return AskResponse(**result)


@app.get("/sources", response_model=list[str])
def sources():
    return list_sources()


@app.delete("/sources/{filename}")
def remove_source(filename: str):
    deleted = delete_source(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"'{filename}' indekslənməmişdi")
    logger.info("source_deleted", extra={"file": filename, "deleted_chunks": deleted})
    clear_cache()
    return {"file": filename, "deleted_chunks": deleted}
