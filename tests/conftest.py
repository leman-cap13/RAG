import asyncio
import os
import tempfile
import threading
import time

# Must be set before `config`/`rag`/`api`/`worker` are imported anywhere,
# since Settings() and the module-level Gemini/Chroma clients are built at import time.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("CHROMA_PATH", tempfile.mkdtemp(prefix="rag-test-chroma-"))
os.environ.setdefault("DATA_DIR", "tests/fixtures/data")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")  # RediSearch indexes can only live in db 0
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
os.environ.setdefault("RABBITMQ_ASK_QUEUE", "test_ask_queue")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import google.genai as genai  # noqa: E402


def _deterministic_vector(text, dims=16):
    seed = sum(text.encode("utf-8"))
    return [((seed * (i + 1)) % 97) / 97 for i in range(dims)]


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeGenerateResponse:
    def __init__(self, text):
        self.text = text


FAKE_ANSWER = "Bu, test üçün simulyasiya edilmiş cavabdır. [1]"


class _FakeModels:
    def embed_content(self, model, contents, config):
        return _FakeEmbedResponse([_FakeEmbedding(_deterministic_vector(t)) for t in contents])

    def generate_content(self, model, contents, config):
        return _FakeGenerateResponse(FAKE_ANSWER)


class _FakeGenaiClient:
    def __init__(self, api_key=None):
        self.models = _FakeModels()


# Patched before any project module does `from google import genai; genai.Client(...)`.
genai.Client = _FakeGenaiClient

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import api as api_module  # noqa: E402
import worker as worker_module  # noqa: E402
from rag.cache import clear_cache  # noqa: E402
from rag.session import _redis as session_redis  # noqa: E402


def _run_worker_forever():
    asyncio.run(worker_module.main())


@pytest.fixture(scope="session", autouse=True)
def worker_process():
    thread = threading.Thread(target=_run_worker_forever, daemon=True)
    thread.start()
    time.sleep(1.5)  # give the consumer time to connect and start consuming
    yield


@pytest_asyncio.fixture
async def client():
    async with api_module.lifespan(api_module.app):
        transport = ASGITransport(app=api_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    yield
    clear_cache()
    for key in session_redis.scan_iter("chat_session:*"):
        session_redis.delete(key)
