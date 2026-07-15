from google import genai
from google.genai import types

from config import settings

client = genai.Client(api_key=settings.gemini_api_key)


def _embed_batch(texts, task_type):
    embeddings = []
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i:i + settings.embed_batch_size]
        response = client.models.embed_content(
            model=settings.embed_model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.extend(e.values for e in response.embeddings)
    return embeddings


def embed_documents(texts):
    return _embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text):
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]


def embed_semantic(text):
    return _embed_batch([text], task_type="SEMANTIC_SIMILARITY")[0]


def embed_semantic_batch(texts):
    return _embed_batch(texts, task_type="SEMANTIC_SIMILARITY")
