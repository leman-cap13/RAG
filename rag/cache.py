import json

from pydantic import ConfigDict
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Num
from redisvl.utils.vectorize.base import BaseVectorizer

from config import settings
from rag.embedder import embed_semantic, embed_semantic_batch


class GeminiCacheVectorizer(BaseVectorizer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, model, **kwargs):
        super().__init__(model=model, dims=len(embed_semantic("dimension probe")), **kwargs)

    def _embed(self, content=None, text=None, **kwargs):
        return embed_semantic(content or text)

    def _embed_many(self, contents=None, texts=None, batch_size=10, **kwargs):
        return embed_semantic_batch(contents or texts or [])


cache = SemanticCache(
    name="rag_cache",
    vectorizer=GeminiCacheVectorizer(model=settings.embed_model),
    distance_threshold=settings.cache_similarity_threshold,
    ttl=settings.cache_ttl, # time-to-live expiration for cached entries in seconds
    redis_url=settings.redis_url,
    filterable_fields=[{"name": "top_k", "type": "numeric"}],
)


def get_cached_answer(question, top_k): #get is cached answer for a given question and top_k value
    hits = cache.check(prompt=question, filter_expression=Num("top_k") == top_k, num_results=1)
    return json.loads(hits[0]["response"]) if hits else None


def set_cached_answer(question, top_k, payload): # set is redis cache for a given question and top_k value
    cache.store(prompt=question, response=json.dumps(payload), filters={"top_k": top_k})


def clear_cache():
    cache.clear()
