import json
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

_redis = redis.from_url(settings.redis_url, decode_responses=True)


def _key(session_id):
    return f"chat_session:{session_id}"


def get_history(session_id):
    raw = _redis.get(_key(session_id))
    if not raw:
        return []
    return json.loads(raw)


def append_turn(session_id, question, answer):
    history = get_history(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    history = history[-(settings.max_history_turns * 2):]
    _redis.set(_key(session_id), json.dumps(history), ex=settings.session_ttl)
