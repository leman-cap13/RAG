import asyncio
import json
import logging
import time
from functools import partial

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage

from config import settings
from logging_config import request_id_var, setup_logging
from rag.cache import set_cached_answer
from rag.embedder import embed_query
from rag.generator import generate_answer
from rag.vector_store import list_sources, query

setup_logging(settings.log_level, settings.log_format)
logger = logging.getLogger(__name__)


async def handle_message(channel: AbstractChannel, message: AbstractIncomingMessage):
    async with message.process():
        payload = json.loads(message.body)
        question = payload["question"]
        top_k = payload["top_k"]
        history = payload.get("history") or []

        if not message.reply_to:
            logger.warning("job_skipped_no_reply_to", extra={"top_k": top_k})
            return

        token = request_id_var.set(message.correlation_id or "-")
        start = time.perf_counter()
        try:
            logger.info("job_started", extra={"top_k": top_k, "question": question})

            qv = embed_query(question)
            context = query(qv, top_k=top_k)
            total_sources = len(list_sources())
            answer = generate_answer(question, context, history=history, total_sources=total_sources)
            sources = sorted({c["source"] for c in context if c.get("source")})

            result = {"answer": answer, "sources": sources, "context": context}
            if not history:
                # Cached answers aren't scoped to a conversation, so only cache
                # context-free (first-turn) answers to avoid leaking one chat's
                # context into an unrelated session that asks the same question.
                set_cached_answer(question, top_k, result)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "job_completed",
                extra={"status": "success", "top_k": top_k, "duration_ms": duration_ms, "answer": answer},
            )
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "job_failed", extra={"status": "failed", "top_k": top_k, "duration_ms": duration_ms}
            )
            result = {"error": "internal_error"}
        finally:
            request_id_var.reset(token)

        await channel.default_exchange.publish(
            aio_pika.Message(json.dumps(result).encode(), correlation_id=message.correlation_id),
            routing_key=message.reply_to,
        )


async def main():
    logger.info("consumer_starting", extra={"queue": settings.rabbitmq_ask_queue})

    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1) # prefetch_count=1 means that the consumer will only receive one message at a time, and it will not receive another message until it has acknowledged the previous one. This is useful for ensuring that the consumer does not get overwhelmed with too many messages at once, and it allows for better control over the processing of messages.
    queue = await channel.declare_queue(settings.rabbitmq_ask_queue, durable=True)
    await queue.consume(partial(handle_message, channel))

    logger.info("consumer_waiting_for_messages", extra={"queue": settings.rabbitmq_ask_queue})

    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
