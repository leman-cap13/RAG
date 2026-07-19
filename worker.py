import asyncio
import json
import logging 
import time
from functools import partial

import aio_pika
from aio_pika.abc import AbstractChannel,AbstractIncomingMessage

from config import settings
from logging_config import request_id_var, setup_logging
from rag.cache import set_cached_answer
from rag.embedder import embed_query
from rag.generator import generate_answer
from rag.vector_store import query

setup_logging(settings.log_level,settings.log_format)
logger=logging.getLogger(__name__)

async def handle_message(channel:AbstractChannel,message:AbstractIncomingMessage):
    async with message.process():
        payload=json.loads(message.body)
        question=payload['question']
        top_k=payload['top_k']

        if not message.reply_to:
            logger.warning('job_skipped_no_reply_to',extra={'top_k':top_k})
            return
        
        token=request_id_var.set(message.correlation_id or '-')
        start=time.perf_counter()
        try:
            logger.info('job_started',extra={'top_k':top_k,'question':question})
            
            qv=embed_query(question)
            context=query(qv,top_k=top_k)
            answer=generate_answer(question,context)
            sources=sorted({c['source'] for c in context if c.get('source')})

            result={'answer':answer,'sources':sources,'context':context}
            set_cached_answer(question,top_k,result)
            duration_ms=round((time.perf_counter()-start)*1000,2)
            logger.info(
                'job_complated',extra={'status':'failed','top_k':top_k,'duration_ms':duration_ms}
            )
            result={'error':'internal_error'}
        finally:
            request_id_var.reset(token)

        await channel.default_exchange.publish(
            aio_pika.Message(json.dumps(result).encode(),correlation_id=message.correlation_id)
        )

async def main():
    logger.info('consumer_starting',extra={'queue':settings.rabbitmq_ask_queue})
    
    connection=await aio_pika.connect_robust(settings.rabbitmq_url)
    channel= await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue=await channel.declare_queue(settings.rabbitmq_ask_queue,durable=True)
    await queue.consume(partial(handle_message,channel()))

    logger.info('consumee_waiting_for_message',extra={'queue':settings.rabbit_ask_queue})

    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__=='__main__':
    asyncio.run(main())
