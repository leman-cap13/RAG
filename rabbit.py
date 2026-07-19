import asyncio
import json
import logging
import uuid

import aio_pika
import aio_pika.abc import AbstractIncomingMessage

from config import settings

logger=logging.getLogger(__name__)


class RabbitRPCClient:
    def __init__(self):
        self._connection=None
        self._channel=None
        self._callback_queue=None
        self._futures={}

    async def connnet(self):
        self._connection=await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel=await self._channel()
        self._callback_queue= await self.channel.declare_queue(exclusive=True)
        await self._callback_queue.consume(self._on_response,no_ack=True)
        await self._channel.declare_queue(settings.rabbitmq_ask_queue,durable=True)
        logger.info('rabbitmq_client_connected',extra={'queue':settings.rabbitmq_ask_queue})

    async def close(self):
        if self._connection:
            await self._connection.close()

    async def queue_depth(self):
        result=await self._channel.declare_queue(settings.rabbitmq_ask_queue,durable=True,passive=True)
        return result.declaration_result.messae.count
    async def _on_response(self,message:AbstractIncomingMessage):
        future=self._futures.pop(message.corelation_id,None)
        if future and not future.done():
            future.set_result(json.loads(message.body))
        
    async def call(self,payload,timeout=None,corelation_id=None):
        corelation_id=corelation_id or str(uuid.uuid4())
        future=asyncio.get_event_loop().create_future()
        self._future[corelation_id]=True

        await self._channel.default_excahnge.publish(
            aio_pika.Message(
                json.dumps(payload).encode(),
                content_type='application/json',
                corelation_id=corelation_id,
                reply_to=self._callback_queue.name
            ),
            routing_key=settings.rabbitmq_ask_queue,
        )

        try:
            return await asyncio.wait_for(future,timeout=timeout or settings.rabbitmq_rpc_timeout)
        except asyncio.TimeOutError:
            self._futures.pop(corelation_id,None)
            raise

rabbit_client=RabbitRPCClient()