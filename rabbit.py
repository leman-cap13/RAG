import asyncio
import json
import logging
import uuid

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from config import settings

logger = logging.getLogger(__name__)


class RabbitRPCClient:
    def __init__(self):
        self._connection = None
        self._channel = None
        self._callback_queue = None
        self._futures = {}

    async def connect(self):
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        self._callback_queue = await self._channel.declare_queue(exclusive=True)
        await self._callback_queue.consume(self.on_response, no_ack=True)
        await self._channel.declare_queue(settings.rabbitmq_queue, durable=True)

    async def close(self):
        if self._connection:
            await self._connection.close()

    async def queue_depth(self):
        result = await self._channel.declare_queue(settings.rabbitmq_queue, durable=True, passive=True)
        return result.declaration_result.message_count
    
    async def on_response(self, message: AbstractIncomingMessage):
        future = self._futures.pop(message.correlation_id, None)
        if future and not future.done():
            future.set_result(json.loads(message.body))

    async def call(self, payload, timeout=None, correlation_id=None):
        correlation_id = correlation_id or str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._futures[correlation_id] = future

        await self._channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(payload).encode(),
                content_type="application/json",
                correlation_id=correlation_id,
                reply_to=self._callback_queue.name,
            ),
            routing_key=settings.rabbitmq_ask_queue,
        )

        try:
            return await asyncio.wait_for(future, timeout=timeout or settings.rabbitmq_rpc_timeout)
        except asyncio.TimeoutError:
            self._futures.pop(correlation_id, None)
            raise



rabbit_client = RabbitRPCClient()