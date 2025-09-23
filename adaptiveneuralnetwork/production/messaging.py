"""
Message queue integration for Kafka and RabbitMQ.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import logging
from datetime import datetime

# Optional messaging dependencies
try:
    import aiokafka
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Dummy classes to avoid import errors
    class AIOKafkaProducer:
        pass
    class AIOKafkaConsumer:
        pass
    class KafkaError(Exception):
        pass

try:
    import aio_pika
    from aio_pika import connect_robust, Message, DeliveryMode
    from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    # Dummy classes to avoid import errors
    class Message:
        pass
    class DeliveryMode:
        PERSISTENT = 2


@dataclass
class MessagingConfig:
    """Configuration for message queue systems."""
    # Kafka config
    kafka_bootstrap_servers: Optional[List[str]] = None
    kafka_client_id: str = "adaptive_nn_client"
    kafka_group_id: str = "adaptive_nn_group"
    kafka_auto_offset_reset: str = "latest"
    kafka_enable_auto_commit: bool = True
    
    # RabbitMQ config
    rabbitmq_url: Optional[str] = None  # e.g., "amqp://user:pass@localhost/"
    rabbitmq_exchange: str = "adaptive_nn"
    rabbitmq_queue_prefix: str = "ann_queue"
    rabbitmq_routing_key_prefix: str = "ann"
    
    # General config
    message_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class MessageQueue:
    """Abstract base class for message queue implementations."""
    
    def __init__(self, config: MessagingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> None:
        """Connect to the message queue."""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Disconnect from the message queue."""
        raise NotImplementedError
    
    async def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Send a message to the specified topic."""
        raise NotImplementedError
    
    async def consume_messages(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Consume messages from the specified topic."""
        raise NotImplementedError


class KafkaProducer(MessageQueue):
    """Kafka message producer for async processing."""
    
    def __init__(self, config: MessagingConfig):
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka not available. Install with: pip install aiokafka")
        
        super().__init__(config)
        
        if not config.kafka_bootstrap_servers:
            raise ValueError("Kafka bootstrap servers not provided in config")
        
        self.producer = None
        self.consumer = None
        self.consumer_tasks = {}
    
    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                client_id=self.config.kafka_client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=int(self.config.retry_delay * 1000),
                retries=self.config.max_retries
            )
            
            await self.producer.start()
            self.logger.info("Kafka producer connected")
            
        except Exception as e:
            self.logger.error(f"Failed to connect Kafka producer: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        try:
            if self.producer:
                await self.producer.stop()
                self.producer = None
            
            if self.consumer:
                await self.consumer.stop()
                self.consumer = None
            
            # Cancel consumer tasks
            for task in self.consumer_tasks.values():
                task.cancel()
            self.consumer_tasks.clear()
            
            self.logger.info("Kafka connection closed")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Kafka: {e}")
    
    async def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Send a message to a Kafka topic."""
        if not self.producer:
            await self.connect()
        
        try:
            # Add metadata
            enriched_message = {
                **message,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "adaptive_neural_network"
            }
            
            await self.producer.send(topic, enriched_message, key=key)
            self.logger.debug(f"Message sent to topic {topic}")
            
        except KafkaError as e:
            self.logger.error(f"Failed to send message to {topic}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error sending message: {e}")
            raise
    
    async def consume_messages(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Consume messages from a Kafka topic."""
        if topic in self.consumer_tasks:
            self.logger.warning(f"Consumer for topic {topic} already running")
            return
        
        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                auto_offset_reset=self.config.kafka_auto_offset_reset,
                enable_auto_commit=self.config.kafka_enable_auto_commit,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                consumer_timeout_ms=self.config.message_timeout * 1000
            )
            
            await consumer.start()
            self.logger.info(f"Started consuming from topic: {topic}")
            
            # Create consumer task
            task = asyncio.create_task(self._consume_loop(consumer, handler, topic))
            self.consumer_tasks[topic] = task
            
        except Exception as e:
            self.logger.error(f"Failed to start consumer for {topic}: {e}")
            raise
    
    async def _consume_loop(self, consumer: AIOKafkaConsumer, handler: Callable, topic: str) -> None:
        """Internal consumer loop."""
        try:
            async for message in consumer:
                try:
                    # Process message
                    await self._handle_message(message.value, handler)
                    
                except Exception as e:
                    self.logger.error(f"Error handling message from {topic}: {e}")
                    
        except asyncio.CancelledError:
            self.logger.info(f"Consumer for {topic} cancelled")
        except Exception as e:
            self.logger.error(f"Consumer loop error for {topic}: {e}")
        finally:
            await consumer.stop()
    
    async def _handle_message(self, message_data: Dict[str, Any], handler: Callable) -> None:
        """Handle a single message."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message_data)
            else:
                handler(message_data)
        except Exception as e:
            self.logger.error(f"Message handler error: {e}")
            raise


class RabbitMQProducer(MessageQueue):
    """RabbitMQ message producer for async processing."""
    
    def __init__(self, config: MessagingConfig):
        if not RABBITMQ_AVAILABLE:
            raise ImportError("aio-pika not available. Install with: pip install aio-pika")
        
        super().__init__(config)
        
        if not config.rabbitmq_url:
            raise ValueError("RabbitMQ URL not provided in config")
        
        self.connection = None
        self.channel = None
        self.exchange = None
        self.consumer_tasks = {}
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            self.connection = await connect_robust(
                self.config.rabbitmq_url,
                timeout=self.config.message_timeout
            )
            
            self.channel = await self.connection.channel()
            
            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                self.config.rabbitmq_exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            self.logger.info("RabbitMQ connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            # Cancel consumer tasks
            for task in self.consumer_tasks.values():
                task.cancel()
            self.consumer_tasks.clear()
            
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            
            self.logger.info("RabbitMQ connection closed")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    async def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Send a message to a RabbitMQ exchange."""
        if not self.connection or self.connection.is_closed:
            await self.connect()
        
        try:
            # Add metadata
            enriched_message = {
                **message,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "adaptive_neural_network"
            }
            
            routing_key = f"{self.config.rabbitmq_routing_key_prefix}.{topic}"
            if key:
                routing_key += f".{key}"
            
            message_body = Message(
                json.dumps(enriched_message).encode('utf-8'),
                delivery_mode=DeliveryMode.PERSISTENT
            )
            
            await self.exchange.publish(message_body, routing_key=routing_key)
            self.logger.debug(f"Message sent to exchange with routing key: {routing_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise
    
    async def consume_messages(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Consume messages from a RabbitMQ queue."""
        if topic in self.consumer_tasks:
            self.logger.warning(f"Consumer for topic {topic} already running")
            return
        
        if not self.connection or self.connection.is_closed:
            await self.connect()
        
        try:
            # Declare queue
            queue_name = f"{self.config.rabbitmq_queue_prefix}_{topic}"
            queue = await self.channel.declare_queue(queue_name, durable=True)
            
            # Bind queue to exchange
            routing_key = f"{self.config.rabbitmq_routing_key_prefix}.{topic}"
            await queue.bind(self.exchange, routing_key)
            
            self.logger.info(f"Started consuming from queue: {queue_name}")
            
            # Create consumer task
            task = asyncio.create_task(self._consume_loop(queue, handler, topic))
            self.consumer_tasks[topic] = task
            
        except Exception as e:
            self.logger.error(f"Failed to start consumer for {topic}: {e}")
            raise
    
    async def _consume_loop(self, queue, handler: Callable, topic: str) -> None:
        """Internal consumer loop."""
        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    try:
                        # Parse message
                        message_data = json.loads(message.body.decode('utf-8'))
                        
                        # Handle message
                        await self._handle_message(message_data, handler)
                        
                        # Acknowledge message
                        await message.ack()
                        
                    except Exception as e:
                        self.logger.error(f"Error handling message from {topic}: {e}")
                        await message.nack()
                        
        except asyncio.CancelledError:
            self.logger.info(f"Consumer for {topic} cancelled")
        except Exception as e:
            self.logger.error(f"Consumer loop error for {topic}: {e}")
    
    async def _handle_message(self, message_data: Dict[str, Any], handler: Callable) -> None:
        """Handle a single message."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message_data)
            else:
                handler(message_data)
        except Exception as e:
            self.logger.error(f"Message handler error: {e}")
            raise


class HybridMessageQueue:
    """Hybrid message queue that can use both Kafka and RabbitMQ."""
    
    def __init__(self, config: MessagingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.kafka_producer = None
        self.rabbitmq_producer = None
        
        # Initialize Kafka if configured
        if config.kafka_bootstrap_servers:
            try:
                self.kafka_producer = KafkaProducer(config)
                self.logger.info("Kafka producer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Kafka: {e}")
        
        # Initialize RabbitMQ if configured
        if config.rabbitmq_url:
            try:
                self.rabbitmq_producer = RabbitMQProducer(config)
                self.logger.info("RabbitMQ producer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize RabbitMQ: {e}")
        
        if not self.kafka_producer and not self.rabbitmq_producer:
            raise ValueError("At least one message queue must be configured")
    
    async def connect(self) -> None:
        """Connect to all configured message queues."""
        if self.kafka_producer:
            await self.kafka_producer.connect()
        
        if self.rabbitmq_producer:
            await self.rabbitmq_producer.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from all message queues."""
        if self.kafka_producer:
            await self.kafka_producer.disconnect()
        
        if self.rabbitmq_producer:
            await self.rabbitmq_producer.disconnect()
    
    async def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None, 
                          prefer_kafka: bool = True) -> None:
        """Send message to preferred queue system."""
        if prefer_kafka and self.kafka_producer:
            try:
                await self.kafka_producer.send_message(topic, message, key)
                return
            except Exception as e:
                self.logger.error(f"Kafka send failed, trying RabbitMQ: {e}")
        
        if self.rabbitmq_producer:
            try:
                await self.rabbitmq_producer.send_message(topic, message, key)
                return
            except Exception as e:
                self.logger.error(f"RabbitMQ send failed: {e}")
                if not prefer_kafka and self.kafka_producer:
                    await self.kafka_producer.send_message(topic, message, key)
                else:
                    raise
        
        raise RuntimeError("No message queue available for sending")
    
    async def consume_messages(self, topic: str, handler: Callable[[Dict[str, Any]], None], 
                              prefer_kafka: bool = True) -> None:
        """Consume messages from preferred queue system."""
        if prefer_kafka and self.kafka_producer:
            try:
                await self.kafka_producer.consume_messages(topic, handler)
                return
            except Exception as e:
                self.logger.error(f"Kafka consume failed, trying RabbitMQ: {e}")
        
        if self.rabbitmq_producer:
            try:
                await self.rabbitmq_producer.consume_messages(topic, handler)
                return
            except Exception as e:
                self.logger.error(f"RabbitMQ consume failed: {e}")
                if not prefer_kafka and self.kafka_producer:
                    await self.kafka_producer.consume_messages(topic, handler)
                else:
                    raise
        
        raise RuntimeError("No message queue available for consuming")


# Message schemas for different use cases
class PredictionMessage:
    """Message schema for prediction requests."""
    
    @staticmethod
    def create(input_data: List[List[float]], model_name: str = "default", 
               user_id: Optional[str] = None, session_id: Optional[str] = None,
               options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "type": "prediction_request",
            "data": {
                "input_data": input_data,
                "model_name": model_name,
                "user_id": user_id,
                "session_id": session_id,
                "options": options or {}
            }
        }


class MetricsMessage:
    """Message schema for metrics data."""
    
    @staticmethod
    def create(metrics: Dict[str, float], model_name: str, 
               tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "type": "metrics_update",
            "data": {
                "metrics": metrics,
                "model_name": model_name,
                "tags": tags or {}
            }
        }


class ModelUpdateMessage:
    """Message schema for model updates."""
    
    @staticmethod
    def create(model_name: str, version: str, action: str,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "type": "model_update",
            "data": {
                "model_name": model_name,
                "version": version,
                "action": action,  # "load", "unload", "update"
                "metadata": metadata or {}
            }
        }