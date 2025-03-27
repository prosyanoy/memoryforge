"""Event queue abstraction: in-process log (default) or Kafka/Redpanda, with replay support."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .schemas import ConversationEvent


@dataclass(frozen=True)
class QueueMessage:
    offset: int
    value: bytes
    partition: int = 0


class EventQueue(Protocol):
    def publish(self, event: ConversationEvent) -> int: ...
    def publish_raw(self, value: bytes, key: str | None = None) -> int: ...
    def poll(self, max_messages: int = 100, timeout: float = 0.0) -> list[QueueMessage]: ...
    def commit(self, msg: QueueMessage) -> None: ...
    def seek_to(self, offset: int) -> None: ...
    def lag(self) -> int: ...
    def close(self) -> None: ...


class InMemoryQueue:
    """Append-only log + a consumer position and committed offset. The log is retained, so replay works."""

    def __init__(self) -> None:
        self._log: list[bytes] = []
        self._pos = 0
        self._committed = 0
        self._cv = threading.Condition()

    def publish(self, event: ConversationEvent) -> int:
        return self.publish_raw(event.model_dump_json().encode(), event.conversation_id)

    def publish_raw(self, value: bytes, key: str | None = None) -> int:
        with self._cv:
            self._log.append(value)
            self._cv.notify_all()
            return len(self._log) - 1

    def poll(self, max_messages: int = 100, timeout: float = 0.0) -> list[QueueMessage]:
        with self._cv:
            if self._pos >= len(self._log) and timeout > 0:
                self._cv.wait(timeout)
            end = min(len(self._log), self._pos + max_messages)
            msgs = [QueueMessage(i, self._log[i]) for i in range(self._pos, end)]
            self._pos = end
            return msgs

    def commit(self, msg: QueueMessage) -> None:
        with self._cv:
            self._committed = max(self._committed, msg.offset + 1)

    def seek_to(self, offset: int) -> None:
        with self._cv:
            self._pos = max(0, min(offset, len(self._log)))
            self._committed = min(self._committed, self._pos)

    def lag(self) -> int:
        with self._cv:
            return len(self._log) - self._pos

    @property
    def committed(self) -> int:
        return self._committed

    def close(self) -> None:
        pass


def _topic_partition(topic: str, partition: int, offset: int | None = None):
    """confluent-kafka's TopicPartition; a plain namespace when the package is absent (injected test doubles)."""
    try:
        from confluent_kafka import TopicPartition
    except ImportError:
        return SimpleNamespace(topic=topic, partition=partition, offset=offset)
    return TopicPartition(topic, partition) if offset is None else TopicPartition(topic, partition, offset)


class KafkaQueue:
    """Kafka / Redpanda queue (confluent-kafka). Offsets are committed only after processing."""

    def __init__(self, bootstrap: str, topic: str, group_id: str, *, producer=None, consumer=None):
        self.topic = topic
        if producer is None or consumer is None:
            try:
                from confluent_kafka import Consumer, Producer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the 'kafka' extra: pip install -e '.[kafka]'") from exc
            producer = producer or Producer({"bootstrap.servers": bootstrap})
            consumer = consumer or Consumer(
                {
                    "bootstrap.servers": bootstrap,
                    "group.id": group_id,
                    "enable.auto.commit": False,
                    "auto.offset.reset": "earliest",
                }
            )
        self._producer, self._consumer = producer, consumer
        self._consumer.subscribe([topic])

    def publish(self, event: ConversationEvent) -> int:
        return self.publish_raw(event.model_dump_json().encode(), event.conversation_id)

    def publish_raw(self, value: bytes, key: str | None = None) -> int:
        self._producer.produce(self.topic, value=value, key=key)
        self._producer.flush(10)
        return -1  # offset is assigned by the broker

    def poll(self, max_messages: int = 100, timeout: float = 0.0) -> list[QueueMessage]:
        msgs = self._consumer.consume(num_messages=max_messages, timeout=max(timeout, 0.05))
        out = []
        for m in msgs:
            if m.error():
                continue
            out.append(QueueMessage(m.offset(), m.value(), m.partition()))
        return out

    def commit(self, msg: QueueMessage) -> None:
        self._consumer.commit(offsets=[_topic_partition(self.topic, msg.partition, msg.offset + 1)], asynchronous=False)

    def seek_to(self, offset: int) -> None:
        """Replay: rewind every assigned partition to `offset` (0 = beginning)."""
        deadline = time.time() + 10
        while not self._consumer.assignment() and time.time() < deadline:
            self._consumer.poll(0.2)  # drive the group join; any message returned is re-read after the seek
        for tp in self._consumer.assignment():
            self._consumer.seek(_topic_partition(tp.topic, tp.partition, offset))

    def lag(self) -> int:
        total = 0
        for tp in self._consumer.assignment():
            _, high = self._consumer.get_watermark_offsets(tp, timeout=2)
            pos = self._consumer.position([_topic_partition(tp.topic, tp.partition)])[0].offset
            total += max(0, high - max(pos, 0))
        return total

    def close(self) -> None:
        self._consumer.close()


def build_queue(settings: Settings) -> EventQueue:
    if settings.queue_backend == "kafka":
        return KafkaQueue(settings.kafka_bootstrap, settings.kafka_topic, settings.kafka_group)
    return InMemoryQueue()
