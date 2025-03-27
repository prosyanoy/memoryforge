import threading
import time
from types import SimpleNamespace

from conftest import ev
from memoryforge.queue import InMemoryQueue, KafkaQueue, QueueMessage


def test_in_memory_queue_poll_commit_replay():
    q = InMemoryQueue()
    offs = [q.publish(ev(f"e{i}", "hi", email="a@x.com")) for i in range(5)]
    assert offs == [0, 1, 2, 3, 4] and q.lag() == 5
    first = q.poll(3)
    assert [m.offset for m in first] == [0, 1, 2] and q.lag() == 2
    q.commit(first[-1])
    assert q.committed == 3
    q.seek_to(1)
    assert [m.offset for m in q.poll(10)] == [1, 2, 3, 4] and q.committed == 1
    assert q.poll(10) == []


def test_in_memory_queue_blocking_poll_wakes_on_publish():
    q = InMemoryQueue()
    threading.Timer(0.05, lambda: q.publish(ev("e", "hi", email="a@x.com"))).start()
    t0 = time.time()
    assert len(q.poll(1, timeout=2)) == 1 and time.time() - t0 < 1.5


def test_background_worker_consumes(make_services):
    svc = make_services()
    svc.worker.start(poll_timeout=0.05)
    try:
        svc.queue.publish(ev("e1", "I prefer email.", email="a@x.com"))
        deadline = time.time() + 5
        while not svc.store.is_processed("e1") and time.time() < deadline:
            time.sleep(0.02)
        assert svc.store.is_processed("e1")
    finally:
        svc.worker.stop()


class FakeMsg:
    def __init__(self, offset, value):
        self._o, self._v = offset, value
    def error(self): return None
    def offset(self): return self._o
    def value(self): return self._v
    def partition(self): return 0


class FakeBroker:
    """Just enough of the confluent-kafka producer/consumer API to test KafkaQueue's logic."""
    def __init__(self):
        self.log, self.pos, self.committed, self.subscribed = [], 0, [], None
        self.flushed = 0
    # producer
    def produce(self, topic, value=None, key=None): self.log.append(value)
    def flush(self, timeout): self.flushed += 1
    # consumer
    def subscribe(self, topics): self.subscribed = topics
    def consume(self, num_messages, timeout):
        out = [FakeMsg(i, self.log[i]) for i in range(self.pos, min(len(self.log), self.pos + num_messages))]
        self.pos += len(out)
        return out
    def commit(self, offsets, asynchronous): self.committed.append(offsets[0].offset)
    def assignment(self): return [SimpleNamespace(topic="t", partition=0)]
    def seek(self, tp): self.pos = tp.offset
    def poll(self, t): return None
    def close(self): pass


def test_kafka_queue_logic_with_fake_broker(make_services):
    broker = FakeBroker()
    q = KafkaQueue("unused", "t", "g", producer=broker, consumer=broker)
    assert broker.subscribed == ["t"]
    q.publish(ev("e1", "I prefer email.", email="a@x.com"))
    q.publish_raw(b"garbage")
    msgs = q.poll(10)
    assert [m.offset for m in msgs] == [0, 1] and isinstance(msgs[0], QueueMessage)
    q.commit(msgs[1])
    assert broker.committed == [2]
    q.seek_to(0)
    assert len(q.poll(10)) == 2  # replay re-reads from the beginning


def test_pipeline_runs_over_kafka_queue_class(make_services):
    broker = FakeBroker()
    svc = make_services(queue=KafkaQueue("unused", "t", "g", producer=broker, consumer=broker))
    svc.queue.publish(ev("e1", "I prefer email.", email="a@x.com"))
    svc.queue.publish_raw(b"garbage")
    assert dict(svc.worker.drain()) == {"processed": 1, "failed": 1}
    assert broker.committed == [1, 2]
    assert dict(svc.worker.replay(0)) == {"duplicate": 1, "failed": 1}
