import io
from datetime import datetime, timedelta, timezone

import pytest

from memoryforge.config import Settings
from memoryforge.schemas import ConversationEvent, Identifiers
from memoryforge.services import build_services
from memoryforge.telemetry import configure_logging

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def quiet_logs():
    configure_logging("WARNING", "json", stream=io.StringIO())


@pytest.fixture
def make_services():
    created = []

    def _make(llm=None, clock=None, queue=None, **overrides):
        settings = Settings(_env_file=None, **{"db_path": ":memory:", "run_worker": False, **overrides})
        kwargs = {"llm": llm, "queue": queue}
        if clock:
            kwargs["clock"] = clock
        svc = build_services(settings, **kwargs)
        created.append(svc)
        return svc

    yield _make
    for s in created:
        s.close()


@pytest.fixture
def svc(make_services):
    return make_services()


def ev(event_id, text, *, email=None, phone=None, user_id=None, minutes=0, role="customer", conv="c1", **kw):
    if not (email or phone or user_id):
        email = "default@example.com"
    ids = Identifiers(email=email, phone=phone, user_id=user_id)
    return ConversationEvent(event_id=event_id, conversation_id=conv, timestamp=T0 + timedelta(minutes=minutes),
                             role=role, text=text, identifiers=ids, **kw)


def ingest(svc, *events):
    for e in events:
        svc.queue.publish(e)
    return svc.worker.drain()
