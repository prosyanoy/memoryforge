import json
from datetime import timedelta

from conftest import T0, ev, ingest


def active(svc, ref):
    cid = svc.identity.lookup(ref)
    return {(m.kind.value, m.key): m for m in svc.store.list_memories(cid)}


def test_happy_path_stores_facts_and_marks_agent_skipped(svc):
    out = ingest(svc,
                 ev("e1", "I prefer email. I can't log in.", email="a@x.com"),
                 ev("e2", "Sorry to hear that, I prefer tea", email="a@x.com", role="agent", minutes=1))
    assert dict(out) == {"processed": 1, "skipped": 1}
    mem = active(svc, "a@x.com")
    assert mem[("preference", "contact_channel")].value == "email"
    assert mem[("issue", "login")].status.value == "open"
    assert ("preference", "likes_tea") not in mem  # agent text is never extracted


def test_duplicate_events_are_idempotent(svc):
    e = ev("e1", "I prefer email.", email="a@x.com")
    ingest(svc, e)
    before = svc.store.stats()
    out = ingest(svc, e, e)
    assert dict(out) == {"duplicate": 2}
    assert svc.store.stats() == {**before}


def test_newer_fact_supersedes_and_history_is_kept(svc):
    ingest(svc, ev("e1", "I prefer email.", email="a@x.com"),
           ev("e2", "Actually I prefer phone calls.", email="a@x.com", minutes=5))
    assert active(svc, "a@x.com")[("preference", "contact_channel")].value == "phone"
    cid = svc.identity.lookup("a@x.com")
    hist = svc.store.list_memories(cid, include_history=True)
    assert [m.active for m in hist if m.key == "contact_channel"] == [True, False]
    assert svc.index.size(cid) == len(svc.store.list_memories(cid))  # index mirrors active set only


def test_out_of_order_event_cannot_overwrite_newer_fact(svc):
    ingest(svc, ev("late", "I prefer phone calls.", email="a@x.com", minutes=10),
           ev("early", "I prefer email.", email="a@x.com", minutes=1))
    assert active(svc, "a@x.com")[("preference", "contact_channel")].value == "phone"
    assert 'forgotten:superseded' in svc.telemetry.metrics.render().decode()


def test_resolved_issue_replaces_open_issue(svc):
    ingest(svc, ev("e1", "My package never arrived.", email="a@x.com"),
           ev("e2", "The package finally arrived, all good now.", email="a@x.com", minutes=60))
    assert active(svc, "a@x.com")[("issue", "shipping")].status.value == "resolved"


def test_pii_and_low_confidence_facts_are_forgotten(svc):
    ingest(svc, ev("e1", "My card is 4242 4242 4242 4242. My SSN is 123-45-6789. My password is hunter2. "
                         "Maybe I can't log in, not sure.", email="a@x.com"))
    assert active(svc, "a@x.com") == {}
    dump = json.dumps([m.model_dump(mode="json") for m in svc.store.list_memories(svc.identity.lookup("a@x.com"), True)])
    assert "4242" not in dump and "hunter2" not in dump
    metrics = svc.telemetry.metrics.render().decode()
    assert 'forgotten:low_confidence' in metrics


def test_identity_variants_accumulate_into_one_customer(svc):
    ingest(svc, ev("e1", "My name is Ann.", email="a@x.com", phone="415-555-0100", minutes=0),
           ev("e2", "I love dark mode.", phone="+1 (415) 555 0100", minutes=1),
           ev("e3", "I prefer email.", email="A@X.COM", minutes=2))
    assert svc.store.stats()["customers"] == 1
    assert {"preferred_name", "likes_dark_mode", "contact_channel"} <= {k for _, k in active(svc, "a@x.com")}


def test_identity_conflict_is_quarantined_not_merged(svc):
    ingest(svc, ev("e1", "I love tea.", email="a@x.com"), ev("e2", "I love coffee.", email="b@x.com"))
    out = ingest(svc, ev("e3", "I love juice.", email="a@x.com", phone="415-555-0100", user_id="u1", minutes=1),
                 ev("e4", "I love soda.", email="b@x.com", phone="415-555-0100", minutes=2))
    assert dict(out) == {"processed": 1, "failed": 1}  # e3 links phone to a; e4 now conflicts
    dl = svc.store.get_dead_letter("e4")
    assert dl["stage"] == "identity" and dl["error_type"] == "IdentityConflict"
    assert svc.store.stats()["customers"] == 2
    assert ("preference", "likes_soda") not in active(svc, "b@x.com")


def test_replay_is_idempotent_and_rebuild_reproduces_state(svc):
    from memoryforge.synthetic import generate_events
    events, _ = generate_events(6, 2, seed=3)
    ingest(svc, *events)
    snapshot = lambda: sorted((m.customer_id, m.kind.value, m.key, m.value, m.status and m.status.value)
                              for c in svc.store._q("SELECT DISTINCT customer_id FROM memories")
                              for m in svc.store.list_memories(c[0]))
    base, stats = snapshot(), svc.store.stats()
    out = svc.worker.replay(0)
    assert set(out) == {"duplicate"} and snapshot() == base and svc.store.stats() == stats
    out = svc.worker.replay(0, rebuild=True)
    assert "duplicate" not in out and snapshot() == base
    assert svc.index.size() == stats["active_memories"]


def test_index_rebuild_from_store_matches_live_index(svc):
    ingest(svc, ev("e1", "I prefer email. I love dark mode.", email="a@x.com"))
    cid = svc.identity.lookup("a@x.com")
    q = svc.embedder.embed(["email"])[0]
    live = svc.index.search(cid, q, 5)
    svc.memory.rebuild_index()
    assert svc.index.search(cid, q, 5) == live


def test_failed_extraction_is_diagnosable_end_to_end(make_services):
    class Bad:
        name = "bad-llm"
        def generate(self, **kw):
            return "I'm sorry, I cannot help with that."

    svc = make_services(llm=Bad())
    out = ingest(svc, ev("boom", "I prefer email.", email="a@x.com", correlation_id="req-42"))
    assert dict(out) == {"failed": 1}
    dl = svc.store.get_dead_letter("boom")
    assert dl["stage"] == "extract" and dl["correlation_id"] == "req-42" and dl["attempts"] == 1
    assert "cannot help" in dl["error"]  # raw model output kept for diagnosis
    names = {s["name"]: s for s in dl["spans"]}
    assert {"pipeline.process", "stage.dedupe", "stage.identity", "stage.extract"} <= set(names)
    assert names["stage.extract"]["status"] == "ERROR" and names["pipeline.process"]["status"] == "ERROR"
    assert all(s["trace_id"] == dl["trace_id"] for s in dl["spans"])
    assert names["stage.extract"]["attributes"]["correlation_id"] == "req-42"
    assert not svc.store.is_processed("boom")  # retried on replay
    # replay retries and bumps attempts instead of duplicating the record
    svc.worker.replay(0)
    assert svc.store.get_dead_letter("boom")["attempts"] == 2 and len(svc.store.list_dead_letters()) == 1
    m = svc.telemetry.metrics.render().decode()
    assert 'memoryforge_stage_total{stage="extract",status="error"}' in m
    assert 'memoryforge_events_total{outcome="failed"}' in m


def test_dead_letter_clears_once_event_succeeds(make_services):
    state = {"fail": True}

    class Sometimes:
        name = "s"
        def generate(self, **kw):
            return "garbage" if state["fail"] else '{"facts": []}'

    svc = make_services(llm=Sometimes(), extraction_attempts=1)
    ingest(svc, ev("e1", "hello", email="a@x.com"))
    assert svc.store.stats()["dead_letters"] == 1
    state["fail"] = False
    svc.worker.replay(0)
    assert svc.store.stats()["dead_letters"] == 0 and svc.store.is_processed("e1")


def test_poison_message_is_dead_lettered_and_does_not_block(svc):
    svc.queue.publish_raw(b"{not valid json")
    svc.queue.publish(ev("e1", "I prefer email.", email="a@x.com"))
    out = svc.worker.drain()
    assert dict(out) == {"failed": 1, "processed": 1}
    assert svc.store.list_dead_letters()[0]["stage"] == "decode"
    assert svc.queue.committed == 2


def test_structured_logs_carry_correlation_id_across_stages(make_services):
    import io
    from memoryforge.telemetry import configure_logging
    buf = io.StringIO()
    configure_logging("INFO", "json", stream=buf)
    svc = make_services()
    ingest(svc, ev("e1", "My card is 4242 4242 4242 4242. I prefer email.", email="a@x.com", correlation_id="corr-1"))
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert lines and all(l["correlation_id"] == "corr-1" for l in lines)
    assert {"pii_redacted", "facts_stored", "event_done"} <= {l["event"] for l in lines}
    assert all("trace_id" in l for l in lines if l["event"] == "facts_stored")


def test_stage_metrics_cover_every_stage(svc):
    ingest(svc, ev("e1", "I prefer email.", email="a@x.com"))
    m = svc.telemetry.metrics.render().decode()
    for stage in ("dedupe", "identity", "extract", "forget", "embed", "store"):
        assert f'memoryforge_stage_total{{stage="{stage}",status="ok"}} 1.0' in m
        assert f'memoryforge_stage_duration_seconds_count{{stage="{stage}"}} 1.0' in m


def test_memory_timestamp_is_the_event_time_not_ingest_time(svc):
    ingest(svc, ev("e1", "I prefer email.", email="a@x.com"))
    m = active(svc, "a@x.com")[("preference", "contact_channel")]
    assert m.observed_at == T0
