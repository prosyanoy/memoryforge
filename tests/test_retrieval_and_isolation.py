import math
import random
from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import T0, ev, ingest
from memoryforge.api import create_app
from memoryforge.config import Settings
from memoryforge.synthetic import generate_events
from memoryforge.vector import VectorIndex, faiss


# ---- vector index --------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy"] + (["faiss"] if faiss else []))
def test_vector_index_backends(backend):
    idx = VectorIndex(4, backend)
    eye = np.eye(4, dtype=np.float32)
    idx.add_many("a", [10, 11, 12], eye[:3])
    idx.add("b", 99, eye[0])
    assert [i for i, _ in idx.search("a", eye[1], 2)][0] == 11
    assert idx.search("a", eye[0], 5)[0][0] == 10 and len(idx.search("a", eye[0], 5)) == 3
    idx.remove("a", [10])
    assert 10 not in [i for i, _ in idx.search("a", eye[0], 5)]
    assert idx.search("nobody", eye[0], 5) == [] and idx.size() == 3 and idx.size("b") == 1
    idx.drop_customer("a")
    assert idx.size() == 1
    idx.clear()
    assert idx.size() == 0


def test_backends_agree():
    if faiss is None:
        pytest.skip("faiss missing")
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    a, b = VectorIndex(16, "faiss"), VectorIndex(16, "numpy")
    for idx in (a, b):
        idx.add_many("c", range(50), vecs)
    q = vecs[7]
    ra, rb = a.search("c", q, 5), b.search("c", q, 5)
    assert [i for i, _ in ra] == [i for i, _ in rb]
    assert np.allclose([s for _, s in ra], [s for _, s in rb], atol=1e-5)


# ---- ranking -------------------------------------------------------------------------

def test_similarity_drives_ranking(svc):
    ingest(svc, ev("e1", "My email is ann@example.com. I can't log in. I love dark mode.", email="a@x.com"))
    top = svc.memory.retrieve(svc.identity.lookup("a@x.com"), "login problem", k=3).results
    assert (top[0].memory.kind.value, top[0].memory.key) == ("issue", "login")
    top = svc.memory.retrieve(svc.identity.lookup("a@x.com"), "email address", k=3).results
    assert top[0].memory.key == "email"


def test_recency_and_confidence_break_ties(make_services):
    now = T0 + timedelta(days=30)
    svc = make_services(clock=lambda: now)
    ingest(svc, ev("o2", "I love mugs.", email="a@x.com", minutes=0), ev("n2", "I love cups.", email="a@x.com", minutes=60 * 24 * 29))
    res = svc.memory.retrieve(svc.identity.lookup("a@x.com"), "", k=10).results  # no query: recency + confidence only
    keys = [r.memory.key for r in res]
    assert keys.index("likes_cups") < keys.index("likes_mugs")
    r = {x.memory.key: x.breakdown for x in res}
    assert math.isclose(r["likes_mugs"].recency, 0.5, rel_tol=0.05)  # 30 days old = one half-life
    assert r["likes_cups"].recency > 0.95


def test_include_history_and_k_limit(svc):
    ingest(svc, ev("e1", "I prefer email.", email="a@x.com"), ev("e2", "I prefer phone calls.", email="a@x.com", minutes=5))
    cid = svc.identity.lookup("a@x.com")
    assert len(svc.memory.retrieve(cid, "contact channel").results) == 1
    hist = svc.memory.retrieve(cid, "contact channel", include_history=True).results
    assert len(hist) == 2 and not all(r.memory.active for r in hist)
    assert len(svc.memory.retrieve(cid, "", k=1).results) == 1


# ---- isolation guarantee ---------------------------------------------------------------

def test_isolation_property_over_synthetic_population(svc):
    """Every retrieved memory must originate from an event that belongs to the queried customer,
    for every customer and many queries, including queries copied from other customers' facts."""
    events, owner = generate_events(25, 3, seed=11)
    ingest(svc, *events)
    assert svc.store.stats()["customers"] == 25 and svc.store.stats()["dead_letters"] == 0
    cust_of_user = {}
    for e in events:
        if e.identifiers.user_id:
            cust_of_user[owner[e.event_id]] = svc.identity.lookup(e.identifiers.user_id)
    assert len(set(cust_of_user.values())) == 25  # every persona resolved to its own single customer
    event_owner_cid = {e.event_id: cust_of_user[owner[e.event_id]] for e in events}
    all_facts = [m for c in cust_of_user.values() for m in svc.store.list_memories(c)]
    rng = random.Random(1)
    queries = ["", "contact", "login problem", "email", "phone number"] + [m.value for m in rng.sample(all_facts, 30)]
    for user, cid in cust_of_user.items():
        for q in queries:
            for include_history in (False, True):
                res = svc.memory.retrieve(cid, q, k=100, include_history=include_history).results
                assert res, (user, q)
                for r in res:
                    assert r.memory.customer_id == cid
                    assert event_owner_cid[r.memory.source_event_id] == cid


def test_identical_facts_for_two_customers_stay_separate(svc):
    ingest(svc, ev("a1", "My email is shared@example.com. I love dark mode.", email="a@x.com"),
           ev("b1", "My email is shared@example.com. I love dark mode.", email="b@x.com"))
    a, b = svc.identity.lookup("a@x.com"), svc.identity.lookup("b@x.com")
    assert a != b
    for cid, ev_id in ((a, "a1"), (b, "b1")):
        res = svc.memory.retrieve(cid, "dark mode").results
        assert res and {r.memory.source_event_id for r in res} == {ev_id}


def test_store_and_index_layers_each_enforce_isolation(svc):
    ingest(svc, ev("a1", "I love tea.", email="a@x.com"), ev("b1", "I love coffee.", email="b@x.com"))
    a, b = svc.identity.lookup("a@x.com"), svc.identity.lookup("b@x.com")
    b_ids = [m.id for m in svc.store.list_memories(b)]
    assert svc.store.get_memories(a, b_ids) == {}                      # SQL layer refuses cross-customer ids
    q = svc.embedder.embed(["coffee"])[0]
    assert {i for i, _ in svc.index.search(a, q, 50)}.isdisjoint(b_ids)  # separate index partitions


def test_delete_customer_removes_only_that_customer(svc):
    ingest(svc, ev("a1", "I love tea.", email="a@x.com"), ev("b1", "I love coffee.", email="b@x.com"))
    a, b = svc.identity.lookup("a@x.com"), svc.identity.lookup("b@x.com")
    assert svc.memory.forget_customer(a) == 1
    assert svc.identity.lookup("a@x.com") is None and svc.index.size(a) == 0
    assert len(svc.memory.retrieve(b, "coffee").results) == 1


def test_merge_moves_memories_and_old_id_redirects(svc):
    ingest(svc, ev("a1", "I love tea.", email="a@x.com"), ev("b1", "I love coffee.", phone="415-555-0100"))
    a, b = svc.identity.lookup("a@x.com"), svc.identity.lookup("4155550100")
    svc.store.merge_customers(b, a)
    svc.memory.rebuild_index()
    assert {m.key for m in svc.store.list_memories(a)} == {"likes_tea", "likes_coffee"}
    assert svc.identity.lookup(b) == a and svc.index.size(b) == 0


def test_retention_sweep(make_services):
    state = {"now": T0}
    svc = make_services(clock=lambda: state["now"], resolved_issue_ttl_days=10, retention_days=100)
    ingest(svc, ev("e1", "The login problem is fixed now.", email="a@x.com"), ev("e2", "I love tea.", email="a@x.com"))
    cid = svc.identity.lookup("a@x.com")
    state["now"] = T0 + timedelta(days=11)
    assert svc.memory.sweep_expired() == 1  # resolved issue expired, preference kept
    assert [m.key for m in svc.store.list_memories(cid)] == ["likes_tea"]
    state["now"] = T0 + timedelta(days=101)
    assert svc.memory.sweep_expired() == 1 and svc.index.size(cid) == 0


# ---- HTTP API -------------------------------------------------------------------------

@pytest.fixture
def client(make_services):
    svc = make_services()
    with TestClient(create_app(services=svc)) as c:
        yield c


def post_event(client, **kw):
    body = {"event_id": "e1", "conversation_id": "c1", "timestamp": "2026-03-01T12:00:00Z", "role": "customer",
            "text": "I prefer email. I can't log in.", "identifiers": {"email": "Ann@Example.com"}, **kw}
    return client.post("/events?wait=true", json=body)


def test_api_ingest_and_retrieve(client):
    r = post_event(client)
    assert r.status_code == 202 and r.json()["outcomes"] == {"processed": 1}
    cid = client.get("/customers/resolve", params={"email": "ann@example.com"}).json()["customer_id"]
    for ref in (cid, "ANN@example.com"):
        body = client.get(f"/memory/{ref}", params={"q": "login trouble"}).json()
        assert body["customer_id"] == cid and body["results"][0]["memory"]["key"] == "login"
        assert body["took_ms"] >= 0 and set(body["results"][0]["breakdown"]) == {"similarity", "recency", "confidence"}
    assert client.get("/memory/%2B14155550100").status_code == 404
    assert client.get("/memory/cus_nope").status_code == 404
    assert client.get(f"/memory/{cid}", params={"k": 0}).status_code == 422


def test_api_batch_ingest_and_validation(client):
    batch = [{"event_id": f"b{i}", "conversation_id": "c", "timestamp": "2026-03-01T12:00:00Z",
              "text": "I love dark mode.", "identifiers": {"user_id": "u1"}} for i in range(3)]
    r = client.post("/events?wait=true", json=batch)
    assert r.json()["accepted"] == 3 and r.json()["outcomes"] == {"processed": 3}
    assert client.post("/events", json={"event_id": "x"}).status_code == 422
    assert client.post("/events", json={**batch[0], "identifiers": {}}).status_code == 422


def test_api_correlation_id_flows_from_request_to_pipeline_failure(make_services):
    class Bad:
        name = "bad"
        def generate(self, **kw): return "nope"
    svc = make_services(llm=Bad())
    with TestClient(create_app(services=svc)) as c:
        r = c.post("/events?wait=true", headers={"X-Correlation-ID": "trace-me"}, json={
            "event_id": "bad1", "conversation_id": "c", "timestamp": "2026-03-01T12:00:00Z", "text": "hello",
            "identifiers": {"email": "a@x.com"}})
        assert r.headers["X-Correlation-ID"] == "trace-me" and r.json()["outcomes"] == {"failed": 1}
        listing = c.get("/failures").json()
        assert listing[0]["event_id"] == "bad1" and listing[0]["correlation_id"] == "trace-me"
        detail = c.get("/failures/bad1").json()
        assert detail["stage"] == "extract" and any(s["name"] == "stage.extract" for s in detail["spans"])
        assert c.get("/failures/unknown").status_code == 404


def test_api_metrics_health_forget_merge_replay(client):
    post_event(client)
    post_event(client, event_id="e2", text="I love tea.", identifiers={"phone": "415-555-0100"})
    metrics = client.get("/metrics").text
    assert 'memoryforge_stage_total{stage="extract",status="ok"} 2.0' in metrics
    assert "memoryforge_stage_duration_seconds_bucket" in metrics
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and health["customers"] == 2
    a = client.get("/customers/resolve", params={"email": "ann@example.com"}).json()["customer_id"]
    b = client.get("/customers/resolve", params={"phone": "4155550100"}).json()["customer_id"]
    assert client.get("/customers/resolve", params={"email": "ann@example.com", "phone": "4155550100"}).status_code == 409
    assert client.post("/admin/replay", json={"from_offset": 0}).json()["outcomes"] == {"duplicate": 2}
    assert client.post("/admin/merge", json={"source": b, "target": a}).json()["memories_moved"] == 1
    assert len(client.get(f"/memory/{a}", params={"q": "tea"}).json()["results"]) >= 1
    assert client.post("/admin/merge", json={"source": a, "target": a}).status_code == 400
    assert client.delete(f"/memory/{a}").json()["customer_id"] == a
    assert client.get(f"/memory/{a}").status_code == 404
    assert client.post("/admin/sweep").json() == {"expired_removed": 0}


def test_admin_token_protects_sensitive_routes(make_services):
    svc = make_services(admin_token="s3cret")
    with TestClient(create_app(services=svc)) as c:
        assert c.get("/failures").status_code == 401
        assert c.post("/admin/replay", json={}).status_code == 401
        assert c.delete("/memory/cus_x").status_code == 401
        assert c.get("/failures", headers={"X-Admin-Token": "s3cret"}).status_code == 200
        assert c.get("/healthz").status_code == 200


def test_hash_embedder_is_deterministic_and_normalised():
    from memoryforge.embeddings import HashingEmbedder
    e = HashingEmbedder(64)
    v = e.embed(["hello world", "hello world", "something else"])
    assert np.allclose(v[0], v[1]) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_wait_true_reports_outcome_even_when_background_worker_runs(make_services):
    svc = make_services(run_worker=True)
    with TestClient(create_app(services=svc)) as c:  # lifespan starts the consumer thread
        r = post_event(c)
        assert r.json()["outcomes"] == {"processed": 1}


def test_evaluation_harness_small_population():
    from memoryforge.evaluation import evaluate
    r = evaluate(customers=15, conversations=3, seed=2)
    assert r.leaked_results == 0 and r.results_checked > 0
    assert r.customers_resolved_uniquely == 15
    assert r.f1 >= 0.95
