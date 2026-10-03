"""The helper scripts: the synthetic model used to build the image in CI, and the latency summary."""
import sys

import pytest
from fastapi.testclient import TestClient

from scripts import benchmark_api, make_dummy_model
from src import api


def test_dummy_model_can_be_served(tmp_path, monkeypatch):
    out = tmp_path / "serving_model"
    monkeypatch.setattr(sys, "argv", ["make_dummy_model", "--out", str(out)])

    make_dummy_model.main()

    monkeypatch.setenv("MODEL_DIR", str(out))
    with TestClient(api.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["model_version"] == "dummy"
        response = client.get("/recommend/1", params={"n": 5})        # the request the CI smoke test makes
        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "personalized" and len(body["items"]) == 5
        assert body["items"][0]["title"].startswith("Dummy Movie")
        assert client.post("/recommend", json={"liked_movie_ids": [1, 2, 3]}).status_code == 200


def test_dummy_model_is_deterministic():
    first, second = make_dummy_model.synthetic_ratings(), make_dummy_model.synthetic_ratings()

    assert first.equals(second)
    assert (first["timestamp"] < make_dummy_model.CUTOFF_TIMESTAMP).all()


def test_latency_summary():
    stats = benchmark_api.summarise([0.010] * 98 + [0.050, 0.100])

    assert stats["requests"] == 100
    assert stats["p50_ms"] == pytest.approx(10.0)
    assert stats["p95_ms"] == pytest.approx(10.0)
    assert stats["p99_ms"] > 50.0
    assert stats["mean_ms"] == pytest.approx(11.3)


def test_benchmark_measures_the_requested_number_of_calls(tmp_path, monkeypatch):
    """Against the in-process app: 200 GET and 50 POST are measured, 20 warm-up requests are not."""
    out = tmp_path / "serving_model"
    monkeypatch.setattr(sys, "argv", ["make_dummy_model", "--out", str(out)])
    make_dummy_model.main()
    monkeypatch.setenv("MODEL_DIR", str(out))
    monkeypatch.setattr(benchmark_api, "WARMUP_REQUESTS", 5)
    monkeypatch.setattr(benchmark_api, "GET_REQUESTS", 30)
    monkeypatch.setattr(benchmark_api, "POST_REQUESTS", 10)

    with TestClient(api.app) as client:
        monkeypatch.setattr(benchmark_api.httpx, "Client", lambda **kwargs: _Passthrough(client))
        report = benchmark_api.run("http://testserver", str(out), "in-process test")

    assert report["get_recommend_user"]["requests"] == 30
    assert report["post_recommend"]["requests"] == 10
    assert report["protocol"]["warmup_requests_not_counted"] == 5
    assert report["model"]["version"] == "dummy" and report["machine"]["logical_cpus"] >= 1


class _Passthrough:
    """Lets benchmark_api use the TestClient through its `with httpx.Client(...)` block."""

    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *exc):
        return False
