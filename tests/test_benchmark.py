from scripts import benchmark


def test_there_are_200_distinct_queries():
    queries = benchmark.load_queries()

    assert len(queries) == 200
    assert len(set(queries)) == 200


def test_timed_query_returns_what_the_app_serves(engine):
    """The benchmark must time the same pipeline as search_by_description."""
    results, seconds = benchmark.timed_query(engine, "a heist movie", top_k=5)

    assert results == engine.search_by_description("a heist movie", top_k=5)
    assert set(seconds) == set(benchmark.STEPS)
    assert all(value >= 0 for value in seconds.values())
    parts = seconds["embedding"] + seconds["search"] + seconds["rerank"] + seconds["validation"]
    assert abs(parts - seconds["total"]) < 1e-6


def test_run_queries_reports_every_step(engine):
    latency = benchmark.run_queries(engine, ["one", "two", "three"], warmup=1)

    assert set(latency) == set(benchmark.STEPS)
    for stats in latency.values():
        assert stats["p50_ms"] <= stats["p95_ms"]


def test_machine_info_has_the_fields_the_report_needs():
    info = benchmark.machine_info()

    assert info["cpu_count"] >= 1
    assert info["device"] == "cpu"
    assert "lancedb" in info["versions"]
