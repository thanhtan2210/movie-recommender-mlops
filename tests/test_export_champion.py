"""Exporting the champion from a local MLflow registry into a serving directory."""
import json
import os

import mlflow
import numpy as np
import pandas as pd
import pytest

from src import export_champion, tracking
from src.model import BlendRecommender
from tests.conftest import CUTOFF
from tests.test_recommender import ROWS, ratings_frame

USERS = [1, 2, 3, 4, 5, 6, 7, 999]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A working directory with a local MLflow store; setup() always points at it."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    calls = []

    def setup(remote=False, tracking_uri=None, _setup=tracking.setup):
        calls.append(remote)
        return _setup(tracking_uri=uri)

    monkeypatch.setattr(tracking, "setup", setup)
    setup()
    calls.clear()
    yield calls
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.fixture
def blend():
    return BlendRecommender.fit(ratings_frame(rows=ROWS), "2019-06-01", CUTOFF, 4.0, k=2, train_window="1y",
                                blend_weight=0.5)


def register_champion(blend, tmp_path, with_movies=True):
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"train": {"rows": 24173907}}), encoding="utf-8")
    logged = tracking.log_run(
        "production-champion-blend", blend,
        {"k": 2, "train_window": "1y", "blend_weight": 0.5, "random_state": 42, "cutoff": "2019-07-01",
         "train_sha256": "abc", "git_commit": "deadbeef"},
        {"window": "production"}, str(stats), result=None, register=True,
        version_tags={"cutoff": "2019-07-01", "trained_before": "2019-07-01"})
    mlflow.MlflowClient().log_metric(logged.run_id, "ndcg_at_10", 0.0831)
    tracking.set_champion(logged.version)
    if with_movies:
        directory = tmp_path / "data" / "processed" / "2019-07-01"
        directory.mkdir(parents=True)
        pd.DataFrame({"movieId": np.array([10, 20, 30, 40, 50], dtype="int32"),
                      "title": ["Ten (1990)", "Twenty (1991)", "Thirty (1992)", "Forty (1993)", "No Year"],
                      "genres": ["Drama"] * 5}).to_parquet(directory / "movies.parquet", index=False)
    return logged


def test_export_writes_a_self_contained_serving_directory(registry, blend, tmp_path):
    logged = register_champion(blend, tmp_path)

    meta = export_champion.export(serving_dir="serving_model")

    assert registry == [False]                                   # read from the local store, not DagsHub
    assert sorted(os.listdir("serving_model")) == ["meta.json", "model", "movies.json"]
    with open(os.path.join("serving_model", "meta.json"), encoding="utf-8") as f:
        assert json.load(f) == meta
    assert meta["model_version"] == logged.version and meta["model_type"] == "blend"
    assert meta["cutoff"] == "2019-07-01" and meta["trained_before"] == "2019-07-01"
    assert meta["trained_on_rows"] == 24173907
    assert meta["config"] == {"k": "2", "train_window": "1y", "blend_weight": "0.5", "random_state": "42"}
    assert meta["production_metrics"] == {"ndcg_at_10": 0.0831}
    assert meta["run_id"] == logged.run_id and meta["registry"] != ""

    with open(os.path.join("serving_model", "movies.json"), encoding="utf-8") as f:
        movies = json.load(f)
    assert movies[0] == {"movie_id": 10, "title": "Ten (1990)", "year": 1990}
    assert movies[-1] == {"movie_id": 50, "title": "No Year", "year": None}

    # The exported state recommends exactly like the registered model.
    restored = BlendRecommender.load_state(export_champion.state_path("serving_model"))
    assert np.array_equal(restored.recommend(USERS, n=3), blend.recommend(USERS, n=3))


def test_export_replaces_a_previous_export(registry, blend, tmp_path):
    register_champion(blend, tmp_path)
    export_champion.export(serving_dir="serving_model")
    leftover = os.path.join("serving_model", "model", "leftover.txt")
    with open(leftover, "w", encoding="utf-8") as f:
        f.write("from an older export")

    export_champion.export(serving_dir="serving_model")

    assert not os.path.exists(leftover)


def test_export_without_a_champion_is_a_clear_error(registry):
    with pytest.raises(SystemExit, match="No champion version"):
        export_champion.export(serving_dir="serving_model")


def test_export_needs_the_movie_titles_locally_unless_remote(registry, blend, tmp_path):
    register_champion(blend, tmp_path, with_movies=False)

    with pytest.raises(SystemExit, match="use --remote to fetch it from R2"):
        export_champion.export(serving_dir="serving_model")


def test_remote_export_fetches_the_titles_from_r2(registry, blend, tmp_path, storage, monkeypatch):
    register_champion(blend, tmp_path, with_movies=False)
    source = tmp_path / "movies.parquet"
    pd.DataFrame({"movieId": np.array([10, 20, 30, 40, 50], dtype="int32"), "title": ["A (2001)"] * 5,
                  "genres": ["Drama"] * 5}).to_parquet(source, index=False)
    storage.upload(str(source), "processed/2019-07-01/movies.parquet")
    monkeypatch.setattr(export_champion.Storage, "from_env", classmethod(lambda cls: storage))

    export_champion.export(remote=True, serving_dir="serving_model")

    assert registry == [True]
    with open(os.path.join("serving_model", "movies.json"), encoding="utf-8") as f:
        assert json.load(f)[0] == {"movie_id": 10, "title": "A (2001)", "year": 2001}
