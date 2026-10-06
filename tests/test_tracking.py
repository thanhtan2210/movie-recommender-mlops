"""MLflow tracking and registry, always against a local store in a temporary directory."""
import json
import os

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src import metrics, tracking
from src.baseline import PopularityRecommender
from src.model import BlendRecommender
from tests.conftest import CUTOFF
from tests.test_recommender import ROWS, ratings_frame

USERS = np.array([1, 2, 3, 4, 5, 6, 7, 999])


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A fresh MLflow store; nothing leaves the temporary directory."""
    monkeypatch.chdir(tmp_path)
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tracking, "load_dotenv", lambda: None)
    # An absolute path per test: MLflow caches stores by URI, so a relative one would be shared.
    uri = tracking.setup(tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield uri
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)  # mlflow.set_tracking_uri exports it; do not leak into other tests


@pytest.fixture
def stats_path(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"sha256": {"train.parquet": "abc"}}), encoding="utf-8")
    return str(path)


@pytest.fixture
def blend():
    return BlendRecommender.fit(ratings_frame(rows=ROWS), "2019-06-01", CUTOFF, 4.0, k=2, train_window="1y",
                                blend_weight=0.5)


@pytest.fixture
def popularity():
    return PopularityRecommender.fit(ratings_frame(rows=ROWS), CUTOFF, 4.0)


def fake_result():
    recommended = np.array([[0, 1], [2, 3]])
    targets = np.zeros((2, 5), dtype=bool)
    targets[0, 1] = True
    return metrics.evaluate(recommended, targets, np.zeros(5, dtype=bool), metrics.bootstrap_indices(2))


def test_setup_is_local_by_default_even_when_a_server_is_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.example/user/repo.mlflow")
    monkeypatch.setattr(tracking, "load_dotenv", lambda: pytest.fail(".env must not be read without remote=True"))

    uri = tracking.setup()

    assert uri == tracking.LOCAL_TRACKING_URI == mlflow.get_tracking_uri()
    assert mlflow.get_experiment_by_name(tracking.EXPERIMENT) is not None
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


def test_remote_setup_needs_a_configured_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(tracking, "load_dotenv", lambda: None)

    with pytest.raises(SystemExit, match="--remote needs MLFLOW_TRACKING_URI"):
        tracking.setup(remote=True)


def test_remote_setup_uses_the_configured_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'other.db').as_posix()}")

    uri = tracking.setup(remote=True)

    assert uri.endswith("other.db") and mlflow.get_tracking_uri() == uri
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.mark.parametrize("name", ["blend", "popularity"])
def test_pyfunc_round_trip_gives_the_same_recommendations(name, request, tmp_path):
    recommender = request.getfixturevalue(name)

    loaded = mlflow.pyfunc.load_model(tracking.save_pyfunc(recommender, str(tmp_path / name)))

    assert np.array_equal(loaded.predict(USERS, params={"n": 3}), recommender.recommend(USERS, n=3))
    assert loaded.predict(USERS).shape == (len(USERS), 10)            # n defaults to 10
    assert np.array_equal(loaded.predict(USERS.tolist(), params={"n": 3}), recommender.recommend(USERS, n=3))
    assert np.array_equal(loaded.predict(pd.DataFrame({"userId": USERS.astype("int32")}), params={"n": 3}),
                          recommender.recommend(USERS, n=3))
    # The unknown user gets the popularity list.
    assert loaded.predict(np.array([999]), params={"n": 3}).tolist() == [[50, 10, 20]]


def test_evaluation_only_run_logs_params_metrics_and_tags_but_no_artifacts(local_mlflow, blend, stats_path):
    logged = tracking.log_run("gate-challenger", blend, {"k": 2, "cutoff": "2019-07-01"}, {"window": "gate"},
                              stats_path, result=fake_result())

    run = MlflowClient().get_run(logged.run_id)
    assert run.data.params["model_type"] == "blend" and run.data.tags["window"] == "gate"
    assert "ndcg_at_10" in run.data.metrics
    assert MlflowClient().list_artifacts(logged.run_id) == []
    assert logged.version is None


def test_registered_run_records_params_metrics_tags_and_artifacts(local_mlflow, blend, stats_path):
    result = fake_result()

    logged = tracking.log_run("production-champion", blend,
                              params={"k": 2, "train_window": "1y", "blend_weight": 0.5, "cutoff": "2019-07-01",
                                      "train_sha256": "abc", "git_commit": "deadbeef"},
                              tags={"window": "gate"}, stats_path=stats_path, result=result, register=True)

    run = MlflowClient().get_run(logged.run_id)
    assert run.data.params["model_type"] == "blend" and run.data.params["k"] == "2"
    assert run.data.params["train_sha256"] == "abc" and run.data.params["git_commit"] == "deadbeef"
    assert run.data.tags["window"] == "gate"
    assert run.data.metrics["ndcg_at_10"] == pytest.approx(result["ndcg_at_10"]["value"])
    for name in tracking.METRICS_WITH_INTERVAL:
        assert f"{name}_ci95_low" in run.data.metrics and f"{name}_ci95_high" in run.data.metrics
    assert "catalog_coverage" in run.data.metrics
    artifacts = {item.path for item in MlflowClient().list_artifacts(logged.run_id)}
    assert {"model", "stats.json"} <= artifacts
    assert logged.version == "1"

    # The logged model loads and recommends like the original.
    loaded = mlflow.pyfunc.load_model(f"runs:/{logged.run_id}/model")
    assert np.array_equal(loaded.predict(USERS, params={"n": 3}), blend.recommend(USERS, n=3))


def test_registry_champion_moves_between_versions(local_mlflow, blend, popularity, stats_path):
    assert tracking.get_champion() is None
    with pytest.raises(LookupError):
        tracking.load_champion()

    first = tracking.log_run("initial", popularity, {}, {"window": "gate"}, stats_path, register=True,
                             version_tags={"cutoff": "2019-07-01"})
    assert tracking.set_champion(first.version) == "alias"
    champion = tracking.get_champion()
    assert champion["version"] == first.version == "1"
    assert champion["model_type"] == "popularity" and champion["tags"]["cutoff"] == "2019-07-01"

    second = tracking.log_run("refit", blend, {}, {"window": "production"}, stats_path, register=True)
    tracking.set_champion(second.version)

    champion = tracking.get_champion()
    assert champion["version"] == "2" and champion["model_type"] == "blend"
    client = MlflowClient()
    assert tracking.ROLE_TAG not in client.get_model_version(tracking.REGISTERED_MODEL, "1").tags
    assert client.get_model_version(tracking.REGISTERED_MODEL, "2").tags[tracking.ROLE_TAG] == tracking.CHAMPION
    assert np.array_equal(tracking.load_champion().predict(USERS, params={"n": 3}), blend.recommend(USERS, n=3))


def test_champion_works_through_tags_when_the_server_has_no_aliases(local_mlflow, popularity, blend, stats_path,
                                                                    monkeypatch):
    def unsupported(self, *args, **kwargs):
        raise MlflowException("aliases are not supported by this server")

    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", unsupported)
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", unsupported)

    first = tracking.log_run("initial", popularity, {}, {"window": "gate"}, stats_path, register=True)
    assert tracking.set_champion(first.version) == "tag"
    assert tracking.get_champion()["version"] == "1"

    second = tracking.log_run("refit", blend, {}, {"window": "production"}, stats_path, register=True)
    assert tracking.set_champion(second.version) == "tag"
    assert tracking.get_champion()["version"] == "2" and tracking.get_champion()["model_type"] == "blend"


def test_flatten_metrics_names():
    flat = tracking.flatten_metrics(fake_result())

    assert set(flat) == {"catalog_coverage", *tracking.METRICS_WITH_INTERVAL,
                         *(f"{m}_ci95_low" for m in tracking.METRICS_WITH_INTERVAL),
                         *(f"{m}_ci95_high" for m in tracking.METRICS_WITH_INTERVAL)}
    assert os.path.basename(tracking.SOURCE_DIR) == "src"
