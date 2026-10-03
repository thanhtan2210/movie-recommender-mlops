"""The promotion gate and the pipeline, on synthetic data with a local MLflow store."""
import json
import os

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src import evaluate, pipeline, tracking
from src.data import read_parquet
from tests.conftest import CUTOFF, DAY
from tests.test_train_evaluate import TRAIN_YAML, write_raw

DATA_YAML = ("min_item_ratings: 5\npositive_threshold: 4.0\ntest_window_days: 30\n"
             "min_user_train_positives: 3\ncutoff: \"2019-06-01\"\n")
NEXT_CUTOFF = CUTOFF + 30 * DAY  # 2019-07-01


def difference(low, value=0.01, high=0.02):
    return {"ndcg_at_10": {"value": value, "ci95_low": low, "ci95_high": high}}


# ---------------------------------------------------------------- the rule


def test_gate_promotes_when_the_lower_bound_is_above_zero():
    decision = pipeline.gate_decision(difference(low=0.0001), "blend", "popularity")

    assert decision["promote"] is True and decision["decision"] == "promote"
    assert "lower bound is above 0" in decision["reason"]


@pytest.mark.parametrize("low", [0.0, -0.0001, -0.02])
def test_gate_keeps_the_champion_when_the_lower_bound_is_not_above_zero(low):
    decision = pipeline.gate_decision(difference(low=low), "blend", "popularity")

    assert decision["promote"] is False and decision["decision"] == "keep"
    assert "not above 0" in decision["reason"]


def test_gate_keeps_a_champion_of_the_same_type():
    decision = pipeline.gate_decision(difference(low=0.0, value=0.0, high=0.0), "blend", "blend")

    assert decision["promote"] is False
    assert "already a blend model" in decision["reason"]


# ---------------------------------------------------------------- the pipeline


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A working directory with raw data (two months after the first cutoff), configs and a local MLflow store."""
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path / "raw", months_after=2)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    (tmp_path / "configs" / "train.yaml").write_text(TRAIN_YAML, encoding="utf-8")
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_HIT_RATE", 1.1)
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_NDCG", 1.1)
    # No R2 and no DagsHub: the developer's .env must not be picked up.
    monkeypatch.setattr(pipeline, "connect_storage", lambda: None)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(tracking, "setup", lambda tracking_uri=None, _setup=tracking.setup: _setup(uri))
    yield tmp_path
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


def run(cutoff, project):
    return pipeline.run_pipeline(cutoff, source=f"local:{project / 'raw'}")


def read_report(cutoff, name):
    with open(os.path.join("reports", cutoff, name), encoding="utf-8") as f:
        return json.load(f)


def test_first_run_registers_popularity_then_promotes_the_challenger(project):
    outcome = run("2019-06-01", project)

    gate = read_report("2019-06-01", "gate.json")
    production = read_report("2019-06-01", "production.json")
    # Windows: the gate is the last 30 days before the cutoff, production the 30 days after it.
    assert gate["gate_window_utc"] == {"from": "2019-05-02T00:00:00Z", "to_exclusive": "2019-06-01T00:00:00Z"}
    assert gate["models_trained_on"] == "ratings before 2019-05-02"
    assert production["production_window_utc"] == {"from": "2019-06-01T00:00:00Z", "to_exclusive": "2019-07-01T00:00:00Z"}

    # The registry was empty: popularity became version 1, the champion the challenger had to beat.
    assert gate["initial_champion_registered"]["registry_version"] == "1"
    assert gate["champion"]["model_type"] == "popularity" and gate["challenger"]["model_type"] == "blend"
    # The synthetic users have clear tastes, so the blend wins by a wide margin.
    assert gate["difference_challenger_minus_champion"]["ndcg_at_10"]["ci95_low"] > 0
    assert gate["promote"] is True and outcome["champion_type"] == "blend"

    assert production["champion"]["model_type"] == "blend" and production["champion"]["registry_version"] == "2"
    assert production["champion"]["marked_as_champion_by"] == "alias"
    assert production["difference_champion_minus_popularity"]["ndcg_at_10"]["value"] == pytest.approx(
        production["champion"]["ndcg_at_10"]["value"] - production["popularity"]["ndcg_at_10"]["value"])

    champion = tracking.get_champion()
    assert champion["version"] == "2" and champion["model_type"] == "blend"
    assert champion["tags"]["gate_decision"] == "promote" and champion["tags"]["cutoff"] == "2019-06-01"

    # Runs: initial champion, two gate runs, champion and baseline on the production window.
    client = MlflowClient()
    experiment = client.get_experiment_by_name(tracking.EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert sorted(r.data.tags["window"] for r in runs) == ["gate", "gate", "gate", "production", "production"]
    by_name = {r.info.run_name: r for r in runs}
    challenger = by_name["gate-challenger-blend"]
    assert challenger.data.params["trained_before"] == "2019-05-02" and challenger.data.params["k"] == "4"
    assert challenger.data.tags["decision"] == "promote"
    assert challenger.data.metrics["ndcg_at_10"] == pytest.approx(gate["challenger"]["ndcg_at_10"]["value"])
    refit = by_name["production-champion-blend"]
    assert refit.data.params["trained_before"] == "2019-06-01"
    assert refit.data.metrics["ndcg_at_10"] == pytest.approx(production["champion"]["ndcg_at_10"]["value"])
    assert {"model", "stats.json"} <= {a.path for a in client.list_artifacts(refit.info.run_id)}

    # The registered champion serves recommendations, also for a user it has never seen.
    served = tracking.load_champion().predict(np.array([2, 999999]), params={"n": 5})
    assert served.shape == (2, 5) and (served > 0).all()


def test_time_windows_do_not_overlap(project):
    run("2019-06-01", project)
    directory = os.path.join("data", "processed", "2019-06-01")
    train, production = read_parquet(directory, "train.parquet"), read_parquet(directory, "test.parquet")
    movies = read_parquet(directory, "movies.parquet")
    config = pipeline.load_data_config(cutoff="2019-06-01")

    gate_train, gate_eval, gate_config = pipeline.inner_split(train, movies, config, pipeline.GATE_DAYS)

    gate_start = CUTOFF - 30 * DAY
    assert gate_config.cutoff_timestamp == gate_start
    assert gate_train["timestamp"].max() < gate_start            # gate models never see the gate window
    assert gate_eval["timestamp"].min() >= gate_start and gate_eval["timestamp"].max() < CUTOFF
    assert train["timestamp"].max() < CUTOFF                     # the refit never sees the production window
    assert production["timestamp"].min() >= CUTOFF               # production has nothing from before the cutoff
    assert production["timestamp"].max() < CUTOFF + 30 * DAY


def test_second_run_keeps_a_champion_of_the_same_type_and_refits_it(project):
    run("2019-06-01", project)

    outcome = run("2019-07-01", project)

    gate = read_report("2019-07-01", "gate.json")
    assert gate["initial_champion_registered"] is None
    assert gate["champion"]["model_type"] == "blend" and gate["champion"]["registry_version"] == "2"
    assert gate["difference_challenger_minus_champion"]["ndcg_at_10"] == {"value": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    assert gate["promote"] is False and "already a blend model" in gate["reason"]
    # The champion type is kept, but refitted on data up to the new cutoff and registered as a new version.
    assert outcome["champion_type"] == "blend" and outcome["champion_version"] == "3"
    assert tracking.get_champion()["version"] == "3"
    production = read_report("2019-07-01", "production.json")
    assert production["production_window_utc"]["from"] == "2019-07-01T00:00:00Z"
    assert read_parquet(os.path.join("data", "processed", "2019-07-01"), "test.parquet")["timestamp"].min() >= NEXT_CUTOFF


def test_champion_is_kept_when_the_challenger_does_not_clear_the_gate(project, monkeypatch):
    monkeypatch.setattr(pipeline, "gate_decision", lambda difference, challenger, champion: {
        "rule": pipeline.RULE, "decision": "keep", "promote": False, "reason": "forced for the test"})

    outcome = run("2019-06-01", project)

    assert outcome["champion_type"] == "popularity"
    production = read_report("2019-06-01", "production.json")
    assert production["champion"]["model_type"] == "popularity"
    assert production["difference_champion_minus_popularity"]["ndcg_at_10"]["value"] == 0.0
    assert tracking.get_champion()["model_type"] == "popularity"
    assert tracking.get_champion()["tags"]["gate_decision"] == "keep"


def test_pipeline_needs_r2_or_a_local_source(project):
    with pytest.raises(SystemExit, match="R2 is not configured"):
        pipeline.run_pipeline("2019-06-01")
