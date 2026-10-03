"""train.py and evaluate.py on a small synthetic dataset (no network)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src import evaluate, prepare as prep, train
from src.baseline import PopularityRecommender
from src.config import DataConfig
from src.data import read_parquet
from src.model import BlendRecommender
from tests.conftest import CUTOFF, DAY

INNER_CUTOFF = CUTOFF - 30 * DAY  # 2019-05-02
DATA_CONFIG = DataConfig(min_item_ratings=5, positive_threshold=4.0, test_window_days=30,
                         min_user_train_positives=3, cutoff="2019-06-01")
TRAIN_YAML = "k: 4\ntrain_window: 1y\nblend_weight: 1.0\nrandom_state: 42\n"


def synthetic_ratings(n_users=120, n_items=40, seed=0, months_after=1) -> pd.DataFrame:
    """Two taste groups: even users like movies 1-20, odd users like movies 21-40.

    Each user rates movies of their group with 4 or 5 stars and a few of the
    other group with 2 stars: 10 ratings in the four months before the last
    30 days of train, 4 in those last 30 days, and 4 in each 30-day window
    after the cutoff.
    """
    rng = np.random.default_rng(seed)
    group_a, group_b = np.arange(1, n_items // 2 + 1), np.arange(n_items // 2 + 1, n_items + 1)
    per_user = 14 + 4 * months_after
    rows = []
    for user in range(1, n_users + 1):
        own, other = (group_a, group_b) if user % 2 == 0 else (group_b, group_a)
        n_own = per_user - 4
        movies = np.concatenate([rng.choice(own, n_own, replace=False), rng.choice(other, 4, replace=False)])
        ratings = np.concatenate([rng.choice([4.0, 5.0], n_own), np.full(4, 2.0)])
        order = rng.permutation(per_user)
        times = [rng.integers(INNER_CUTOFF - 120 * DAY, INNER_CUTOFF, 10), rng.integers(INNER_CUTOFF, CUTOFF, 4)]
        for month in range(months_after):
            times.append(rng.integers(CUTOFF + month * 30 * DAY, CUTOFF + (month + 1) * 30 * DAY, 4))
        rows += [(user, int(movies[i]), float(ratings[i]), int(t)) for i, t in zip(order, np.concatenate(times))]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return frame.astype({"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})


def write_raw(directory, months_after=1):
    directory.mkdir()
    synthetic_ratings(months_after=months_after).to_csv(directory / "ratings.csv", index=False)
    titles = [f"Movie {i} ({1999 if i <= 20 else 2018})" for i in range(1, 41)]
    pd.DataFrame({"movieId": range(1, 41), "title": titles,
                  "genres": ["A"] * 20 + ["B"] * 20}).to_csv(directory / "movies.csv", index=False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An empty working directory holding data/processed/2019-06-01 and a small train config."""
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path / "raw")
    prep.run(DATA_CONFIG, source=f"local:{tmp_path / 'raw'}")
    (tmp_path / "train.yaml").write_text(TRAIN_YAML, encoding="utf-8")
    # The synthetic tastes are far easier than real data; the leakage guard is tested separately.
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_HIT_RATE", 1.1)
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_NDCG", 1.1)
    return tmp_path


def run_train(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--cutoff", "2019-06-01", "--config", "train.yaml"])
    train.main()


def run_evaluate(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--cutoff", "2019-06-01"])
    evaluate.main()


def load_frames():
    directory = os.path.join("data", "processed", "2019-06-01")
    return (read_parquet(directory, "train.parquet"), read_parquet(directory, "test.parquet"),
            read_parquet(directory, "movies.parquet"))


def read_json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- configuration and splits


def test_repo_train_config_holds_the_frozen_parameters():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "train.yaml")

    assert train.load_train_config(path) == train.TrainConfig(k=64, train_window="1y", blend_weight=4.0, random_state=42)


def test_inner_split_stays_before_the_cutoff(workspace):
    full_train, test, movies = load_frames()

    inner_train, inner_eval, inner_config = train.inner_split(full_train, movies, DATA_CONFIG, 30)

    assert inner_config.cutoff == "2019-05-02" and inner_config.test_end_timestamp == CUTOFF
    assert len(inner_train) and len(inner_eval)
    assert inner_train["timestamp"].max() < INNER_CUTOFF          # nothing from the window in the training part
    assert inner_eval["timestamp"].min() >= INNER_CUTOFF
    assert inner_eval["timestamp"].max() < CUTOFF                 # nothing from the cutoff onwards
    assert (inner_eval["rating"] >= 4.0).all()
    # The window is carved out of train: every row of it is a row of train, none is a row of test.
    full_train, test, _ = load_frames()
    keys = ["userId", "movieId", "rating", "timestamp"]
    assert (inner_eval.merge(full_train, on=keys, how="left", indicator=True)["_merge"] == "both").all()
    assert test["timestamp"].min() >= CUTOFF


def test_fit_model_builds_each_type(workspace):
    full_train, _, _ = load_frames()
    config = train.TrainConfig(k=4, train_window="1y", blend_weight=1.0)

    blend = train.fit_model(train.BLEND, full_train, DATA_CONFIG, config)
    popularity = train.fit_model(train.POPULARITY, full_train, DATA_CONFIG, config)

    assert isinstance(blend, BlendRecommender) and blend.svd.k == 4 and blend.svd.blend_weight == 1.0
    assert isinstance(popularity, PopularityRecommender)
    assert train.model_params(train.BLEND, config) == {"k": 4, "train_window": "1y", "blend_weight": 1.0,
                                                       "random_state": 42}
    assert train.model_params(train.POPULARITY, config) == {}
    with pytest.raises(ValueError):
        train.fit_model("two-tower", full_train, DATA_CONFIG, config)


# ---------------------------------------------------------------- evaluation


def test_evaluate_window_scores_recommenders_on_the_same_users(workspace):
    full_train, test, _ = load_frames()
    config = train.TrainConfig(k=4, train_window="1y", blend_weight=1.0)
    models = {name: train.fit_model(name, full_train, DATA_CONFIG, config) for name in (train.BLEND, train.POPULARITY)}

    evaluation = evaluate.evaluate_window(models, test)

    assert evaluation.users.tolist() == sorted(test["userId"].unique())
    assert evaluation.targets.sum() == len(test)
    for name, recs in evaluation.recommendations.items():
        assert recs.shape == (len(evaluation.users), 10)
        # No recommended movie was already rated in train.
        seen = models[name].seen[np.searchsorted(models[name].user_ids, evaluation.users)].toarray() > 0
        assert not np.take_along_axis(seen, recs, axis=1).any()
    for result in evaluation.results.values():
        for metric in ("hit_rate_at_10", "recall_at_10", "ndcg_at_10", "long_tail_share"):
            assert 0.0 <= result[metric]["ci95_low"] <= result[metric]["value"] <= result[metric]["ci95_high"] <= 1.0
    difference = evaluation.difference(train.BLEND, train.POPULARITY)
    assert difference["ndcg_at_10"]["value"] == pytest.approx(
        evaluation.results["blend"]["ndcg_at_10"]["value"] - evaluation.results["popularity"]["ndcg_at_10"]["value"])
    # Users have clear tastes here, so the model must beat popularity; a failure means the pipeline is broken.
    assert difference["ndcg_at_10"]["ci95_low"] > 0


def test_evaluate_window_rejects_models_fitted_on_different_data(workspace):
    full_train, test, _ = load_frames()
    popularity = PopularityRecommender.fit(full_train, CUTOFF, 4.0)
    other = PopularityRecommender.fit(full_train[full_train["movieId"] != full_train["movieId"].iloc[0]], CUTOFF, 4.0)

    with pytest.raises(ValueError, match="same training data"):
        evaluate.evaluate_window({"a": popularity, "b": other}, test)


# ---------------------------------------------------------------- command line


def test_train_and_evaluate_commands(workspace, monkeypatch):
    run_train(monkeypatch)

    meta = read_json(train.artifact_dir("2019-06-01"), train.META_FILE)
    stats = read_json("data", "processed", "2019-06-01", "stats.json")
    assert meta["model_type"] == "blend" and meta["k"] == 4 and meta["train_window"] == "1y"
    assert meta["train_sha256"] == stats["sha256"]["train.parquet"]

    run_evaluate(monkeypatch)
    report = read_json(train.artifact_dir("2019-06-01"), evaluate.EVALUATION_FILE)
    assert set(report["models"]) == {"blend", "popularity"}
    assert report["evaluated_users"] > 0

    # Training and evaluating again gives the same model and the same numbers.
    run_train(monkeypatch)
    assert read_json(train.artifact_dir("2019-06-01"), train.META_FILE)["item_factors_sha256"] == meta["item_factors_sha256"]
    run_evaluate(monkeypatch)
    assert read_json(train.artifact_dir("2019-06-01"), evaluate.EVALUATION_FILE) == report


def test_evaluate_refuses_an_implausible_result(workspace, monkeypatch):
    run_train(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_HIT_RATE", 0.0)

    with pytest.raises(SystemExit, match="suspect leakage"):
        run_evaluate(monkeypatch)

    assert not os.path.exists(os.path.join(train.artifact_dir("2019-06-01"), evaluate.EVALUATION_FILE))


def test_missing_processed_data_without_r2_is_a_clear_error(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg, "load_dotenv", lambda: None)
    for name in cfg.R2_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["train", "--cutoff", "2019-06-01", "--config",
                                      os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "train.yaml")])

    with pytest.raises(FileNotFoundError, match="python -m src.prepare"):
        train.main()
