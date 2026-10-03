"""train.py and evaluate.py end to end on a small synthetic dataset (no network)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src import evaluate, prepare as prep, train
from src.config import DataConfig
from src.data import build_interactions, read_parquet
from src.model import PureSVD, artifact_dir, load_meta
from tests.conftest import CUTOFF, DAY

VAL_CUTOFF = CUTOFF - 31 * DAY  # 2019-05-01
DATA_CONFIG = DataConfig(min_item_ratings=5, positive_threshold=4.0, test_window_days=30,
                         min_user_train_positives=3, cutoff="2019-06-01")


def synthetic_ratings(n_users=120, n_items=40, seed=0) -> pd.DataFrame:
    """Two taste groups: even users like movies 1-20, odd users like movies 21-40.

    Each user rates 14 movies of their group (4 or 5 stars) and 4 of the other
    group (2 stars), spread over January-April, May and June 2019.
    """
    rng = np.random.default_rng(seed)
    group_a, group_b = np.arange(1, n_items // 2 + 1), np.arange(n_items // 2 + 1, n_items + 1)
    rows = []
    for user in range(1, n_users + 1):
        own, other = (group_a, group_b) if user % 2 == 0 else (group_b, group_a)
        movies = np.concatenate([rng.choice(own, 14, replace=False), rng.choice(other, 4, replace=False)])
        ratings = np.concatenate([rng.choice([4.0, 5.0], 14), np.full(4, 2.0)])
        order = rng.permutation(18)
        times = np.concatenate([
            rng.integers(VAL_CUTOFF - 120 * DAY, VAL_CUTOFF, 10),      # before May
            rng.integers(VAL_CUTOFF, CUTOFF, 4),                        # May: validation window
            rng.integers(CUTOFF, CUTOFF + 30 * DAY, 4),                 # June: test window
        ])
        rows += [(user, int(movies[i]), float(ratings[i]), int(t)) for i, t in zip(order, times)]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return frame.astype({"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An empty working directory holding data/processed/2019-06-01 and a small train config."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_ratings().to_csv(raw / "ratings.csv", index=False)
    titles = [f"Movie {i} ({1999 if i <= 20 else 2018})" for i in range(1, 41)]
    pd.DataFrame({"movieId": range(1, 41), "title": titles,
                  "genres": ["A"] * 20 + ["B"] * 20}).to_csv(raw / "movies.csv", index=False)
    prep.run(DATA_CONFIG, source=f"local:{raw}")
    (tmp_path / "train.yaml").write_text("k_values: [2, 4]\nvalidation_days: 31\nrandom_state: 42\n", encoding="utf-8")
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


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- validation split


def test_validation_config_moves_the_cutoff_back():
    val = train.validation_config(DATA_CONFIG, 31)

    assert val.cutoff == "2019-05-01"
    assert val.cutoff_timestamp == VAL_CUTOFF
    assert val.test_end_timestamp == CUTOFF          # the validation window ends where the test window starts
    assert val.min_item_ratings == DATA_CONFIG.min_item_ratings
    assert val.min_user_train_positives == DATA_CONFIG.min_user_train_positives


def test_validation_split_stays_before_the_cutoff(workspace):
    directory = os.path.join("data", "processed", "2019-06-01")
    full_train = read_parquet(directory, "train.parquet")
    movies = read_parquet(directory, "movies.parquet")

    train_val, validation, _ = train.validation_split(full_train, movies, DATA_CONFIG, 31)

    assert len(train_val) and len(validation)
    assert train_val["timestamp"].max() < VAL_CUTOFF
    assert validation["timestamp"].min() >= VAL_CUTOFF
    assert validation["timestamp"].max() < CUTOFF     # nothing from the cutoff onwards
    assert (validation["rating"] >= 4.0).all()
    # Every validation row is a row of train: validation is carved out of train, not out of test.
    full_train = read_parquet(directory, "train.parquet")
    merged = validation.merge(full_train, on=["userId", "movieId", "rating", "timestamp"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    test = read_parquet(directory, "test.parquet")
    assert test["timestamp"].min() >= CUTOFF


def test_choose_k_takes_the_best_ndcg_and_the_smaller_k_on_a_tie():
    assert train.choose_k([{"k": 32, "ndcg_at_10": 0.10}, {"k": 64, "ndcg_at_10": 0.12},
                           {"k": 128, "ndcg_at_10": 0.11}]) == 64
    assert train.choose_k([{"k": 64, "ndcg_at_10": 0.12}, {"k": 32, "ndcg_at_10": 0.12}]) == 32


# ---------------------------------------------------------------- end to end


def test_train_writes_the_sweep_and_the_model_without_reading_the_test_set(workspace, monkeypatch):
    original = train.read_parquet

    def guarded(directory, name):
        assert name != "test.parquet", "training must not read the test set"
        return original(directory, name)

    monkeypatch.setattr(train, "read_parquet", guarded)
    run_train(monkeypatch)

    sweep = pd.read_csv(os.path.join("reports", "2019-06-01", "validation.csv"))
    assert sweep["k"].tolist() == [2, 4]
    assert sweep["selected"].sum() == 1
    assert (sweep["validation_users"] > 0).all()
    best_k = int(sweep.loc[sweep["ndcg_at_10"].idxmax(), "k"])

    meta = load_meta(artifact_dir("2019-06-01"))
    stats = json.loads(read_text(os.path.join("data", "processed", "2019-06-01", "stats.json")))
    assert meta["k"] == best_k == int(sweep.loc[sweep["selected"], "k"].iloc[0])
    assert meta["train_sha256"] == stats["sha256"]["train.parquet"]
    assert meta["validation"]["cutoff"] == "2019-05-01"
    model = PureSVD.load(artifact_dir("2019-06-01"))
    assert model.item_factors.shape == (meta["n_items"], best_k)
    full_train = read_parquet(os.path.join("data", "processed", "2019-06-01"), "train.parquet")
    interactions = build_interactions(full_train, 4.0)
    assert np.array_equal(model.item_ids, interactions.item_ids)
    assert np.array_equal(model.user_ids, interactions.user_ids)


def test_training_twice_gives_the_same_model_and_sweep(workspace, monkeypatch):
    run_train(monkeypatch)
    first_meta = load_meta(artifact_dir("2019-06-01"))
    first_sweep = read_text(os.path.join("reports", "2019-06-01", "validation.csv"))

    run_train(monkeypatch)

    assert load_meta(artifact_dir("2019-06-01"))["item_factors_sha256"] == first_meta["item_factors_sha256"]
    assert read_text(os.path.join("reports", "2019-06-01", "validation.csv")) == first_sweep


def test_evaluate_writes_metrics_for_both_recommenders(workspace, monkeypatch):
    run_train(monkeypatch)
    run_evaluate(monkeypatch)

    path = os.path.join("reports", "2019-06-01", "test_metrics.json")
    report = json.loads(read_text(path))
    test = read_parquet(os.path.join("data", "processed", "2019-06-01"), "test.parquet")
    assert report["evaluated_users"] == test["userId"].nunique()
    assert report["test_rows"] == len(test)
    assert report["targets_already_rated_in_train"] == 0
    assert set(report["models"]) == {"pure_svd", "popularity"}
    for result in report["models"].values():
        for name in ("hit_rate_at_10", "recall_at_10", "ndcg_at_10", "long_tail_share"):
            assert 0.0 <= result[name]["ci95_low"] <= result[name]["value"] <= result[name]["ci95_high"] <= 1.0
        assert 0.0 < result["catalog_coverage"]["value"] <= 1.0
    difference = report["difference_pure_svd_minus_popularity"]
    assert difference["ndcg_at_10"]["value"] == pytest.approx(
        report["models"]["pure_svd"]["ndcg_at_10"]["value"] - report["models"]["popularity"]["ndcg_at_10"]["value"])
    # Users have clear tastes here, so the model must beat popularity; a failure means the pipeline is broken.
    assert difference["ndcg_at_10"]["ci95_low"] > 0

    examples = json.loads(read_text(os.path.join("reports", "2019-06-01", "examples.json")))
    assert len(examples) == 3
    for example in examples:
        assert len(example["pure_svd_top_10"]) == 10 and len(example["popularity_top_10"]) == 10
        assert len(example["last_liked_in_train"]) <= 5
        assert not set(example["pure_svd_top_10"]) & set(example["last_liked_in_train"])

    # Evaluating again reproduces the report exactly.
    run_evaluate(monkeypatch)
    assert read_text(path) == json.dumps(report, indent=2) + "\n"


def test_evaluate_refuses_an_implausible_result(workspace, monkeypatch):
    run_train(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_PLAUSIBLE_HIT_RATE", 0.0)

    with pytest.raises(SystemExit, match="suspect leakage"):
        run_evaluate(monkeypatch)

    assert not os.path.exists(os.path.join("reports", "2019-06-01", "test_metrics.json"))


def test_evaluate_refuses_a_model_trained_on_other_data(workspace, monkeypatch):
    run_train(monkeypatch)
    meta_path = os.path.join(artifact_dir("2019-06-01"), "model_meta.json")
    meta = json.loads(read_text(meta_path))
    meta["train_sha256"] = "0" * 64
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(SystemExit, match="different train.parquet"):
        run_evaluate(monkeypatch)


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
