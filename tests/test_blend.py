import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src import blend
from src.data import build_interactions
from src.model import PureSVD, artifact_dir, load_meta
from tests.conftest import CUTOFF, DAY, ratings_frame
from tests.test_train_evaluate import run_evaluate, run_train, workspace  # noqa: F401  (fixture)

REPORTS = os.path.join("reports", "2019-06-01")


def read_text(name):
    with open(os.path.join(REPORTS, name), encoding="utf-8") as f:
        return f.read()


def run_blend(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["blend", "--cutoff", "2019-06-01", "--config", "blend.yaml"])
    blend.main()


@pytest.fixture
def trained(workspace, monkeypatch):  # noqa: F811
    """The synthetic workspace after src.train and src.evaluate, with a small blend grid."""
    (workspace / "configs").mkdir()
    (workspace / "configs" / "train.yaml").write_text(
        "k_values: [2, 4]\nvalidation_days: 31\nrandom_state: 42\n", encoding="utf-8")
    (workspace / "blend.yaml").write_text(
        "k: 2\ntrain_windows: [all, 1y]\nblend_weights: [0, 0.5, 4]\n", encoding="utf-8")
    run_train(monkeypatch)
    run_evaluate(monkeypatch)
    return workspace


def test_window_start():
    assert blend.window_start("2019-06-01", "all") is None
    assert blend.window_start("2019-06-01", "1y") == 1527811200    # 2018-06-01T00:00:00Z
    assert blend.window_start("2019-06-01", "3y") == 1464739200    # 2016-06-01T00:00:00Z
    assert blend.window_start("2020-02-29", "1y") == 1551312000    # 2019-02-28T00:00:00Z
    with pytest.raises(ValueError):
        blend.window_start("2019-06-01", "6m")


def test_liked_window_keeps_the_index_and_the_seen_movies():
    train = ratings_frame(rows=[(1, 10, 5.0, CUTOFF - 500 * DAY), (1, 20, 4.0, CUTOFF - 10 * DAY),
                                (2, 10, 2.0, CUTOFF - 10 * DAY)])

    everything = build_interactions(train, 4.0)
    recent = build_interactions(train, 4.0, liked_since=CUTOFF - 365 * DAY)

    assert recent.user_ids.tolist() == everything.user_ids.tolist() == [1, 2]
    assert recent.item_ids.tolist() == everything.item_ids.tolist() == [10, 20]
    assert everything.liked.toarray().tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert recent.liked.toarray().tolist() == [[0.0, 1.0], [0.0, 0.0]]      # the old like is dropped
    assert recent.seen.toarray().tolist() == [[1, 1], [1, 0]]               # but it is still "seen"


def test_select_prefers_ndcg_then_the_smaller_weight_then_the_earlier_window():
    table = pd.DataFrame([
        {"train_window": "all", "blend_weight": 0.0, "ndcg_at_10": 0.05},
        {"train_window": "1y", "blend_weight": 1.0, "ndcg_at_10": 0.08},
        {"train_window": "all", "blend_weight": 1.0, "ndcg_at_10": 0.08},
        {"train_window": "all", "blend_weight": 2.0, "ndcg_at_10": 0.08},
    ])

    assert blend.select(table, ["all", "3y", "1y"]) == ("all", 1.0)


def test_blend_writes_the_grid_the_test_table_and_the_trade_off(trained, monkeypatch):
    first_test_report = read_text("test_metrics.json")
    run_blend(monkeypatch)

    grid = pd.read_csv(os.path.join(REPORTS, "validation_blend.csv"))
    blends = grid[grid["model"] == "pure_svd_blend"]
    assert len(blends) == 6 and (grid["model"] == "popularity").sum() == 1
    assert grid["selected"].sum() == 1
    # Weight 0 on all of train is the PureSVD row of the k sweep, to the last digit.
    sweep = pd.read_csv(os.path.join(REPORTS, "validation.csv"))
    plain = blends[(blends["train_window"] == "all") & (blends["blend_weight"] == 0)]
    assert plain["ndcg_at_10"].iloc[0] == pytest.approx(sweep.loc[sweep["k"] == 2, "ndcg_at_10"].iloc[0], abs=1e-12)
    assert plain["hit_rate_at_10"].iloc[0] == pytest.approx(sweep.loc[sweep["k"] == 2, "hit_rate_at_10"].iloc[0], abs=1e-12)

    report = json.loads(read_text("test_metrics_v2.json"))
    chosen = grid[grid["selected"]].iloc[0]
    assert report["test_looks"] == 2 and "Second look" in report["note"]
    assert report["selected"]["train_window"] == chosen["train_window"]
    assert report["selected"]["blend_weight"] == chosen["blend_weight"]
    assert chosen["ndcg_at_10"] == blends["ndcg_at_10"].max()
    assert list(report["models"]) == ["pure_svd", "pure_svd_blend", "popularity"]
    assert report["weight_0_all_matches_validation_csv"] is True
    difference = report["differences"]["pure_svd_blend_minus_popularity"]["ndcg_at_10"]
    assert difference["value"] == pytest.approx(
        report["models"]["pure_svd_blend"]["ndcg_at_10"]["value"] - report["models"]["popularity"]["ndcg_at_10"]["value"])
    assert difference["ci95_low"] <= difference["value"] <= difference["ci95_high"]

    tradeoff = pd.read_csv(os.path.join(REPORTS, "tradeoff.csv"))
    assert tradeoff["blend_weight"].tolist() == [0.0, 0.5, 4.0]
    assert (tradeoff["train_window"] == chosen["train_window"]).all()
    assert tradeoff["selected_on_validation"].sum() == 1
    selected_row = tradeoff[tradeoff["selected_on_validation"]].iloc[0]
    assert selected_row["ndcg_at_10"] == pytest.approx(report["models"]["pure_svd_blend"]["ndcg_at_10"]["value"])

    # The first test report is left exactly as it was.
    assert read_text("test_metrics.json") == first_test_report

    model_dir = os.path.join(artifact_dir("2019-06-01"), "blend")
    meta = load_meta(model_dir)
    assert meta["train_window"] == chosen["train_window"] and meta["blend_weight"] == chosen["blend_weight"]
    assert PureSVD.load(model_dir).k == 2


def test_blend_is_reproducible(trained, monkeypatch):
    run_blend(monkeypatch)
    first = {name: read_text(name) for name in ("validation_blend.csv", "test_metrics_v2.json", "tradeoff.csv")}

    run_blend(monkeypatch)

    assert {name: read_text(name) for name in first} == first


def test_blend_stops_before_the_test_set_when_popularity_wins_on_validation(trained, monkeypatch):
    def perfect(fold):
        """A stand-in 'popularity' that recommends each user's own targets."""
        return np.argsort(~fold.targets, axis=1, kind="stable")[:, :10]

    monkeypatch.setattr(blend, "popularity_recommendations", perfect)
    run_blend(monkeypatch)

    grid = pd.read_csv(os.path.join(REPORTS, "validation_blend.csv"))
    assert grid.loc[grid["model"] == "popularity", "ndcg_at_10"].iloc[0] > grid.loc[
        grid["model"] == "pure_svd_blend", "ndcg_at_10"].max()
    assert not os.path.exists(os.path.join(REPORTS, "test_metrics_v2.json"))
    assert not os.path.exists(os.path.join(REPORTS, "tradeoff.csv"))
