"""PureSVD blended with recent popularity: choose the blend on validation, then score it on test.

    score(u, i) = z_u(svd(u, i)) + weight * z(log1p(pop90(i)))

Grid: train_window in {all, 3y, 1y} x weight in {0, 0.25, 0.5, 1, 2, 4}, with k fixed.
The configuration with the highest NDCG@10 on the validation window (the last
31 days of train) is selected. Only if it beats popularity there is it scored
on the test window, next to plain PureSVD and popularity.

Run from the repo root, after src.train and src.evaluate:
  python -m src.blend --cutoff 2019-06-01

Writes, in reports/<cutoff>/: validation_blend.csv, test_metrics_v2.json and
tradeoff.csv; and the selected model in artifacts/<cutoff>/blend/.
"""
import argparse
import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from src import baseline, evaluate, metrics
from src.config import load_data_config
from src.data import build_interactions, data_config_of, ensure_processed, load_stats, read_parquet
from src.evaluate import TEST_METRICS_FILE, TOP_N, build_targets, head_mask, report_dir
from src.model import PureSVD, artifact_dir, popularity_boost
from src.prepare import git_state, iso
from src.train import VALIDATION_FILE, load_train_config, validation_split

BLEND_CONFIG_PATH = os.path.join("configs", "blend.yaml")
VALIDATION_BLEND_FILE = "validation_blend.csv"
TEST_METRICS_V2_FILE = "test_metrics_v2.json"
TRADEOFF_FILE = "tradeoff.csv"
BLEND_ARTIFACT = "blend"
POPULARITY = "popularity"
BLEND = "pure_svd_blend"
METRIC_COLUMNS = ["hit_rate_at_10", "recall_at_10", "ndcg_at_10", "ndcg_ci95_low", "ndcg_ci95_high",
                  "catalog_coverage", "long_tail_share"]


@dataclass(frozen=True)
class BlendConfig:
    k: int = 64
    train_windows: Tuple[str, ...] = ("all", "3y", "1y")
    blend_weights: Tuple[float, ...] = (0, 0.25, 0.5, 1, 2, 4)


def load_blend_config(path: str = BLEND_CONFIG_PATH) -> BlendConfig:
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return BlendConfig(k=values["k"], train_windows=tuple(values["train_windows"]),
                       blend_weights=tuple(float(w) for w in values["blend_weights"]))


def window_start(cutoff: str, window: str) -> Optional[int]:
    """Start (Unix seconds, UTC) of a training window ending at `cutoff`: 'all', or '<n>y' for n years."""
    if window == "all":
        return None
    if not window.endswith("y") or not window[:-1].isdigit():
        raise ValueError(f"Unknown train_window {window!r}: use 'all' or '<n>y'")
    day = datetime.date.fromisoformat(cutoff)
    try:
        start = day.replace(year=day.year - int(window[:-1]))
    except ValueError:  # 29 February
        start = day.replace(year=day.year - int(window[:-1]), day=28)
    return int(datetime.datetime(start.year, start.month, start.day, tzinfo=datetime.timezone.utc).timestamp())


@dataclass
class Fold:
    """One (training data, evaluation window) pair with everything the grid shares."""
    cutoff: str
    cutoff_timestamp: int
    threshold: float
    train: pd.DataFrame
    item_ids: np.ndarray
    rows: np.ndarray  # rows of the evaluated users in the training matrices
    targets: np.ndarray
    seen_rows: Any
    head: np.ndarray
    indices: np.ndarray  # bootstrap resamples, shared by every configuration (paired)


def make_fold(train: pd.DataFrame, eval_frame: pd.DataFrame, cutoff: str, cutoff_timestamp: int,
              threshold: float) -> Fold:
    interactions = build_interactions(train, threshold)
    users, rows, targets = build_targets(interactions, eval_frame)
    return Fold(cutoff=cutoff, cutoff_timestamp=cutoff_timestamp, threshold=threshold, train=train,
                item_ids=interactions.item_ids, rows=rows, targets=targets,
                seen_rows=interactions.seen[rows], head=head_mask(interactions),
                indices=metrics.bootstrap_indices(len(users)))


def blend_recommendations(fold: Fold, k: int, weights_by_window: Dict[str, Sequence[float]],
                          random_state: int) -> Tuple[Dict[Tuple[str, float], np.ndarray], Dict[str, Any]]:
    """Top-10 lists for every (train_window, weight). One SVD is fitted per window."""
    boost = popularity_boost(baseline.liked_counts(fold.train, fold.item_ids, fold.cutoff_timestamp, fold.threshold))
    recommendations, info = {}, {}
    for window, weights in weights_by_window.items():
        interactions = build_interactions(fold.train, fold.threshold, liked_since=window_start(fold.cutoff, window))
        liked_rows = interactions.liked[fold.rows]
        started = time.time()
        model = PureSVD.fit(interactions.liked, k, random_state, item_ids=interactions.item_ids,
                            user_ids=interactions.user_ids)
        info[window] = {
            "liked_interactions": int(interactions.liked.nnz),
            "evaluated_users_without_liked_in_window": int((np.diff(liked_rows.indptr) == 0).sum()),
            "fit_seconds": round(time.time() - started, 1),
            "model": model,
            "boost": boost,
        }
        for weight in weights:
            blended = model.with_blend(boost, weight) if weight else model
            recommendations[(window, float(weight))] = blended.recommend(liked_rows, fold.seen_rows, TOP_N)
    return recommendations, info


def popularity_recommendations(fold: Fold) -> np.ndarray:
    ranking = baseline.popularity_ranking(fold.train, fold.item_ids, fold.cutoff_timestamp, fold.threshold)
    return baseline.recommend(ranking, fold.seen_rows, TOP_N)


def flat_row(result: Dict[str, Any]) -> Dict[str, float]:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"]["value"],
        "recall_at_10": result["recall_at_10"]["value"],
        "ndcg_at_10": result["ndcg_at_10"]["value"],
        "ndcg_ci95_low": result["ndcg_at_10"]["ci95_low"],
        "ndcg_ci95_high": result["ndcg_at_10"]["ci95_high"],
        "catalog_coverage": result["catalog_coverage"]["value"],
        "long_tail_share": result["long_tail_share"]["value"],
    }


def grid_table(fold: Fold, recommendations: Dict[Tuple[str, float], np.ndarray]) -> pd.DataFrame:
    rows = []
    for (window, weight), recs in recommendations.items():
        rows.append({"model": BLEND, "train_window": window, "blend_weight": weight,
                     **flat_row(metrics.evaluate(recs, fold.targets, fold.head, fold.indices))})
    return pd.DataFrame(rows)


def select(table: pd.DataFrame, windows: Sequence[str]) -> Tuple[str, float]:
    """The configuration with the highest NDCG@10; ties go to the smaller weight, then the earlier window."""
    order = {window: position for position, window in enumerate(windows)}
    best = max(table.to_dict("records"),
               key=lambda row: (row["ndcg_at_10"], -row["blend_weight"], -order[row["train_window"]]))
    return best["train_window"], float(best["blend_weight"])


def check_against_k_sweep(table: pd.DataFrame, k: int, cutoff: str) -> Optional[bool]:
    """weight 0 on all of train must reproduce the PureSVD row of validation.csv for the same k."""
    path = os.path.join(report_dir(cutoff), VALIDATION_FILE)
    if not os.path.exists(path):
        return None
    sweep = pd.read_csv(path)
    reference = sweep.loc[sweep["k"] == k, "ndcg_at_10"]
    plain = table.loc[(table["train_window"] == "all") & (table["blend_weight"] == 0), "ndcg_at_10"]
    if reference.empty or plain.empty:
        return None
    return bool(abs(float(reference.iloc[0]) - float(plain.iloc[0])) < 1e-9)


def main():
    parser = argparse.ArgumentParser(description="Choose the PureSVD + recent popularity blend on validation.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    parser.add_argument("--config", default=BLEND_CONFIG_PATH)
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff
    blend_config, train_config = load_blend_config(args.config), load_train_config()
    grid = {window: blend_config.blend_weights for window in blend_config.train_windows}

    directory = ensure_processed(cutoff)
    stats = load_stats(directory)
    config = data_config_of(stats)
    movies = read_parquet(directory, "movies.parquet")
    os.makedirs(report_dir(cutoff), exist_ok=True)

    # ---- 1. Choose the configuration on the validation window (inside train).
    train_val, validation, val_config = validation_split(
        read_parquet(directory, "train.parquet"), movies, config, train_config.validation_days)
    fold = make_fold(train_val, validation, val_config.cutoff, val_config.cutoff_timestamp, config.positive_threshold)
    print(f"Validation: {len(validation):,} liked ratings of {len(fold.rows):,} users, "
          f"{val_config.cutoff} up to {cutoff}; k = {blend_config.k}")
    recommendations, _ = blend_recommendations(fold, blend_config.k, grid, train_config.random_state)
    table = grid_table(fold, recommendations)
    popularity_recs = popularity_recommendations(fold)
    popularity_row = flat_row(metrics.evaluate(popularity_recs, fold.targets, fold.head, fold.indices))

    matches = check_against_k_sweep(table, blend_config.k, cutoff)
    if matches is False:
        raise SystemExit("weight 0 on all of train does not reproduce validation.csv: there is a bug.")

    window, weight = select(table, blend_config.train_windows)
    selected_recs = recommendations[(window, weight)]
    beats_popularity = bool(table["ndcg_at_10"].max() > popularity_row["ndcg_at_10"])
    table["selected"] = (table["train_window"] == window) & (table["blend_weight"] == weight)
    table = pd.concat([table, pd.DataFrame([{"model": POPULARITY, "train_window": "", "blend_weight": np.nan,
                                             **popularity_row, "selected": False}])], ignore_index=True)
    table.insert(3, "validation_users", len(fold.rows))
    table.to_csv(os.path.join(report_dir(cutoff), VALIDATION_BLEND_FILE), index=False, encoding="utf-8")
    print(table[["model", "train_window", "blend_weight", "hit_rate_at_10", "recall_at_10", "ndcg_at_10",
                 "catalog_coverage", "long_tail_share", "selected"]].round(4).to_string(index=False))
    validation_difference = metrics.paired_difference(selected_recs, popularity_recs, fold.targets, fold.head,
                                                      fold.indices)
    print(f"Selected on validation: train_window = {window}, weight = {weight} "
          f"(NDCG@10 difference to popularity {validation_difference['ndcg_at_10']['value']:+.4f} "
          f"[{validation_difference['ndcg_at_10']['ci95_low']:+.4f}, "
          f"{validation_difference['ndcg_at_10']['ci95_high']:+.4f}])")
    if not beats_popularity:
        print("No configuration beats popularity on validation. Stopping before the test set.")
        return
    del fold, train_val, validation, recommendations
    pa.default_memory_pool().release_unused()

    # ---- 2. Score the selected configuration on the test window, next to plain PureSVD and popularity.
    test = read_parquet(directory, "test.parquet")
    fold = make_fold(read_parquet(directory, "train.parquet"), test, cutoff, config.cutoff_timestamp,
                     config.positive_threshold)
    needed = {window: blend_config.blend_weights}  # every weight of the selected window, for the trade-off curve
    if window != "all":
        needed["all"] = (0.0,)
    recommendations, info = blend_recommendations(fold, blend_config.k, needed, train_config.random_state)
    named = {
        "pure_svd": recommendations[("all", 0.0)],
        BLEND: recommendations[(window, weight)],
        POPULARITY: popularity_recommendations(fold),
    }
    results = {name: metrics.evaluate(recs, fold.targets, fold.head, fold.indices) for name, recs in named.items()}
    for name, result in results.items():
        if (result["hit_rate_at_10"]["value"] > evaluate.MAX_PLAUSIBLE_HIT_RATE
                or result["ndcg_at_10"]["value"] > evaluate.MAX_PLAUSIBLE_NDCG):
            raise SystemExit(f"{name}: the result is implausibly high for this task: suspect leakage. Nothing was written.")

    differences = {
        f"{BLEND}_minus_popularity": metrics.paired_difference(named[BLEND], named[POPULARITY], fold.targets,
                                                               fold.head, fold.indices),
        f"{BLEND}_minus_pure_svd": metrics.paired_difference(named[BLEND], named["pure_svd"], fold.targets,
                                                             fold.head, fold.indices),
    }
    for name, other in ((f"{BLEND}_minus_popularity", POPULARITY), (f"{BLEND}_minus_pure_svd", "pure_svd")):
        differences[name]["catalog_coverage"] = {
            "value": results[BLEND]["catalog_coverage"]["value"] - results[other]["catalog_coverage"]["value"]}

    # The plain PureSVD row must be the one already reported by src.evaluate.
    first_report_path = os.path.join(report_dir(cutoff), TEST_METRICS_FILE)
    matches_first_report = None
    if os.path.exists(first_report_path):
        with open(first_report_path, encoding="utf-8") as f:
            first = json.load(f)
        if first["k"] == blend_config.k:
            matches_first_report = bool(abs(first["models"]["pure_svd"]["ndcg_at_10"]["value"]
                                            - results["pure_svd"]["ndcg_at_10"]["value"]) < 1e-9)
            if not matches_first_report:
                raise SystemExit("The plain PureSVD row does not reproduce test_metrics.json: there is a bug.")

    print(f"\nTest: {len(test):,} liked ratings of {len(fold.rows):,} users (second look at this test window)")
    for name, result in results.items():
        print(f"  {name:<15} " + " | ".join(
            f"{label} {result[key]['value']:.4f} [{result[key]['ci95_low']:.4f}, {result[key]['ci95_high']:.4f}]"
            for label, key in [("HitRate@10", "hit_rate_at_10"), ("Recall@10", "recall_at_10"),
                               ("NDCG@10", "ndcg_at_10"), ("long tail", "long_tail_share")])
              + f" | coverage {result['catalog_coverage']['value']:.4f}")
    for name, difference in differences.items():
        print(f"  {name} (paired): " + " | ".join(
            f"{key} {value['value']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"
            for key, value in difference.items() if "ci95_low" in value))

    tradeoff = pd.DataFrame([
        {"train_window": window, "blend_weight": float(w), "selected_on_validation": float(w) == weight,
         **flat_row(metrics.evaluate(recommendations[(window, float(w))], fold.targets, fold.head, fold.indices))}
        for w in blend_config.blend_weights])
    tradeoff.to_csv(os.path.join(report_dir(cutoff), TRADEOFF_FILE), index=False, encoding="utf-8")
    print("\nTrade-off on test (descriptive only; the weight was chosen on validation):")
    print(tradeoff[["blend_weight", "hit_rate_at_10", "recall_at_10", "ndcg_at_10", "catalog_coverage",
                    "long_tail_share", "selected_on_validation"]].round(4).to_string(index=False))

    report = {
        "cutoff": cutoff,
        "note": ("Second look at this test window: the configuration was chosen on the validation window "
                 f"({val_config.cutoff} up to {cutoff}), after the first test evaluation of plain PureSVD. "
                 "An independent evaluation needs later months."),
        "test_looks": 2,
        "selected": {"k": blend_config.k, "train_window": window, "blend_weight": weight,
                     "selected_by": "ndcg_at_10 on validation",
                     "validation_ndcg_at_10": float(table.loc[table["selected"], "ndcg_at_10"].iloc[0]),
                     "validation_popularity_ndcg_at_10": popularity_row["ndcg_at_10"],
                     "validation_difference_to_popularity": validation_difference},
        "grid": {"train_windows": list(blend_config.train_windows), "blend_weights": list(blend_config.blend_weights)},
        "top_n": TOP_N,
        "evaluated_users": int(len(fold.rows)),
        "test_rows": int(len(test)),
        "evaluated_users_without_liked_in_window": info[window]["evaluated_users_without_liked_in_window"],
        "liked_interactions_in_window": info[window]["liked_interactions"],
        "bootstrap": {"samples": metrics.BOOTSTRAP_SAMPLES, "seed": metrics.SEED, "resampled_unit": "user"},
        "models": results,
        "differences": differences,
        "pure_svd_row_matches_test_metrics_json": matches_first_report,
        "weight_0_all_matches_validation_csv": matches,
        "train_sha256": stats["sha256"]["train.parquet"],
        **git_state(),
    }
    with open(os.path.join(report_dir(cutoff), TEST_METRICS_V2_FILE), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    selected_model = info[window]["model"].with_blend(info[window]["boost"], weight) if weight else info[window]["model"]
    selected_model.save(os.path.join(artifact_dir(cutoff), BLEND_ARTIFACT), {
        "cutoff": cutoff, "train_window": window, "random_state": train_config.random_state,
        "positive_threshold": config.positive_threshold, "train_sha256": stats["sha256"]["train.parquet"],
        "train_seconds": info[window]["fit_seconds"], "trained_at_utc": iso(time.time()), **git_state(),
    })
    print(f"\nSaved {VALIDATION_BLEND_FILE}, {TEST_METRICS_V2_FILE}, {TRADEOFF_FILE} in {report_dir(cutoff)} "
          f"and the model in {os.path.join(artifact_dir(cutoff), BLEND_ARTIFACT)}")


if __name__ == "__main__":
    main()
