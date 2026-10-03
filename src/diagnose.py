"""Diagnostics: is PureSVD behind popularity because it ignores how recent a movie is?

Measures only; no model is changed. Uses the model saved by src.train and
the popularity baseline on the test window.

Run from the repo root:
  python -m src.diagnose --cutoff 2019-06-01

Writes reports/<cutoff>/diagnostics.json.
"""
import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src import baseline, metrics
from src.config import load_data_config
from src.evaluate import TOP_N, TestSet, load_test_set, load_trained_model, report_dir
from src.model import NO_RECOMMENDATION
from src.prepare import git_state

DIAGNOSTICS_FILE = "diagnostics.json"
NEW_FROM_YEAR = 2018
YEAR_PATTERN = r"\((\d{4})\)"


def release_years(movies: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    """Release year per movie column, taken from the title; NaN when the title has none."""
    years = movies["title"].str.extract(YEAR_PATTERN, expand=False).astype("float64")
    by_id = dict(zip(movies["movieId"].tolist(), years.tolist()))
    return np.array([by_id.get(int(movie_id), np.nan) for movie_id in item_ids], dtype=np.float64)


def year_profile(columns: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
    """Share of new movies and median year for a multiset of movie columns."""
    picked = years[columns]
    known = picked[~np.isnan(picked)]
    return {
        "movies": int(len(picked)),
        "without_year": int(len(picked) - len(known)),
        "share_released_from_2018": float((known >= NEW_FROM_YEAR).mean()) if len(known) else None,
        "median_year": float(np.median(known)) if len(known) else None,
    }


def metrics_on_subset(recommended: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """HitRate@10 and Recall@10 counting only `targets`, over the users who have at least one of them."""
    users = np.flatnonzero(targets.any(axis=1))
    if len(users) == 0:
        return {"users": 0, "targets": 0}
    per_user = metrics.per_user_metrics(recommended[users], targets[users])
    indices = metrics.bootstrap_indices(len(users))
    return {
        "users": int(len(users)),
        "targets": int(targets.sum()),
        "hit_rate_at_10": metrics.mean_with_ci(per_user["hit_rate"], indices),
        "recall_at_10": metrics.mean_with_ci(per_user["recall"], indices),
    }


def long_tail_share(recommended: np.ndarray, head: np.ndarray, keep: np.ndarray) -> Dict[str, Any]:
    """Share of recommendations outside the head, among the recommended movies where `keep` is True."""
    columns = recommended[recommended != NO_RECOMMENDATION]
    columns = columns[keep[columns]]
    return {"recommendations": int(len(columns)),
            "long_tail_share": float((~head[columns]).mean()) if len(columns) else None}


def diagnose(data: TestSet, recommendations: Dict[str, np.ndarray]) -> Dict[str, Any]:
    years = release_years(data.movies, data.interactions.item_ids)
    is_new = years >= NEW_FROM_YEAR  # False for movies without a year
    not_new = ~is_new
    target_users, target_columns = np.nonzero(data.targets)

    # Liked ratings in train, to compare with what users liked in the test window.
    threshold, cutoff = data.config.positive_threshold, data.config.cutoff_timestamp
    liked = data.train[data.train["rating"] >= threshold]
    liked_columns = data.interactions.cols_of(liked["movieId"].to_numpy())
    recent = (liked["timestamp"].to_numpy() >= baseline.window_start(cutoff))

    report = {
        "new_means_release_year_from": NEW_FROM_YEAR,
        "catalog": {**year_profile(np.arange(len(years)), years), "movies_in_head": int(data.head.sum()),
                    "new_movies_in_head": int((is_new & data.head).sum()), "new_movies": int(is_new.sum())},
        "release_year": {
            "train_liked_ratings_all_time": year_profile(liked_columns, years),
            "train_liked_ratings_last_90_days": year_profile(liked_columns[recent], years),
            "test_targets": year_profile(target_columns, years),
        },
        "by_target_age": {},
        "long_tail_share": {},
    }
    for name, recs in recommendations.items():
        report["release_year"][f"{name}_top_10"] = year_profile(recs[recs != NO_RECOMMENDATION], years)
    for label, mask in (("new", is_new), ("old", not_new)):
        subset = data.targets & mask
        report["by_target_age"][label] = {name: metrics_on_subset(recs, subset) for name, recs in recommendations.items()}
    for name, recs in recommendations.items():
        report["long_tail_share"][name] = {
            "all_recommendations": long_tail_share(recs, data.head, np.ones(len(years), dtype=bool)),
            "excluding_movies_from_2018": long_tail_share(recs, data.head, not_new),
            "only_movies_from_2018": long_tail_share(recs, data.head, is_new),
        }
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("Release year (share released from 2018 | median year):")
    for name, profile in report["release_year"].items():
        print(f"  {name:<34} {profile['share_released_from_2018']:.1%} | {profile['median_year']:.0f} "
              f"({profile['movies']:,} movies)")
    print("Metrics by age of the target movie:")
    for label, models in report["by_target_age"].items():
        for name, result in models.items():
            if result["users"]:
                hit, recall = result["hit_rate_at_10"], result["recall_at_10"]
                print(f"  {label} targets, {name:<11} HitRate@10 {hit['value']:.4f} [{hit['ci95_low']:.4f}, "
                      f"{hit['ci95_high']:.4f}] | Recall@10 {recall['value']:.4f} [{recall['ci95_low']:.4f}, "
                      f"{recall['ci95_high']:.4f}] ({result['users']:,} users, {result['targets']:,} targets)")
    print("Long-tail share (all | excluding movies from 2018 | only movies from 2018):")
    for name, shares in report["long_tail_share"].items():
        print(f"  {name:<11} " + " | ".join(
            "n/a" if shares[key]["long_tail_share"] is None else
            f"{shares[key]['long_tail_share']:.1%} of {shares[key]['recommendations']:,}"
            for key in ("all_recommendations", "excluding_movies_from_2018", "only_movies_from_2018")))


def main():
    parser = argparse.ArgumentParser(description="Diagnose PureSVD against popularity by release year.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff

    data = load_test_set(cutoff)
    model, meta = load_trained_model(cutoff, data)
    recommendations = {
        "pure_svd": model.recommend(data.liked_rows, data.seen_rows, TOP_N),
        "popularity": data.popularity_recommendations(),
    }
    report = {
        "cutoff": cutoff,
        "k": model.k,
        "evaluated_users": int(len(data.users)),
        **diagnose(data, recommendations),
        "model_item_factors_sha256": meta["item_factors_sha256"],
        **git_state(),
    }
    print_report(report)

    os.makedirs(report_dir(cutoff), exist_ok=True)
    path = os.path.join(report_dir(cutoff), DIAGNOSTICS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
