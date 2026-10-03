"""Offline evaluation of the recommender with a temporal leave-one-out protocol.

For each sampled MovieLens user, the last movie they liked is hidden. The
movies they liked before it are the seeds; the recommender must put the
hidden movie in its top 10 without recommending anything the user had
already rated by then.

Compared: a popularity baseline, and the content-based recommender with its
reranker (default weights, plus a sweep of the popularity weight).

Run from the repo root:  python -m scripts.evaluate_offline
Writes reports/offline_eval.json and reports/tradeoff.csv.
"""
import argparse
import json
import math
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, NamedTuple, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.serving import storage
from src.serving.semantic_search import Reranker, SemanticSearchEngine

REPORT_DIR = "reports"
EVAL_PATH = os.path.join(REPORT_DIR, "offline_eval.json")
TRADEOFF_PATH = os.path.join(REPORT_DIR, "tradeoff.csv")
RATINGS_PATH = os.path.join("ml-25m", "ratings.csv")
RATINGS_R2_KEY = "raw/ratings.csv"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

SEED = 42
N_USERS = 2000
MIN_LIKED = 20
LIKE_THRESHOLD = 4.0
TOP_K = 10
N_CANDIDATES = 100
QUAL_WEIGHT = 0.1
DEFAULT_POP_WEIGHT = 0.3
POP_WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7]
HEAD_SHARE = 0.2  # the most popular 20% of movies are the "head"; the rest is the long tail
BOOTSTRAP_SAMPLES = 1000
CHUNK_ROWS = 2_000_000
# HitRate above this for a content-based recommender on MovieLens points to leakage.
SUSPICIOUS_HIT_RATE = 0.5

RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}


class Case(NamedTuple):
    user_id: int
    target: int  # the last liked movie, hidden from the recommender
    seeds: Dict[int, float]  # movies liked before the target -> rating
    exclude: List[int]  # every movie rated up to the target's timestamp (target not included)


# ---------------------------------------------------------------- data


def ensure_ratings(path: str = RATINGS_PATH) -> str:
    """Make sure ratings.csv of MovieLens 25M is on disk; download it if not."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if storage.has_r2_credentials():
        print(f"Downloading {RATINGS_R2_KEY} from R2...")
        storage.download_object(RATINGS_R2_KEY, path)
        return path

    print(f"Downloading {MOVIELENS_URL} ...")
    zip_path = path + ".zip"
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive, archive.open("ml-25m/ratings.csv") as src, open(path, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    os.remove(zip_path)
    return path


def read_rating_chunks(path: str):
    return pd.read_csv(path, usecols=list(RATING_DTYPES), dtype=RATING_DTYPES, chunksize=CHUNK_ROWS)


def count_liked_per_user(path: str, catalog_ids: np.ndarray) -> np.ndarray:
    """liked[u] = number of catalogue movies user u rated >= LIKE_THRESHOLD."""
    liked = np.zeros(0, dtype=np.int64)
    for chunk in read_rating_chunks(path):
        mask = (chunk["rating"].to_numpy() >= LIKE_THRESHOLD) & np.isin(chunk["movieId"].to_numpy(), catalog_ids)
        counts = np.bincount(chunk["userId"].to_numpy()[mask])
        if len(counts) > len(liked):
            liked = np.pad(liked, (0, len(counts) - len(liked)))
        liked[:len(counts)] += counts
    return liked


def sample_users(liked_counts: np.ndarray, n_users: int = N_USERS, seed: int = SEED) -> np.ndarray:
    eligible = np.flatnonzero(liked_counts >= MIN_LIKED)
    if len(eligible) <= n_users:
        return eligible
    return np.sort(np.random.default_rng(seed).choice(eligible, size=n_users, replace=False))


def load_user_ratings(path: str, users: np.ndarray, catalog_ids: np.ndarray) -> pd.DataFrame:
    """All ratings of the sampled users for movies that are in the catalogue."""
    parts = []
    for chunk in read_rating_chunks(path):
        keep = np.isin(chunk["userId"].to_numpy(), users) & np.isin(chunk["movieId"].to_numpy(), catalog_ids)
        parts.append(chunk[keep])
    return pd.concat(parts, ignore_index=True)


def build_cases(ratings: pd.DataFrame) -> List[Case]:
    """One leave-one-out case per user: hide the last liked movie."""
    cases = []
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"])
    for user_id, history in ordered.groupby("userId", sort=True):
        liked = history[history["rating"] >= LIKE_THRESHOLD]
        if len(liked) < 2:
            continue
        target_row = liked.iloc[-1]
        target = int(target_row["movieId"])
        seeds_frame = liked.iloc[:-1]
        seeds = dict(zip(seeds_frame["movieId"].astype(int), seeds_frame["rating"].astype(float)))
        seen = history.loc[history["timestamp"] <= target_row["timestamp"], "movieId"].astype(int)
        cases.append(Case(int(user_id), target, seeds, [m for m in seen if m != target]))
    return cases


# ---------------------------------------------------------------- recommenders


def reranker_weights(pop_weight: float) -> Dict[str, float]:
    """Quality weight is fixed; similarity takes what popularity leaves."""
    return {"sim_weight": round(1.0 - QUAL_WEIGHT - pop_weight, 6), "pop_weight": pop_weight, "qual_weight": QUAL_WEIGHT}


def recommend_content(pool: Sequence[Dict[str, Any]], pop_weight: float, top_k: int = TOP_K) -> List[int]:
    ranked = Reranker.rerank(list(pool), **reranker_weights(pop_weight))
    return [movie["movie_id"] for movie in ranked[:top_k]]


def recommend_popular(popular_ids: Sequence[int], exclude: Sequence[int], top_k: int = TOP_K) -> List[int]:
    """The most rated movies the user has not rated yet."""
    seen = set(exclude)
    picks = []
    for movie_id in popular_ids:
        if movie_id not in seen:
            picks.append(movie_id)
            if len(picks) == top_k:
                break
    return picks


# ---------------------------------------------------------------- metrics


def ndcg_at_k(recommended: Sequence[int], target: int) -> float:
    """NDCG with a single relevant item: 1 / log2(rank + 1), 0 if it is missing."""
    if target in recommended:
        return 1.0 / math.log2(list(recommended).index(target) + 2)
    return 0.0


def bootstrap_indices(n: int, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def mean_with_ci(values: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return {"value": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def ratio_with_ci(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    draws = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(numerator.sum() / denominator.sum()), "ci95_low": float(low), "ci95_high": float(high)}


def summarise(recommendations: List[List[int]], targets: List[int], head: set, catalog_size: int,
              indices: np.ndarray) -> Dict[str, Any]:
    hits = np.array([float(t in recs) for recs, t in zip(recommendations, targets)])
    ndcg = np.array([ndcg_at_k(recs, t) for recs, t in zip(recommendations, targets)])
    tail = np.array([sum(m not in head for m in recs) for recs in recommendations], dtype=float)
    sizes = np.array([len(recs) for recs in recommendations], dtype=float)
    distinct = {m for recs in recommendations for m in recs}
    return {
        "hit_rate_at_10": mean_with_ci(hits, indices),
        "ndcg_at_10": mean_with_ci(ndcg, indices),
        "long_tail_share": ratio_with_ci(tail, sizes, indices),
        # A distinct count shrinks under resampling, so a bootstrap interval
        # would sit below the estimate; only the point estimate is reported.
        "catalog_coverage": {"value": len(distinct) / catalog_size, "distinct_movies": len(distinct)},
        "_hits": hits,
    }


# ---------------------------------------------------------------- evaluation


def evaluate(engine: SemanticSearchEngine, cases: List[Case], n_candidates: int = N_CANDIDATES,
             pop_weights: Sequence[float] = tuple(POP_WEIGHTS)) -> Dict[str, Any]:
    catalog = engine.catalog
    by_popularity = catalog.sort_values(["rating_count", "movieId"], ascending=[False, True])
    popular_ids = [int(m) for m in by_popularity["movieId"]]
    head = set(popular_ids[:math.ceil(HEAD_SHARE * len(popular_ids))])

    served_candidates = TOP_K * 2  # what personalized_recommend uses when n_candidates is not given
    recs: Dict[str, List[List[int]]] = {"popularity": [], "content_as_served": []}
    recs.update({f"pop_{w}": [] for w in pop_weights})
    targets, short_pools = [], 0

    for i, case in enumerate(cases, 1):
        user_vec = engine.get_user_vector(case.seeds)
        pool = engine.retrieve_candidates(user_vec, n_candidates, exclude_ids=case.exclude)
        short_pools += len(pool) < n_candidates

        targets.append(case.target)
        recs["popularity"].append(recommend_popular(popular_ids, case.exclude))
        recs["content_as_served"].append(recommend_content(pool[:served_candidates], DEFAULT_POP_WEIGHT))
        for w in pop_weights:
            recs[f"pop_{w}"].append(recommend_content(pool, w))
        if i % 200 == 0:
            print(f"  {i:,} / {len(cases):,} users")

    indices = bootstrap_indices(len(cases), BOOTSTRAP_SAMPLES)
    summary = {name: summarise(lists, targets, head, len(catalog), indices) for name, lists in recs.items()}

    # Paired difference in HitRate between the default recommender and popularity.
    default = summary[f"pop_{DEFAULT_POP_WEIGHT}"]
    difference = mean_with_ci(default["_hits"] - summary["popularity"]["_hits"], indices)
    for result in summary.values():
        del result["_hits"]

    tradeoff = []
    for w in pop_weights:
        result = summary[f"pop_{w}"]
        tradeoff.append({
            **reranker_weights(w),
            "hit_rate_at_10": result["hit_rate_at_10"]["value"],
            "hit_rate_ci95_low": result["hit_rate_at_10"]["ci95_low"],
            "hit_rate_ci95_high": result["hit_rate_at_10"]["ci95_high"],
            "ndcg_at_10": result["ndcg_at_10"]["value"],
            "ndcg_ci95_low": result["ndcg_at_10"]["ci95_low"],
            "ndcg_ci95_high": result["ndcg_at_10"]["ci95_high"],
            "catalog_coverage": result["catalog_coverage"]["value"],
            "long_tail_share": result["long_tail_share"]["value"],
            "long_tail_ci95_low": result["long_tail_share"]["ci95_low"],
            "long_tail_ci95_high": result["long_tail_share"]["ci95_high"],
        })

    return {
        "results": {
            "popularity": summary["popularity"],
            "content_default": {"candidates": n_candidates, **reranker_weights(DEFAULT_POP_WEIGHT), **default},
            "content_as_served": {"candidates": served_candidates, **reranker_weights(DEFAULT_POP_WEIGHT),
                                  **summary["content_as_served"]},
        },
        "hit_rate_difference_default_minus_popularity": difference,
        "tradeoff": tradeoff,
        "users_with_fewer_candidates_than_requested": int(short_pools),
        "head_movies": len(head),
    }


def print_result(name: str, result: Dict[str, Any]) -> None:
    hit, ndcg, tail = result["hit_rate_at_10"], result["ndcg_at_10"], result["long_tail_share"]
    print(
        f"  {name:<34} HitRate@10 {hit['value']:.2%} [{hit['ci95_low']:.2%}, {hit['ci95_high']:.2%}] | "
        f"NDCG@10 {ndcg['value']:.4f} [{ndcg['ci95_low']:.4f}, {ndcg['ci95_high']:.4f}] | "
        f"coverage {result['catalog_coverage']['value']:.2%} | long tail {tail['value']:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ratings", default=RATINGS_PATH, help="path to MovieLens 25M ratings.csv")
    parser.add_argument("--db", default=storage.DB_PATH, help="path to the LanceDB directory")
    args = parser.parse_args()
    load_dotenv()

    if not os.path.exists(args.db):
        if not storage.has_r2_credentials():
            raise SystemExit(f"{args.db} not found and no R2 credentials are set: nothing to evaluate.")
        print("Downloading the vector database from R2...")
        storage.download_database(args.db)
    engine = SemanticSearchEngine(lancedb_uri=args.db)
    engine.load_table()
    catalog_ids = np.sort(engine.catalog["movieId"].to_numpy())

    ratings_path = ensure_ratings(args.ratings)
    print("Pass 1: counting liked movies per user...")
    liked_counts = count_liked_per_user(ratings_path, catalog_ids)
    users = sample_users(liked_counts, N_USERS)
    print(f"  {int((liked_counts >= MIN_LIKED).sum()):,} eligible users, {len(users):,} sampled (seed {SEED})")
    print("Pass 2: loading the ratings of the sampled users...")
    cases = build_cases(load_user_ratings(ratings_path, users, catalog_ids))

    print(f"Evaluating {len(cases):,} users...")
    report = evaluate(engine, cases, N_CANDIDATES)
    report = {
        "protocol": {
            "split": "temporal leave-one-out: the last liked movie of each user is the target",
            "liked_means_rating_at_least": LIKE_THRESHOLD,
            "min_liked_movies_per_user": MIN_LIKED,
            "seed": SEED,
            "top_k": TOP_K,
            "candidates": N_CANDIDATES,
            "excluded": "every movie the user rated up to the target's timestamp",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "long_tail": f"movies outside the {HEAD_SHARE:.0%} most rated",
            "popularity_counts": "rating_count stored in the database, computed on all ratings",
        },
        "data": {
            "catalog_movies": int(len(catalog_ids)),
            "eligible_users": int((liked_counts >= MIN_LIKED).sum()),
            "evaluated_users": len(cases),
            "median_seed_movies": float(np.median([len(c.seeds) for c in cases])),
            "median_excluded_movies": float(np.median([len(c.exclude) for c in cases])),
        },
        **report,
    }

    print("\nResults")
    for name, result in report["results"].items():
        print_result(name, result)
    diff = report["hit_rate_difference_default_minus_popularity"]
    print(f"  HitRate difference (content default - popularity): {diff['value']:+.2%} "
          f"[{diff['ci95_low']:+.2%}, {diff['ci95_high']:+.2%}]")
    print("\nTrade-off (quality weight fixed at 0.1)")
    tradeoff = pd.DataFrame(report["tradeoff"])
    print(tradeoff[["pop_weight", "sim_weight", "hit_rate_at_10", "ndcg_at_10", "catalog_coverage",
                    "long_tail_share"]].round(4).to_string(index=False))
    print(f"\nUsers with fewer than {N_CANDIDATES} candidates after exclusion: "
          f"{report['users_with_fewer_candidates_than_requested']}")

    hit_rate = report["results"]["content_default"]["hit_rate_at_10"]["value"]
    if hit_rate > SUSPICIOUS_HIT_RATE:
        raise SystemExit(
            f"HitRate@10 = {hit_rate:.1%} is implausibly high for this task: suspect leakage. "
            "Nothing was written."
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    tradeoff.to_csv(TRADEOFF_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {EVAL_PATH} and {TRADEOFF_PATH}")


if __name__ == "__main__":
    main()
