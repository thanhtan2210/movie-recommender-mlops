"""Prepare MovieLens ratings for training: a time-based split with train-only filters.

  train = every rating before the cutoff, for movies with enough ratings in train
  test  = liked movies in the window right after the cutoff, for users and
          movies the model can know from train

Run from the repo root:
  python -m src.prepare --cutoff 2019-06-01 [--source local:<dir>] [--upload]

Writes data/processed/<cutoff>/{train,test,movies}.parquet and stats.json.
Running it again with the same cutoff produces byte-identical parquet files.
"""
import argparse
import datetime
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from src.config import DATA_CONFIG_PATH, DataConfig, load_data_config
from src.storage import STATS_FILE, RemoteConflictError, Storage, sha256_file, upload_directory

RAW_RATINGS_KEY = "raw/ratings.csv"
RAW_MOVIES_KEY = "raw/movies.csv"
PROCESSED_PREFIX = "processed"
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
DATA_FILES = ["train.parquet", "test.parquet", "movies.parquet"]
SMALL_TEST_ROWS = 1000

RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}
MOVIE_DTYPES = {"movieId": "int32", "title": "string", "genres": "string"}


@dataclass
class Prepared:
    train: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    report: Dict[str, Any]


# ---------------------------------------------------------------- reading


def read_ratings(path: str) -> pd.DataFrame:
    """Read the four rating columns with small dtypes.

    pyarrow's reader is used because pandas.read_csv needs several GB at its
    peak for the 25M-row file, against well under 1 GB of final data.
    """
    options = pa_csv.ConvertOptions(
        include_columns=list(RATING_DTYPES),
        column_types={name: pa.type_for_alias(dtype) for name, dtype in RATING_DTYPES.items()},
    )
    table = pa_csv.read_csv(path, convert_options=options)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def read_movies(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(MOVIE_DTYPES), dtype=MOVIE_DTYPES, encoding="utf-8")


def resolve_source(source: str, storage: Optional[Storage] = None) -> Tuple[str, str, str]:
    """Paths of ratings.csv and movies.csv, downloading them from R2 if needed."""
    if source.startswith("local:"):
        directory = source[len("local:"):]
        return os.path.join(directory, "ratings.csv"), os.path.join(directory, "movies.csv"), source
    if source != "r2":
        raise ValueError(f"Unknown source {source!r}: use 'r2' or 'local:<directory>'")

    storage = storage or Storage.from_env()
    paths = []
    for key in (RAW_RATINGS_KEY, RAW_MOVIES_KEY):
        path = os.path.join(RAW_DIR, os.path.basename(key))
        downloaded = storage.download_if_changed(key, path)
        print(f"  {key}: {'downloaded' if downloaded else 'already on disk'}")
        paths.append(path)
    return paths[0], paths[1], f"r2://{storage.bucket}/raw/"


# ---------------------------------------------------------------- split and filters


def iso(timestamp) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.datetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def time_range(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    if frame.empty:
        return {"first": None, "last": None}
    return {"first": iso(frame["timestamp"].min()), "last": iso(frame["timestamp"].max())}


def ratings_per_month(timestamps: np.ndarray, year: int) -> Dict[str, int]:
    """Number of ratings in each month of `year` (UTC)."""
    edges = [datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc).timestamp() for month in range(1, 13)]
    edges.append(datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    counts, _ = np.histogram(timestamps, bins=edges)
    return {f"{year}-{month:02d}": int(count) for month, count in zip(range(1, 13), counts)}


def sorted_frame(columns: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Rows ordered by (userId, timestamp, movieId), so the output does not depend on input order.

    `columns` is emptied: each unsorted column is dropped as soon as its sorted copy exists.
    """
    order = np.lexsort((columns["movieId"], columns["timestamp"], columns["userId"]))
    ordered = {}
    for name in list(RATING_DTYPES):
        ordered[name] = columns.pop(name)[order]
    return pd.DataFrame(ordered, copy=False)


def count_distinct(ids: np.ndarray) -> int:
    return int(np.count_nonzero(np.bincount(ids))) if len(ids) else 0


def summarise(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "users": count_distinct(frame["userId"].to_numpy()),
        "movies": count_distinct(frame["movieId"].to_numpy()),
        "time_range_utc": time_range(frame),
    }


def prepare(ratings: pd.DataFrame, movies: pd.DataFrame, config: DataConfig) -> Prepared:
    """Split by time, then filter using the training set only.

    Works on the column arrays and releases the raw data before sorting: the
    25M-row file would otherwise need several GB at the peak.
    """
    cutoff, test_end = config.cutoff_timestamp, config.test_end_timestamp
    users = ratings["userId"].to_numpy()
    items = ratings["movieId"].to_numpy()
    values = ratings["rating"].to_numpy()
    times = ratings["timestamp"].to_numpy()
    in_train = times < cutoff
    in_window = (times >= cutoff) & (times < test_end)

    year = datetime.date.fromisoformat(config.cutoff).year
    raw_report = {
        "rows": int(len(ratings)),
        "time_range_utc": time_range(ratings),
        "rows_before_cutoff": int(in_train.sum()),
        "rows_in_test_window": int(in_window.sum()),
        "rows_after_test_window": int((times >= test_end).sum()),
    }
    monthly = ratings_per_month(times, year)

    # Catalogue: movies with enough ratings before the cutoff. Ratings in the
    # test window are never counted here.
    train_counts = np.bincount(items[in_train], minlength=int(items.max()) + 1 if len(items) else 1)
    in_catalog = train_counts >= config.min_item_ratings
    user_space = int(users.max()) + 1 if len(users) else 1

    window = ratings[in_window]
    keep = in_train & in_catalog[items]
    train_columns = {"userId": users[keep], "movieId": items[keep], "rating": values[keep], "timestamp": times[keep]}
    # Release the raw data before sorting.
    del ratings, users, items, values, times, in_train, in_window, keep
    pa.default_memory_pool().release_unused()
    train = sorted_frame(train_columns)

    # Users who can be evaluated: enough liked movies in the (filtered) training set.
    liked = train["rating"].to_numpy() >= config.positive_threshold
    eligible_user = np.bincount(train["userId"].to_numpy()[liked], minlength=user_space) >= config.min_user_train_positives

    step_positive = window[window["rating"] >= config.positive_threshold]
    step_user = step_positive[eligible_user[step_positive["userId"].to_numpy()]]
    test = step_user[in_catalog[step_user["movieId"].to_numpy()]]
    test = sorted_frame({column: test[column].to_numpy() for column in RATING_DTYPES})

    catalog = movies[in_catalog_lookup(movies["movieId"].to_numpy(), in_catalog)]
    catalog = catalog.sort_values("movieId", kind="stable").reset_index(drop=True)

    report = {
        "raw": raw_report,
        "catalog": {
            "movies_with_any_train_rating": int((train_counts > 0).sum()),
            "movies_kept": int(in_catalog.sum()),
            "movies_kept_without_metadata": int(in_catalog.sum() - len(catalog)),
            "train_rows_removed_by_movie_filter": raw_report["rows_before_cutoff"] - int(len(train)),
        },
        "train": {
            **summarise(train),
            "share_rating_at_least_threshold": float(liked.mean()) if len(train) else None,
            "users_with_enough_positives": int(eligible_user.sum()),
        },
        "test": summarise(test),
        "test_filter_steps": [
            {"step": "ratings in the test window", "rows": int(len(window)), "removed": 0},
            {"step": f"rating >= {config.positive_threshold}", "rows": int(len(step_positive)),
             "removed": int(len(window) - len(step_positive))},
            {"step": f"user has >= {config.min_user_train_positives} liked movies in train",
             "rows": int(len(step_user)), "removed": int(len(step_positive) - len(step_user))},
            {"step": "movie is in the train catalogue", "rows": int(len(test)),
             "removed": int(len(step_user) - len(test))},
        ],
        "test_window_share_rating_at_least_threshold": float(len(step_positive) / len(window)) if len(window) else None,
        # Why the user filter removes rows: users who are new, or have too little history, at the cutoff.
        "test_window_users": window_users(step_positive, train, eligible_user),
        f"ratings_per_month_{year}": monthly,
    }
    return Prepared(train=train, test=test, movies=catalog, report=report)


def window_users(liked_in_window: pd.DataFrame, train: pd.DataFrame, eligible_user: np.ndarray) -> Dict[str, int]:
    """Users with a liked rating in the test window, split by how much training history they have."""
    users = np.unique(liked_in_window["userId"].to_numpy())
    train_user = np.bincount(train["userId"].to_numpy(), minlength=len(eligible_user)) > 0
    has_history = train_user[users]
    eligible = eligible_user[users]
    return {
        "with_a_liked_rating_in_window": int(len(users)),
        "without_any_train_rating": int((~has_history).sum()),
        "with_train_ratings_but_too_few_liked": int((has_history & ~eligible).sum()),
        "eligible": int(eligible.sum()),
    }


def in_catalog_lookup(movie_ids: np.ndarray, in_catalog: np.ndarray) -> np.ndarray:
    """Boolean mask for movie ids, safe for ids beyond those seen in the ratings."""
    known = movie_ids < len(in_catalog)
    mask = np.zeros(len(movie_ids), dtype=bool)
    mask[known] = in_catalog[movie_ids[known]]
    return mask


# ---------------------------------------------------------------- output


def write_parquet(frame: pd.DataFrame, path: str) -> None:
    # No index and no pandas metadata: the file content depends on the data only.
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")


def git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"git_commit": commit, "git_uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_uncommitted_changes": None}


def write_outputs(prepared: Prepared, config: DataConfig, out_dir: str, source: str, started: float) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    frames = {"train.parquet": prepared.train, "test.parquet": prepared.test, "movies.parquet": prepared.movies}
    for name in DATA_FILES:
        write_parquet(frames[name], os.path.join(out_dir, name))

    stats = {
        "config": asdict(config),
        "cutoff_timestamp": config.cutoff_timestamp,
        "test_end_timestamp": config.test_end_timestamp,
        "source": source,
        **prepared.report,
        "sha256": {name: sha256_file(os.path.join(out_dir, name)) for name in DATA_FILES},
        **git_state(),
        "generated_at_utc": iso(time.time()),
        "duration_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, STATS_FILE), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    return stats


def print_summary(stats: Dict[str, Any]) -> None:
    raw, train, test = stats["raw"], stats["train"], stats["test"]
    print(f"\nRaw: {raw['rows']:,} ratings, {raw['time_range_utc']['first']} to {raw['time_range_utc']['last']}")
    print(f"Train: {train['rows']:,} rows, {train['users']:,} users, {train['movies']:,} movies "
          f"({train['time_range_utc']['first']} to {train['time_range_utc']['last']})")
    print(f"Test:  {test['rows']:,} rows, {test['users']:,} users, {test['movies']:,} movies "
          f"({test['time_range_utc']['first']} to {test['time_range_utc']['last']})")
    print("Test filter steps:")
    for step in stats["test_filter_steps"]:
        print(f"  {step['step']:<48} {step['rows']:>10,} rows  (removed {step['removed']:,})")
    monthly_key = next(key for key in stats if key.startswith("ratings_per_month_"))
    print("Ratings per month: " + ", ".join(f"{month} {count:,}" for month, count in stats[monthly_key].items()))
    for name, digest in stats["sha256"].items():
        print(f"  sha256 {name}: {digest}")
    if test["rows"] < SMALL_TEST_ROWS:
        print(f"WARNING: the test set has only {test['rows']:,} rows; consider another cutoff.")


def run(config: DataConfig, source: str = "r2", out_root: str = PROCESSED_DIR,
        storage: Optional[Storage] = None) -> Tuple[str, Dict[str, Any]]:
    """Prepare one cutoff; returns the output directory and the stats."""
    started = time.time()
    ratings_path, movies_path, source_description = resolve_source(source, storage)
    prepared = prepare(read_ratings(ratings_path), read_movies(movies_path), config)
    out_dir = os.path.join(out_root, config.cutoff)
    return out_dir, write_outputs(prepared, config, out_dir, source_description, started)


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens ratings: time-based split with train-only filters.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD (UTC); overrides configs/data.yaml")
    parser.add_argument("--source", default="r2", help="'r2' (default) or 'local:<directory with ratings.csv and movies.csv>'")
    parser.add_argument("--config", default=DATA_CONFIG_PATH)
    parser.add_argument("--upload", action="store_true", help="upload the result to R2 at processed/<cutoff>/")
    args = parser.parse_args()

    config = load_data_config(args.config, cutoff=args.cutoff)
    storage = None
    if args.upload or args.source == "r2":
        # Fail before the long computation if R2 is not configured.
        try:
            storage = Storage.from_env()
        except RuntimeError as error:
            raise SystemExit(str(error))
    print(f"Preparing cutoff {config.cutoff} from {args.source}")
    out_dir, stats = run(config, args.source, storage=storage)
    print_summary(stats)
    print(f"Saved {out_dir} in {stats['duration_seconds']} s")

    if args.upload:
        prefix = f"{PROCESSED_PREFIX}/{config.cutoff}"
        try:
            result = upload_directory(storage, out_dir, prefix, DATA_FILES)
        except RemoteConflictError as error:
            raise SystemExit(f"Upload stopped: {error}")
        for name, status in result.items():
            print(f"  s3://{storage.bucket}/{prefix}/{name}: {status}")


if __name__ == "__main__":
    main()
