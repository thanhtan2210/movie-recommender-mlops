"""Package the champion model for serving.

Downloads the model version marked as champion from the MLflow registry into
serving_model/, with the movie titles and a meta.json. The Docker image copies
that directory at build time, so the running container needs no credentials.

Run from the repo root:
  python -m src.export_champion            # from the local MLflow store
  python -m src.export_champion --remote   # from DagsHub (and movie titles from R2 if not on disk)

serving_model/
  model/        the MLflow pyfunc model (artifacts/state.npz inside)
  movies.json   movie_id, title, year for the catalogue
  meta.json     version, cutoff, configuration, production metrics
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src import tracking
from src.data import processed_dir, read_parquet
from src.prepare import PROCESSED_PREFIX, iso
from src.recommender import STATE_FILE
from src.storage import Storage

SERVING_DIR = "serving_model"
MODEL_SUBDIR = "model"
MOVIES_FILE = "movies.json"
META_FILE = "meta.json"
MOVIES_PARQUET = "movies.parquet"
YEAR_PATTERN = r"\((\d{4})\)"
CONFIG_PARAMS = ("k", "train_window", "blend_weight", "random_state")


def state_path(serving_dir: str) -> str:
    return os.path.join(serving_dir, MODEL_SUBDIR, "artifacts", STATE_FILE)


def movies_table(movies: pd.DataFrame) -> list:
    """movie_id, title and release year (from the title, None when it has none)."""
    years = movies["title"].str.extract(YEAR_PATTERN, expand=False)
    return [{"movie_id": int(movie_id), "title": str(title), "year": None if pd.isna(year) else int(year)}
            for movie_id, title, year in zip(movies["movieId"], movies["title"], years)]


def write_serving_dir(serving_dir: str, model_dir: str, movies: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Assemble serving_dir from a pyfunc model directory, the catalogue and the metadata."""
    target = os.path.join(serving_dir, MODEL_SUBDIR)
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(serving_dir, exist_ok=True)
    shutil.copytree(model_dir, target)
    with open(os.path.join(serving_dir, MOVIES_FILE), "w", encoding="utf-8") as f:
        json.dump(movies_table(movies), f, ensure_ascii=False)
    with open(os.path.join(serving_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def load_movies(cutoff: str, remote: bool) -> pd.DataFrame:
    """The catalogue of the cutoff: from data/processed/ if present, else (with --remote) from R2."""
    directory = processed_dir(cutoff)
    if not os.path.exists(os.path.join(directory, MOVIES_PARQUET)):
        if not remote:
            raise SystemExit(f"{os.path.join(directory, MOVIES_PARQUET)} not found. "
                             f"Run: python -m src.prepare --cutoff {cutoff}, or use --remote to fetch it from R2.")
        Storage.from_env().download(f"{PROCESSED_PREFIX}/{cutoff}/{MOVIES_PARQUET}",
                                    os.path.join(directory, MOVIES_PARQUET))
    return read_parquet(directory, MOVIES_PARQUET)


def export(remote: bool = False, serving_dir: str = SERVING_DIR) -> Dict[str, Any]:
    tracking_uri = tracking.setup(remote=remote)
    print(f"Reading the champion from: {tracking_uri}" + ("" if remote else "  (pass --remote to read from DagsHub)"))
    champion: Optional[Dict[str, Any]] = tracking.get_champion()
    if champion is None:
        raise SystemExit(f"No champion version of '{tracking.REGISTERED_MODEL}' in this registry.")

    client = MlflowClient()
    run = client.get_run(champion["run_id"])
    cutoff = champion["tags"].get("cutoff") or run.data.params["cutoff"]
    workdir = tempfile.mkdtemp(prefix="champion_")
    try:
        model_dir = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,
                                                        artifact_path=tracking.MODEL_ARTIFACT_PATH, dst_path=workdir)
        stats_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="stats.json",
                                                         dst_path=workdir)
        with open(stats_file, encoding="utf-8") as f:
            stats = json.load(f)
        meta = {
            "model_name": tracking.REGISTERED_MODEL,
            "model_version": champion["version"],
            "model_type": champion["model_type"],
            "cutoff": cutoff,
            "trained_before": champion["tags"].get("trained_before", cutoff),
            "trained_on_rows": stats["train"]["rows"],
            "config": {name: run.data.params[name] for name in CONFIG_PARAMS if name in run.data.params},
            "production_metrics": dict(sorted(run.data.metrics.items())),
            "train_sha256": run.data.params.get("train_sha256"),
            "git_commit": run.data.params.get("git_commit"),
            "run_id": run.info.run_id,
            "registry": "local" if tracking_uri == tracking.LOCAL_TRACKING_URI else tracking_uri,
            "exported_at_utc": iso(time.time()),
        }
        write_serving_dir(serving_dir, model_dir, load_movies(cutoff, remote), meta)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    size = os.path.getsize(state_path(serving_dir)) / 1024 ** 2
    print(f"Exported {meta['model_name']} version {meta['model_version']} ({meta['model_type']}, cutoff {cutoff}) "
          f"to {serving_dir}/ ({size:.1f} MB of model state)")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Download the champion model into serving_model/ for the API.")
    parser.add_argument("--remote", action="store_true",
                        help="read the registry on DagsHub (default: the local MLflow store)")
    parser.add_argument("--out", default=SERVING_DIR)
    args = parser.parse_args()
    export(args.remote, args.out)


if __name__ == "__main__":
    main()
