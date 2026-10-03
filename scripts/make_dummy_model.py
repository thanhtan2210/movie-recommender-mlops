"""Build a small synthetic model in serving_model/, for building and smoke-testing the image without secrets.

Run from the repo root:  python -m scripts.make_dummy_model
"""
import argparse
import tempfile

import numpy as np
import pandas as pd

from src import tracking
from src.export_champion import SERVING_DIR, write_serving_dir
from src.model import BlendRecommender

CUTOFF = "2019-06-01"
CUTOFF_TIMESTAMP = 1559347200
DAY = 86400


def synthetic_ratings(n_users: int = 60, n_movies: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two taste groups: even users like the first half of the movies, odd users the second half."""
    rng = np.random.default_rng(seed)
    half = n_movies // 2
    rows = []
    for user in range(1, n_users + 1):
        own = np.arange(1, half + 1) if user % 2 == 0 else np.arange(half + 1, n_movies + 1)
        for movie in rng.choice(own, 8, replace=False):
            rows.append((user, int(movie), float(rng.choice([4.0, 5.0])),
                         int(CUTOFF_TIMESTAMP - rng.integers(1, 80) * DAY)))
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return frame.astype({"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})


def main():
    parser = argparse.ArgumentParser(description="Write a synthetic model to a serving directory.")
    parser.add_argument("--out", default=SERVING_DIR)
    args = parser.parse_args()

    ratings = synthetic_ratings()
    model = BlendRecommender.fit(ratings, CUTOFF, CUTOFF_TIMESTAMP, 4.0, k=4, train_window="1y", blend_weight=1.0)
    movies = pd.DataFrame({"movieId": model.item_ids,
                           "title": [f"Dummy Movie {movie_id} ({1980 + movie_id})" for movie_id in model.item_ids]})
    meta = {"model_name": "movie-recommender", "model_version": "dummy", "model_type": model.model_type,
            "cutoff": CUTOFF, "trained_before": CUTOFF, "trained_on_rows": int(len(ratings)),
            "config": {"k": 4, "train_window": "1y", "blend_weight": 1.0}, "production_metrics": {}}
    write_serving_dir(args.out, tracking.save_pyfunc(model, tempfile.mkdtemp(prefix="dummy_")), movies, meta)
    print(f"Wrote a synthetic model ({len(model.user_ids)} users, {len(model.item_ids)} movies) to {args.out}/")


if __name__ == "__main__":
    main()
