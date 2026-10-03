"""Load a prepared cutoff and turn its ratings into sparse user x movie matrices."""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from src.config import DataConfig
from src.prepare import DATA_FILES, PROCESSED_DIR, PROCESSED_PREFIX
from src.storage import STATS_FILE, Storage


def processed_dir(cutoff: str) -> str:
    return os.path.join(PROCESSED_DIR, cutoff)


def ensure_processed(cutoff: str) -> str:
    """Directory of a prepared cutoff; downloaded from R2 when it is not on disk."""
    directory = processed_dir(cutoff)
    names = DATA_FILES + [STATS_FILE]
    if all(os.path.exists(os.path.join(directory, name)) for name in names):
        return directory
    try:
        storage = Storage.from_env()
    except RuntimeError as error:
        raise FileNotFoundError(
            f"{directory} is incomplete and R2 is not configured ({error}). "
            f"Run: python -m src.prepare --cutoff {cutoff}"
        )
    for name in names:
        storage.download(f"{PROCESSED_PREFIX}/{cutoff}/{name}", os.path.join(directory, name))
    return directory


def load_stats(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, STATS_FILE), encoding="utf-8") as f:
        return json.load(f)


def data_config_of(stats: Dict[str, Any]) -> DataConfig:
    """The configuration the data was prepared with (not the current yaml file)."""
    return DataConfig(**stats["config"])


def read_parquet(directory: str, name: str) -> pd.DataFrame:
    return pq.read_table(os.path.join(directory, name)).to_pandas(split_blocks=True, self_destruct=True)


@dataclass
class Interactions:
    """Training interactions. Row i is user_ids[i], column j is item_ids[j]."""
    user_ids: np.ndarray  # sorted
    item_ids: np.ndarray  # sorted
    liked: sparse.csr_matrix  # 1.0 where rating >= positive threshold (what the model learns from)
    seen: sparse.csr_matrix  # 1 where the user rated the movie at all (never recommended again)

    def rows_of(self, user_ids) -> np.ndarray:
        return _positions(self.user_ids, np.asarray(user_ids), "user")

    def cols_of(self, item_ids) -> np.ndarray:
        return _positions(self.item_ids, np.asarray(item_ids), "movie")


def _positions(index: np.ndarray, ids: np.ndarray, what: str) -> np.ndarray:
    positions = np.searchsorted(index, ids)
    positions = np.minimum(positions, len(index) - 1)
    if not np.array_equal(index[positions], ids):
        raise KeyError(f"Unknown {what} ids: not present in the training set")
    return positions


def build_interactions(train: pd.DataFrame, positive_threshold: float) -> Interactions:
    """Binary user x movie matrices; the user and movie index come from the training set only."""
    user_ids = np.unique(train["userId"].to_numpy())
    item_ids = np.unique(train["movieId"].to_numpy())
    rows = np.searchsorted(user_ids, train["userId"].to_numpy()).astype(np.int32)
    cols = np.searchsorted(item_ids, train["movieId"].to_numpy()).astype(np.int32)
    shape = (len(user_ids), len(item_ids))

    seen = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=shape)
    positive = train["rating"].to_numpy() >= positive_threshold
    liked = sparse.csr_matrix(
        (np.ones(int(positive.sum()), dtype=np.float32), (rows[positive], cols[positive])), shape=shape)
    # A (user, movie) pair rated twice would otherwise be summed to 2.
    seen.data[:] = 1
    liked.data[:] = 1.0
    return Interactions(user_ids=user_ids, item_ids=item_ids, liked=liked, seen=seen)
