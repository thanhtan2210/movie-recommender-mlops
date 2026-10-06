"""Shared fixtures: a tiny ratings set around the cutoff, and an in-memory S3 client."""
import io
import os

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.config import DataConfig
from src.storage import Storage

DAY = 86400
CUTOFF = 1559347200  # 2019-06-01T00:00:00Z
BEFORE = CUTOFF - 100 * DAY


@pytest.fixture
def config() -> DataConfig:
    return DataConfig(min_item_ratings=3, positive_threshold=4.0, test_window_days=30,
                      min_user_train_positives=2, cutoff="2019-06-01")


# (userId, movieId, rating, timestamp)
#
# Train (before the cutoff) - ratings per movie: 10 -> 4, 20 -> 4, 50 -> 3, 60 -> 3,
# 30 -> 2 (below min_item_ratings), 40 -> 0 (only rated in the test window).
# Liked movies per user in the filtered train: u1 3, u2 2, u3 1 (not enough), u4 2, u5 2.
TRAIN_ROWS = [
    (1, 10, 5.0, BEFORE), (1, 20, 4.0, BEFORE + 1), (1, 50, 4.5, BEFORE + 2),
    (2, 10, 4.0, BEFORE), (2, 20, 5.0, BEFORE + 1), (2, 60, 3.0, BEFORE + 2),
    (3, 10, 3.0, BEFORE), (3, 20, 2.0, BEFORE + 1), (3, 60, 5.0, BEFORE + 2),
    (4, 10, 5.0, CUTOFF - 1), (4, 60, 4.0, BEFORE), (4, 30, 4.0, BEFORE + 1), (4, 50, 1.0, BEFORE + 2),
    (5, 20, 4.0, BEFORE), (5, 50, 4.0, BEFORE + 1), (5, 30, 5.0, BEFORE + 2),
]
WINDOW_ROWS = [
    (1, 60, 5.0, CUTOFF + DAY),            # kept
    (1, 40, 5.0, CUTOFF + DAY),            # movie 40 is not in the train catalogue
    (2, 50, 4.0, CUTOFF),                  # kept: the cutoff itself belongs to the test window
    (2, 40, 4.5, CUTOFF + 2 * DAY),        # movie filter
    (2, 30, 2.0, CUTOFF + 2 * DAY),        # not a liked rating
    (3, 50, 5.0, CUTOFF + 3 * DAY),        # user 3 has too few liked movies in train
    (3, 40, 4.0, CUTOFF + 3 * DAY),        # user filter (and movie 40)
    (4, 20, 4.0, CUTOFF + 30 * DAY - 1),   # kept: last second of the window
    (4, 40, 3.0, CUTOFF + 4 * DAY),        # not a liked rating
    (5, 60, 3.5, CUTOFF + 5 * DAY),        # not a liked rating
    (5, 40, 5.0, CUTOFF + 5 * DAY),        # movie filter
]
AFTER_ROWS = [
    (5, 10, 5.0, CUTOFF + 30 * DAY),       # first second after the window: in neither set
]
MOVIE_ROWS = [
    (10, "Movie Ten (1990)", "Drama"), (20, "Movie Twenty (1991)", "Comedy"),
    (30, "Movie Thirty (1992)", "Action"), (40, "Movie Forty (2019)", "Sci-Fi"),
    (50, "Movie Fifty (1994)", "Drama|Romance"), (60, "Movie Sixty (1995)", "Thriller"),
    (70, "Never Rated (2000)", "Documentary"),
]


def ratings_frame(rows=None) -> pd.DataFrame:
    rows = TRAIN_ROWS + WINDOW_ROWS + AFTER_ROWS if rows is None else rows
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return frame.astype({"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})


def movies_frame() -> pd.DataFrame:
    frame = pd.DataFrame(MOVIE_ROWS, columns=["movieId", "title", "genres"])
    return frame.astype({"movieId": "int32", "title": "string", "genres": "string"})


@pytest.fixture
def ratings() -> pd.DataFrame:
    return ratings_frame()


@pytest.fixture
def movies() -> pd.DataFrame:
    return movies_frame()


@pytest.fixture
def raw_dir(tmp_path) -> str:
    """A directory laid out like the MovieLens download: ratings.csv and movies.csv."""
    directory = tmp_path / "raw"
    directory.mkdir()
    ratings_frame().to_csv(directory / "ratings.csv", index=False)
    movies_frame().to_csv(directory / "movies.csv", index=False)
    return str(directory)


class FakeS3Client:
    """The few boto3 S3 client calls the code uses, backed by a dict."""

    def __init__(self):
        self.objects = {}  # (bucket, key) -> {"body": bytes, "metadata": dict}
        self.uploads = []

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        with open(Filename, "rb") as f:
            body = f.read()
        self.objects[(Bucket, Key)] = {"body": body, "metadata": dict((ExtraArgs or {}).get("Metadata", {}))}
        self.uploads.append(Key)

    def download_file(self, Bucket, Key, Filename):
        with open(Filename, "wb") as f:
            f.write(self._get(Bucket, Key)["body"])

    def head_object(self, Bucket, Key):
        item = self._get(Bucket, Key)
        return {"ContentLength": len(item["body"]), "Metadata": item["metadata"]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self._get(Bucket, Key)["body"])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix=""):
                contents = [{"Key": key, "Size": len(item["body"])}
                            for (bucket, key), item in sorted(client.objects.items())
                            if bucket == Bucket and key.startswith(Prefix)]
                # Two pages, to exercise pagination.
                middle = len(contents) // 2
                return [{"Contents": contents[:middle]}, {"Contents": contents[middle:]}] if contents else [{}]

        return Paginator()

    def _get(self, bucket, key):
        if (bucket, key) not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return self.objects[(bucket, key)]


@pytest.fixture
def storage() -> Storage:
    return Storage(FakeS3Client(), "test-bucket")


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty directory so nothing is read from or written to the repo."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return str(work)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
