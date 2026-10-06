"""storage.py against an in-memory S3 client: no network, no credentials."""
import hashlib
import os

import pytest

from src.storage import STATS_FILE, RemoteConflictError, sha256_file, upload_directory

DATA_FILES = ["train.parquet", "test.parquet", "movies.parquet"]
PREFIX = "processed/2019-06-01"


@pytest.fixture
def local_dir(tmp_path):
    directory = tmp_path / "2019-06-01"
    directory.mkdir()
    for name in DATA_FILES:
        (directory / name).write_bytes(f"content of {name}".encode())
    (directory / STATS_FILE).write_text('{"run": 1}', encoding="utf-8")
    return str(directory)


def test_sha256_file(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc" * 1000)

    assert sha256_file(str(path)) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_upload_records_the_hash_and_download_returns_the_content(storage, local_dir, tmp_path):
    source = os.path.join(local_dir, "train.parquet")
    storage.upload(source, "some/key.parquet")

    assert storage.remote_sha256("some/key.parquet") == sha256_file(source)
    target = tmp_path / "nested" / "copy.parquet"
    storage.download("some/key.parquet", str(target))
    assert target.read_bytes() == b"content of train.parquet"


def test_head_and_hash_of_a_missing_object(storage):
    assert storage.head("missing") is None
    assert storage.remote_sha256("missing") is None


def test_remote_hash_without_metadata_is_computed_from_the_content(storage):
    storage.client.objects[(storage.bucket, "raw/manual.csv")] = {"body": b"a,b\n1,2\n", "metadata": {}}

    assert storage.remote_sha256("raw/manual.csv") == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_list_objects_filters_by_prefix_across_pages(storage, local_dir):
    for key in ("raw/ratings.csv", "raw/movies.csv", "raw/tags.csv", "processed/x/train.parquet"):
        storage.upload(os.path.join(local_dir, "train.parquet"), key)

    assert sorted(storage.list_objects("raw/")) == ["raw/movies.csv", "raw/ratings.csv", "raw/tags.csv"]
    assert len(storage.list_objects()) == 4
    assert storage.list_objects("nothing/") == {}


def test_download_if_changed(storage, local_dir, tmp_path):
    storage.upload(os.path.join(local_dir, "train.parquet"), "raw/ratings.csv")
    target = str(tmp_path / "ratings.csv")

    assert storage.download_if_changed("raw/ratings.csv", target) is True
    assert storage.download_if_changed("raw/ratings.csv", target) is False
    with open(target, "wb") as f:
        f.write(b"truncated")
    assert storage.download_if_changed("raw/ratings.csv", target) is True
    with pytest.raises(FileNotFoundError):
        storage.download_if_changed("raw/missing.csv", target)


def test_first_upload_sends_everything(storage, local_dir):
    result = upload_directory(storage, local_dir, PREFIX, DATA_FILES)

    assert result == {name: "uploaded" for name in DATA_FILES + [STATS_FILE]}
    assert sorted(storage.list_objects(PREFIX + "/")) == sorted(f"{PREFIX}/{name}" for name in DATA_FILES + [STATS_FILE])


def test_same_content_is_skipped(storage, local_dir):
    upload_directory(storage, local_dir, PREFIX, DATA_FILES)
    uploads = len(storage.client.uploads)
    # stats.json changes on every run (it records the run time); it must not be re-uploaded.
    with open(os.path.join(local_dir, STATS_FILE), "w", encoding="utf-8") as f:
        f.write('{"run": 2}')

    result = upload_directory(storage, local_dir, PREFIX + "/", DATA_FILES)

    assert result == {name: "skipped" for name in DATA_FILES + [STATS_FILE]}
    assert len(storage.client.uploads) == uploads
    assert storage.client.objects[(storage.bucket, f"{PREFIX}/{STATS_FILE}")]["body"] == b'{"run": 1}'


def test_different_content_stops_without_overwriting(storage, local_dir):
    upload_directory(storage, local_dir, PREFIX, DATA_FILES)
    before = dict(storage.client.objects)
    with open(os.path.join(local_dir, "test.parquet"), "wb") as f:
        f.write(b"something else")

    with pytest.raises(RemoteConflictError, match="test.parquet"):
        upload_directory(storage, local_dir, PREFIX, DATA_FILES)

    assert storage.client.objects == before


def test_interrupted_upload_is_completed(storage, local_dir):
    """Only train.parquet made it to the remote: the rest is uploaded, train is left alone."""
    storage.upload(os.path.join(local_dir, "train.parquet"), f"{PREFIX}/train.parquet")

    result = upload_directory(storage, local_dir, PREFIX, DATA_FILES)

    assert result == {"train.parquet": "skipped", "test.parquet": "uploaded", "movies.parquet": "uploaded",
                      STATS_FILE: "uploaded"}
