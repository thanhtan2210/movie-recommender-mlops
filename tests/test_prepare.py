import dataclasses
import json
import os
import sys

import pandas as pd
import pytest

from src import prepare as prep
from src.storage import Storage, sha256_file
from tests.conftest import CUTOFF, DAY, REPO_ROOT, ratings_frame


def pairs(frame):
    return sorted(zip(frame["userId"].tolist(), frame["movieId"].tolist()))


@pytest.fixture
def prepared(ratings, movies, config):
    return prep.prepare(ratings, movies, config)


def test_train_holds_only_ratings_before_the_cutoff(prepared):
    assert (prepared.train["timestamp"] < CUTOFF).all()
    # The last second before the cutoff is still training data.
    assert prepared.train["timestamp"].max() == CUTOFF - 1


def test_test_set_lies_inside_the_window(prepared):
    timestamps = prepared.test["timestamp"]
    assert (timestamps >= CUTOFF).all()
    assert (timestamps < CUTOFF + 30 * DAY).all()
    # Both edges: the cutoff second is in, the first second after the window is out.
    assert timestamps.min() == CUTOFF
    assert timestamps.max() == CUTOFF + 30 * DAY - 1
    assert (5, 10) not in pairs(prepared.test)
    assert prepared.report["raw"]["rows_after_test_window"] == 1


def test_movie_filter_uses_train_counts_only(prepared):
    """Movie 40 has 5 ratings in the test window and none in train: it must be dropped."""
    assert set(prepared.movies["movieId"]) == {10, 20, 50, 60}
    assert 40 not in set(prepared.train["movieId"]) | set(prepared.test["movieId"])
    # Movie 30 has only 2 training ratings (minimum is 3).
    assert 30 not in set(prepared.train["movieId"])
    assert prepared.report["catalog"] == {
        "movies_with_any_train_rating": 5,
        "movies_kept": 4,
        "movies_kept_without_metadata": 0,
        "train_rows_removed_by_movie_filter": 2,
    }
    assert len(prepared.train) == 14


def test_test_keeps_liked_ratings_of_eligible_users(prepared):
    assert pairs(prepared.test) == [(1, 60), (2, 50), (4, 20)]
    assert (prepared.test["rating"] >= 4.0).all()
    # User 3 has a liked rating of a catalogue movie in the window, but only one liked movie in train.
    assert 3 not in set(prepared.test["userId"])
    assert prepared.report["train"]["users_with_enough_positives"] == 4


def test_user_eligibility_counts_liked_movies_in_the_filtered_train(ratings, movies, config):
    """User 4 liked movies 10, 60 and 30; movie 30 is not in the catalogue, so 2 count."""
    stricter = dataclasses.replace(config, min_user_train_positives=3)

    result = prep.prepare(ratings, movies, stricter)

    assert pairs(result.test) == [(1, 60)]


def test_filter_steps_are_counted(prepared):
    steps = prepared.report["test_filter_steps"]

    assert [step["rows"] for step in steps] == [11, 8, 6, 3]
    assert [step["removed"] for step in steps] == [0, 3, 2, 3]
    assert steps[-1]["rows"] == len(prepared.test)
    assert prepared.report["test_window_share_rating_at_least_threshold"] == pytest.approx(8 / 11)


def test_window_users_explain_the_user_filter(ratings, movies, config):
    # Add a user who only appears in the test window.
    extra = ratings_frame(rows=[(9, 10, 5.0, CUTOFF + DAY)])
    result = prep.prepare(pd.concat([ratings, extra], ignore_index=True), movies, config)

    assert result.report["test_window_users"] == {
        "with_a_liked_rating_in_window": 6,       # users 1, 2, 3, 4, 5, 9
        "without_any_train_rating": 1,            # user 9
        "with_train_ratings_but_too_few_liked": 1,  # user 3
        "eligible": 4,
    }


def test_rows_are_sorted_and_typed(prepared):
    for frame in (prepared.train, prepared.test):
        keys = list(zip(frame["userId"], frame["timestamp"], frame["movieId"]))
        assert keys == sorted(keys)
        assert frame.dtypes.astype(str).to_dict() == {
            "userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}
    assert prepared.movies["movieId"].is_monotonic_increasing


def test_ratings_per_month(prepared):
    monthly = prepared.report["ratings_per_month_2019"]

    assert list(monthly) == [f"2019-{month:02d}" for month in range(1, 13)]
    assert monthly["2019-06"] == 11
    assert monthly["2019-07"] == 1
    assert sum(monthly.values()) == 28


def test_empty_test_window_does_not_fail(movies, config):
    result = prep.prepare(ratings_frame(rows=[(1, 10, 5.0, CUTOFF - 5)] * 3), movies, config)

    assert len(result.test) == 0
    assert result.report["test"]["time_range_utc"] == {"first": None, "last": None}


# ---------------------------------------------------------------- files


def run_local(config, raw_dir, out_root):
    return prep.run(config, source=f"local:{raw_dir}", out_root=str(out_root))


def test_outputs_and_stats(config, raw_dir, tmp_path):
    out_dir, stats = run_local(config, raw_dir, tmp_path / "out")

    assert out_dir.endswith(os.path.join("out", "2019-06-01"))
    assert sorted(os.listdir(out_dir)) == ["movies.parquet", "stats.json", "test.parquet", "train.parquet"]
    with open(os.path.join(out_dir, "stats.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == stats
    assert stats["config"]["cutoff"] == "2019-06-01"
    assert stats["cutoff_timestamp"] == CUTOFF
    assert stats["train"]["rows"] == 14 and stats["train"]["users"] == 5 and stats["train"]["movies"] == 4
    assert stats["test"]["rows"] == 3 and stats["test"]["users"] == 3
    assert stats["test"]["time_range_utc"] == {"first": "2019-06-01T00:00:00Z", "last": "2019-06-30T23:59:59Z"}
    assert stats["train"]["share_rating_at_least_threshold"] == pytest.approx(10 / 14)
    for name in prep.DATA_FILES:
        assert stats["sha256"][name] == sha256_file(os.path.join(out_dir, name))
    assert "generated_at_utc" in stats and "duration_seconds" in stats and "git_commit" in stats

    train = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
    assert list(train.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert len(train) == 14


def test_running_twice_gives_identical_files(config, raw_dir, tmp_path):
    _, first = run_local(config, raw_dir, tmp_path / "first")
    _, second = run_local(config, raw_dir, tmp_path / "second")

    assert first["sha256"] == second["sha256"]


def test_output_does_not_depend_on_input_row_order(config, raw_dir, tmp_path):
    _, expected = run_local(config, raw_dir, tmp_path / "ordered")

    shuffled_dir = tmp_path / "shuffled"
    shuffled_dir.mkdir()
    ratings_frame().sample(frac=1.0, random_state=7).to_csv(shuffled_dir / "ratings.csv", index=False)
    pd.read_csv(os.path.join(raw_dir, "movies.csv")).iloc[::-1].to_csv(shuffled_dir / "movies.csv", index=False)
    _, shuffled = run_local(config, str(shuffled_dir), tmp_path / "out")

    assert shuffled["sha256"] == expected["sha256"]


def test_source_must_be_r2_or_local():
    with pytest.raises(ValueError, match="Unknown source"):
        prep.resolve_source("ftp://somewhere")


def test_r2_source_downloads_raw_files_once(storage, raw_dir, isolated_cwd, monkeypatch):
    storage.upload(os.path.join(raw_dir, "ratings.csv"), prep.RAW_RATINGS_KEY)
    storage.upload(os.path.join(raw_dir, "movies.csv"), prep.RAW_MOVIES_KEY)
    monkeypatch.setattr(Storage, "from_env", classmethod(lambda cls: storage))

    ratings_path, movies_path, description = prep.resolve_source("r2")

    assert description == "r2://test-bucket/raw/"
    assert sha256_file(ratings_path) == sha256_file(os.path.join(raw_dir, "ratings.csv"))
    assert sha256_file(movies_path) == sha256_file(os.path.join(raw_dir, "movies.csv"))
    # A second call finds the files on disk and does not download again.
    downloads = []
    monkeypatch.setattr(storage, "download", lambda key, path: downloads.append(key))
    prep.resolve_source("r2")
    assert downloads == []


# ---------------------------------------------------------------- command line


@pytest.fixture
def cli(raw_dir, isolated_cwd, monkeypatch):
    """Run main() in an empty directory with the repo's config file."""
    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["prepare", "--config", os.path.join(REPO_ROOT, "configs", "data.yaml"),
                                          "--source", f"local:{raw_dir}", *extra])
        prep.main()
    return run


def test_cli_writes_the_cutoff_directory(cli, capsys, monkeypatch):
    # Without --remote, R2 is never contacted, even if credentials are available.
    monkeypatch.setattr(Storage, "from_env", classmethod(lambda cls: pytest.fail("R2 must not be used")))
    cli("--cutoff", "2019-06-01")

    out_dir = os.path.join("data", "processed", "2019-06-01")
    assert os.path.exists(os.path.join(out_dir, "stats.json"))
    output = capsys.readouterr().out
    assert "Test filter steps" in output
    assert "only (pass --remote to upload to R2)" in output
    # The repo config asks for 50 ratings per movie; the tiny fixture has none, and the CLI says so.
    assert "WARNING: the test set has only 0 rows" in output


def test_cli_upload_is_idempotent_and_never_overwrites(cli, storage, monkeypatch, capsys):
    monkeypatch.setattr(Storage, "from_env", classmethod(lambda cls: storage))

    cli("--cutoff", "2019-06-01", "--remote")
    assert sorted(storage.list_objects("processed/2019-06-01/")) == [
        f"processed/2019-06-01/{name}" for name in ["movies.parquet", "stats.json", "test.parquet", "train.parquet"]]

    uploads_before = len(storage.client.uploads)
    cli("--cutoff", "2019-06-01", "--remote")
    assert len(storage.client.uploads) == uploads_before
    assert capsys.readouterr().out.count("skipped") == 4

    # A different result for the same cutoff must not replace what is on R2.
    remote_before = dict(storage.client.objects)
    monkeypatch.setattr(prep, "SMALL_TEST_ROWS", 0)
    with open(os.path.join(REPO_ROOT, "configs", "data.yaml"), encoding="utf-8") as f:
        changed = f.read().replace("min_item_ratings: 50", "min_item_ratings: 3")
    with open("other.yaml", "w", encoding="utf-8") as f:
        f.write(changed)
    monkeypatch.setattr(sys, "argv", sys.argv[:2] + ["other.yaml"] + sys.argv[3:])
    with pytest.raises(SystemExit, match="Upload stopped"):
        prep.main()
    assert storage.client.objects == remote_before


def test_cli_stops_early_without_r2_credentials(cli, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "load_dotenv", lambda: None)
    for name in cfg.R2_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit, match="Missing environment variables"):
        cli("--cutoff", "2019-06-01", "--remote")

    # Nothing was computed: the check runs before the data is read.
    assert not os.path.exists(os.path.join("data", "processed"))
