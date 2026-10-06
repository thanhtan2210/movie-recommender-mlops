import os

import pytest

from src import config as cfg
from tests.conftest import CUTOFF, DAY, REPO_ROOT


def test_repo_config_has_the_documented_defaults():
    config = cfg.load_data_config(os.path.join(REPO_ROOT, "configs", "data.yaml"))

    assert config == cfg.DataConfig(min_item_ratings=50, positive_threshold=4.0, test_window_days=30,
                                    min_user_train_positives=5, cutoff="2019-06-01")


def test_cutoff_is_midnight_utc_and_the_window_is_exclusive():
    config = cfg.DataConfig(cutoff="2019-06-01", test_window_days=30)

    assert config.cutoff_timestamp == CUTOFF
    assert config.test_end_timestamp == CUTOFF + 30 * DAY


def test_overrides_replace_yaml_values_but_none_does_not(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("cutoff: 2019-03-01\nmin_item_ratings: 10\n", encoding="utf-8")

    assert cfg.load_data_config(str(path)).cutoff == "2019-03-01"  # unquoted yaml date
    assert cfg.load_data_config(str(path), cutoff=None).cutoff == "2019-03-01"
    overridden = cfg.load_data_config(str(path), cutoff="2019-09-01")
    assert overridden.cutoff == "2019-09-01"
    assert overridden.min_item_ratings == 10


def test_unknown_keys_and_bad_cutoffs_are_rejected(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("min_item_rating: 10\n", encoding="utf-8")  # typo
    with pytest.raises(ValueError, match="Unknown keys"):
        cfg.load_data_config(str(path))

    with pytest.raises(ValueError):
        cfg.DataConfig(cutoff="June 2019")
    with pytest.raises(ValueError):
        cfg.load_data_config(os.path.join(REPO_ROOT, "configs", "data.yaml"), cutoff="2019-13-01")


@pytest.fixture
def no_dotenv(monkeypatch):
    """Keep a developer's real .env out of the tests."""
    monkeypatch.setattr(cfg, "load_dotenv", lambda: None)
    for name in (*cfg.R2_VARIABLES, "S3_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)


def test_r2_settings_report_missing_variables(no_dotenv, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")

    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"):
        cfg.load_r2_settings()


def test_r2_settings_default_bucket(no_dotenv, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    assert cfg.load_r2_settings().bucket == "movie-mlops"
    monkeypatch.setenv("S3_BUCKET_NAME", "other-bucket")
    assert cfg.load_r2_settings().bucket == "other-bucket"
