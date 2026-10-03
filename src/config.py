"""Configuration: configs/data.yaml for the data step, .env for R2 credentials."""
import datetime
import os
from dataclasses import dataclass, fields, replace

import yaml
from dotenv import load_dotenv

DATA_CONFIG_PATH = os.path.join("configs", "data.yaml")
DEFAULT_BUCKET = "movie-mlops"
R2_VARIABLES = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


@dataclass(frozen=True)
class DataConfig:
    min_item_ratings: int = 50
    positive_threshold: float = 4.0
    test_window_days: int = 30
    min_user_train_positives: int = 5
    cutoff: str = "2019-06-01"

    def __post_init__(self):
        # Fails early on a cutoff that is not a YYYY-MM-DD date.
        datetime.date.fromisoformat(self.cutoff)
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be positive")

    @property
    def cutoff_timestamp(self) -> int:
        """The cutoff as Unix seconds; the date is read as midnight UTC."""
        day = datetime.date.fromisoformat(self.cutoff)
        return int(datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc).timestamp())

    @property
    def test_end_timestamp(self) -> int:
        """End of the test window (exclusive)."""
        return self.cutoff_timestamp + self.test_window_days * 86400


def load_data_config(path: str = DATA_CONFIG_PATH, **overrides) -> DataConfig:
    """Read the yaml file; keyword arguments that are not None override it."""
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    known = {field.name for field in fields(DataConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    if "cutoff" in values:
        values["cutoff"] = str(values["cutoff"])  # yaml may parse an unquoted date
    config = DataConfig(**values)
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


@dataclass(frozen=True)
class R2Settings:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str


def load_r2_settings() -> R2Settings:
    """R2 credentials from the environment or a .env file in the working directory."""
    load_dotenv()
    missing = [name for name in R2_VARIABLES if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in the environment or in a .env file (see .env.example)."
        )
    return R2Settings(
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        bucket=os.environ.get("S3_BUCKET_NAME") or DEFAULT_BUCKET,
    )
