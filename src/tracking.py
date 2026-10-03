"""Experiment tracking and model registry on MLflow: a local file by default, DagsHub with remote=True.

Every run logs: parameters (model type and its parameters, cutoff, sha256 of
the training data, git commit), metrics with their interval bounds and a
`window` tag (gate or production). Only a run whose model is registered also
uploads the pyfunc model and the data's stats.json; evaluation-only runs
carry no artifacts.

The registry holds one model, `movie-recommender`. The version in use is
marked as champion with an alias and, for servers without alias support,
with a `role=champion` tag on the model version.
"""
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec

from src.recommender import STATE_ARTIFACT, Recommender

EXPERIMENT = "movie-recommender"
REGISTERED_MODEL = "movie-recommender"
CHAMPION = "champion"
ROLE_TAG = "role"
MODEL_ARTIFACT_PATH = "model"
LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PINNED_PACKAGES = ("mlflow", "numpy", "scipy", "scikit-learn", "pandas")
METRICS_WITH_INTERVAL = ("hit_rate_at_10", "recall_at_10", "ndcg_at_10", "long_tail_share")

# One column of user ids (a list, an array or a one-column DataFrame); int32 ids are accepted.
SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec(DataType.long)]),
    outputs=Schema([TensorSpec(np.dtype("int64"), (-1, -1))]),
    params=ParamSchema([ParamSpec("n", DataType.integer, 10)]),
)


def setup(remote: bool = False, tracking_uri: Optional[str] = None) -> str:
    """Point MLflow at the local store, or at the server of MLFLOW_TRACKING_URI when remote=True.

    A .env file with DagsHub credentials is never enough to write there: the
    caller has to ask for it explicitly.
    """
    if tracking_uri is None:
        if remote:
            load_dotenv()
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise SystemExit("--remote needs MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and "
                                 "MLFLOW_TRACKING_PASSWORD in the environment or in .env (see .env.example).")
        else:
            tracking_uri = LOCAL_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    return tracking_uri


def flatten_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """metrics.evaluate() output as flat MLflow metrics: value plus interval bounds."""
    flat = {"catalog_coverage": result["catalog_coverage"]["value"]}
    for name in METRICS_WITH_INTERVAL:
        flat[name] = result[name]["value"]
        flat[f"{name}_ci95_low"] = result[name]["ci95_low"]
        flat[f"{name}_ci95_high"] = result[name]["ci95_high"]
    return flat


def save_pyfunc(recommender: Recommender, directory: str) -> str:
    """Write the recommender as an MLflow pyfunc model directory; returns its path."""
    state_path = recommender.save_state(os.path.join(directory, "state"))
    model_path = os.path.join(directory, MODEL_ARTIFACT_PATH)
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=type(recommender)(),  # an empty instance: the data is in the state artifact
        artifacts={STATE_ARTIFACT: state_path},
        code_paths=[SOURCE_DIR],
        signature=SIGNATURE,
        pip_requirements=[f"{name}=={metadata.version(name)}" for name in PINNED_PACKAGES],
    )
    return model_path


@dataclass
class LoggedRun:
    run_id: str
    model_source: str
    version: Optional[str] = None  # registry version, when registered


def log_run(run_name: str, recommender: Recommender, params: Dict[str, Any], tags: Dict[str, str],
            stats_path: str, result: Optional[Dict[str, Any]] = None, register: bool = False,
            version_tags: Optional[Dict[str, str]] = None) -> LoggedRun:
    """Log one run: parameters, tags and metrics.

    With register=True the model and the data's stats.json are uploaded too and
    the model becomes a new registry version. Evaluation-only runs upload nothing.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_type": recommender.model_type, **params})
        mlflow.set_tags(tags)
        if result is not None:
            mlflow.log_metrics(flatten_metrics(result))
        if register:
            mlflow.log_artifact(stats_path)
            workdir = tempfile.mkdtemp(prefix="pyfunc_")
            try:
                # Saved locally and uploaded as plain artifacts: works on any MLflow server version.
                mlflow.log_artifacts(save_pyfunc(recommender, workdir), artifact_path=MODEL_ARTIFACT_PATH)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        logged = LoggedRun(run_id=run.info.run_id, model_source=f"{run.info.artifact_uri}/{MODEL_ARTIFACT_PATH}")

    if register:
        client = MlflowClient()
        try:
            client.create_registered_model(REGISTERED_MODEL)
        except MlflowException:
            pass  # already exists
        version = client.create_model_version(
            REGISTERED_MODEL, source=logged.model_source, run_id=logged.run_id,
            tags={"model_type": recommender.model_type, **(version_tags or {})})
        logged.version = str(version.version)
    return logged


def set_champion(version: str) -> str:
    """Mark a registry version as the champion. Returns the mechanism used: 'alias' or 'tag'.

    The role tag is always written (and removed from the other versions); the
    alias is set as well when the server supports aliases.
    """
    client = MlflowClient()
    for other in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        if other.tags.get(ROLE_TAG) == CHAMPION and str(other.version) != str(version):
            client.delete_model_version_tag(REGISTERED_MODEL, other.version, ROLE_TAG)
    client.set_model_version_tag(REGISTERED_MODEL, version, ROLE_TAG, CHAMPION)
    try:
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, version)
        return "alias"
    except (MlflowException, AttributeError, NotImplementedError):
        return "tag"


def get_champion() -> Optional[Dict[str, Any]]:
    """The champion version: {'version', 'model_type', 'run_id', 'source', 'tags'}, or None if there is none yet."""
    client = MlflowClient()
    found = None
    try:
        found = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
    except (MlflowException, AttributeError, NotImplementedError):
        try:
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        except MlflowException:
            versions = []
        tagged = [v for v in versions if v.tags.get(ROLE_TAG) == CHAMPION]
        if tagged:
            found = max(tagged, key=lambda v: int(v.version))
    if found is None:
        return None
    return {"version": str(found.version), "model_type": found.tags.get("model_type"), "run_id": found.run_id,
            "source": found.source, "tags": dict(found.tags)}


def load_champion():
    """The champion as a loaded pyfunc model: .predict(user_ids, params={'n': 10})."""
    champion = get_champion()
    if champion is None:
        raise LookupError(f"No champion version of '{REGISTERED_MODEL}' in the registry.")
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{champion['version']}")
