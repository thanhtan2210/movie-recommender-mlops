"""One pipeline run for a cutoff C: prepare, gate, refit, production evaluation.

  1. prepare   data for C (train = ratings before C, production window = [C, C + 30 days)).
  2. gate      on the last 30 days before C: the challenger (the blend model) and the
               current champion's model type are both fitted on ratings before C - 30 days
               and scored on [C - 30 days, C).
  3. decide    promote the challenger if the lower bound of the 95% interval of
               (challenger - champion) NDCG@10 is above 0; otherwise keep the champion.
  4. refit     the winner on all ratings before C, register it, mark it as champion.
  5. production: score the new champion and popularity on [C, C + 30 days), a window
               no decision has looked at.

The first run registers popularity as the initial champion.

Run from the repo root:
  python -m src.pipeline --cutoff 2019-07-01 [--source local:<dir>] [--remote]

By default nothing leaves the machine: runs go to a local MLflow store
(mlflow.db) and nothing is uploaded to R2. Only --remote logs to DagsHub and
uploads processed/<C>/ to R2.

Writes reports/<C>/gate.json and reports/<C>/production.json.
"""
import argparse
import json
import os
from typing import Any, Dict, Optional

from src import evaluate, tracking
from src.config import DataConfig, load_data_config
from src.data import read_parquet
from src.prepare import DATA_FILES, PROCESSED_PREFIX, git_state, iso, run as run_prepare
from src.recommender import Recommender
from src.storage import STATS_FILE, RemoteConflictError, Storage, upload_directory
from src.train import BLEND, POPULARITY, TrainConfig, fit_model, inner_split, load_train_config, model_params

GATE_DAYS = 30
REPORT_DIR = "reports"
GATE_FILE = "gate.json"
PRODUCTION_FILE = "production.json"
RULE = "promote the challenger if the lower bound of the 95% interval of (challenger - champion) NDCG@10 is above 0"


def report_dir(cutoff: str) -> str:
    return os.path.join(REPORT_DIR, cutoff)


def gate_decision(difference: Dict[str, Any], challenger_type: str, champion_type: str) -> Dict[str, Any]:
    """Apply the promotion rule to the paired NDCG@10 difference (challenger - champion)."""
    ndcg = difference["ndcg_at_10"]
    promote = bool(ndcg["ci95_low"] > 0)
    interval = f"{ndcg['value']:+.4f} [{ndcg['ci95_low']:+.4f}, {ndcg['ci95_high']:+.4f}]"
    if promote:
        reason = f"NDCG@10 difference {interval}: the lower bound is above 0."
    elif challenger_type == champion_type:
        reason = (f"The champion is already a {champion_type} model with the same parameters, so the "
                  f"difference is {interval}; the champion is kept and refitted on the newer data.")
    else:
        reason = f"NDCG@10 difference {interval}: the lower bound is not above 0."
    return {"rule": RULE, "decision": "promote" if promote else "keep", "promote": promote, "reason": reason}


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def connect_storage() -> Storage:
    try:
        return Storage.from_env()
    except RuntimeError as error:
        raise SystemExit(f"{error} Or read the raw data from disk with --source local:<directory>.")


def describe_destinations(remote: bool, tracking_uri: str, storage: Optional[Storage], cutoff: str) -> str:
    """What this run will write to, shown before anything is computed."""
    if remote:
        return (f"Writing to (--remote):\n"
                f"  MLflow: {tracking_uri}\n"
                f"  R2    : s3://{storage.bucket}/{PROCESSED_PREFIX}/{cutoff}/")
    return (f"Writing locally only (pass --remote to log to DagsHub and upload to R2):\n"
            f"  MLflow: {tracking_uri}\n"
            f"  R2    : no upload")


def run_params(model_type: str, train_config: TrainConfig, cutoff: str, trained_before: str,
               stats: Dict[str, Any]) -> Dict[str, Any]:
    return {**model_params(model_type, train_config), "cutoff": cutoff, "trained_before": trained_before,
            "train_sha256": stats["sha256"]["train.parquet"], "git_commit": git_state()["git_commit"]}


def window(first_timestamp: int, end_timestamp: int) -> Dict[str, str]:
    return {"from": iso(first_timestamp), "to_exclusive": iso(end_timestamp)}


def run_pipeline(cutoff: str, source: str = "r2", remote: bool = False) -> Dict[str, Any]:
    tracking_uri = tracking.setup(remote=remote)
    train_config = load_train_config()
    config: DataConfig = load_data_config(cutoff=cutoff)
    # R2 is only contacted to read the raw data (source "r2") or, with --remote, to upload.
    storage = connect_storage() if (remote or source == "r2") else None
    print(describe_destinations(remote, tracking_uri, storage, cutoff))

    # ---- 1. Prepare the data for this cutoff.
    print(f"[1/5] Preparing cutoff {cutoff} from {source}")
    out_dir, stats = run_prepare(config, source, storage=storage)
    stats_path = os.path.join(out_dir, STATS_FILE)
    uploaded = None
    if remote:
        try:
            uploaded = upload_directory(storage, out_dir, f"{PROCESSED_PREFIX}/{cutoff}", DATA_FILES)
        except RemoteConflictError as error:
            raise SystemExit(f"Upload stopped: {error}")
    movies = read_parquet(out_dir, "movies.parquet")

    # ---- 2. Gate window: the last GATE_DAYS days before the cutoff, carved out of train.
    gate_train, gate_eval, gate_config = inner_split(read_parquet(out_dir, "train.parquet"), movies, config, GATE_DAYS)
    if gate_train["timestamp"].max() >= gate_config.cutoff_timestamp:
        raise AssertionError("gate training data reaches into the gate window")
    print(f"[2/5] Gate window {gate_config.cutoff} up to {cutoff}: {len(gate_eval):,} liked ratings of "
          f"{gate_eval['userId'].nunique():,} users; models fitted on ratings before {gate_config.cutoff}")

    champion = tracking.get_champion()
    initial_champion = None
    if champion is None:
        print("      No champion in the registry: registering popularity as the initial champion")
        initial = fit_model(POPULARITY, gate_train, gate_config, train_config)
        logged = tracking.log_run(
            f"initial-champion-{POPULARITY}", initial,
            run_params(POPULARITY, train_config, cutoff, gate_config.cutoff, stats),
            {"window": "gate", "role": "initial-champion", "cutoff": cutoff}, stats_path, register=True,
            version_tags={"cutoff": cutoff, "trained_before": gate_config.cutoff})
        tracking.set_champion(logged.version)
        champion = tracking.get_champion()
        initial_champion = {"registry_version": logged.version, "run_id": logged.run_id}
    champion_type = champion["model_type"]

    gate_models: Dict[str, Recommender] = {
        "challenger": fit_model(BLEND, gate_train, gate_config, train_config),
        "champion": fit_model(champion_type, gate_train, gate_config, train_config),
    }
    gate = evaluate.evaluate_window(gate_models, gate_eval)
    evaluate.check_plausible(gate.results)
    difference = gate.difference("challenger", "champion")
    decision = gate_decision(difference, BLEND, champion_type)
    for name, result in gate.results.items():
        print(evaluate.format_result(name, result))
    print(evaluate.format_difference("challenger - champion", difference))
    print(f"[3/5] Decision: {decision['decision']}. {decision['reason']}")

    gate_runs = {}
    for name, model_type in (("challenger", BLEND), ("champion", champion_type)):
        gate_runs[name] = tracking.log_run(
            f"gate-{name}-{model_type}", gate_models[name],
            run_params(model_type, train_config, cutoff, gate_config.cutoff, stats),
            {"window": "gate", "role": name, "cutoff": cutoff, "decision": decision["decision"]},
            stats_path, result=gate.results[name]).run_id

    write_json(os.path.join(report_dir(cutoff), GATE_FILE), {
        "cutoff": cutoff,
        "gate_window_utc": window(gate_config.cutoff_timestamp, config.cutoff_timestamp),
        "models_trained_on": f"ratings before {gate_config.cutoff}",
        "evaluated_users": int(len(gate.users)),
        "liked_ratings": int(len(gate_eval)),
        "challenger": {"model_type": BLEND, **model_params(BLEND, train_config), **gate.results["challenger"]},
        "champion": {"model_type": champion_type, "registry_version": champion["version"],
                     **model_params(champion_type, train_config), **gate.results["champion"]},
        "difference_challenger_minus_champion": difference,
        **decision,
        "initial_champion_registered": initial_champion,
        "mlflow": {"tracking": "local" if tracking_uri == tracking.LOCAL_TRACKING_URI else tracking_uri,
                   "experiment": tracking.EXPERIMENT, "run_ids": gate_runs},
        "train_sha256": stats["sha256"]["train.parquet"],
        **git_state(),
    })
    del gate_train, gate_models, gate

    # ---- 4. Refit the winner on everything before the cutoff and make it the champion.
    winner_type = BLEND if decision["promote"] else champion_type
    train = read_parquet(out_dir, "train.parquet")
    production = read_parquet(out_dir, "test.parquet")
    if len(train) and train["timestamp"].max() >= config.cutoff_timestamp:
        raise AssertionError("training data reaches the cutoff")
    if len(production) and production["timestamp"].min() < config.cutoff_timestamp:
        raise AssertionError("the production window starts before the cutoff")
    print(f"[4/5] Refitting {winner_type} on ratings before {cutoff}")
    winner = fit_model(winner_type, train, config, train_config)
    popularity = winner if winner_type == POPULARITY else fit_model(POPULARITY, train, config, train_config)

    # ---- 5. Production window: the month after the cutoff, untouched by any decision.
    scored = evaluate.evaluate_window({"champion": winner, "popularity": popularity}, production)
    evaluate.check_plausible(scored.results)
    production_difference = scored.difference("champion", "popularity")
    print(f"[5/5] Production window {cutoff} + {config.test_window_days} days: {len(production):,} liked ratings "
          f"of {len(scored.users):,} users")
    for name, result in scored.results.items():
        print(evaluate.format_result(name, result))
    print(evaluate.format_difference("champion - popularity", production_difference))

    registered = tracking.log_run(
        f"production-champion-{winner_type}", winner, run_params(winner_type, train_config, cutoff, cutoff, stats),
        {"window": "production", "role": "champion", "cutoff": cutoff, "decision": decision["decision"]},
        stats_path, result=scored.results["champion"], register=True,
        version_tags={"cutoff": cutoff, "trained_before": cutoff, "gate_decision": decision["decision"]})
    mechanism = tracking.set_champion(registered.version)
    production_runs = {"champion": registered.run_id}
    if winner_type != POPULARITY:
        production_runs["popularity"] = tracking.log_run(
            f"production-baseline-{POPULARITY}", popularity,
            run_params(POPULARITY, train_config, cutoff, cutoff, stats),
            {"window": "production", "role": "baseline", "cutoff": cutoff}, stats_path,
            result=scored.results["popularity"]).run_id

    write_json(os.path.join(report_dir(cutoff), PRODUCTION_FILE), {
        "cutoff": cutoff,
        "production_window_utc": window(config.cutoff_timestamp, config.test_end_timestamp),
        "models_trained_on": f"ratings before {cutoff}",
        "evaluated_users": int(len(scored.users)),
        "liked_ratings": int(len(production)),
        "champion": {"model_type": winner_type, "registry_model": tracking.REGISTERED_MODEL,
                     "registry_version": registered.version, "marked_as_champion_by": mechanism,
                     **model_params(winner_type, train_config), **scored.results["champion"]},
        "popularity": scored.results["popularity"],
        "difference_champion_minus_popularity": production_difference,
        "mlflow": {"tracking": "local" if tracking_uri == tracking.LOCAL_TRACKING_URI else tracking_uri,
                   "experiment": tracking.EXPERIMENT, "run_ids": production_runs},
        "r2_upload": uploaded,
        "train_sha256": stats["sha256"]["train.parquet"],
        **git_state(),
    })
    print(f"Champion: {tracking.REGISTERED_MODEL} version {registered.version} ({winner_type}), marked by {mechanism}")
    print(f"Saved {os.path.join(report_dir(cutoff), GATE_FILE)} and {PRODUCTION_FILE}")
    return {"decision": decision, "champion_version": registered.version, "champion_type": winner_type}


def main():
    parser = argparse.ArgumentParser(description="Prepare, gate, refit and evaluate for one cutoff.")
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--source", default="r2", help="'r2' (default) or 'local:<directory with ratings.csv and movies.csv>'")
    parser.add_argument("--remote", action="store_true",
                        help="log to MLflow on DagsHub and upload processed/<cutoff>/ to R2 (default: local only)")
    args = parser.parse_args()
    run_pipeline(args.cutoff, args.source, args.remote)


if __name__ == "__main__":
    main()
