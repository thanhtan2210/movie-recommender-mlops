"""Client-side latency of the running API (start the container first).

  docker run -p 8000:8000 movie-rec
  python -m scripts.benchmark_api

After 20 warm-up requests that are not counted, sends GET /recommend/{user_id}
to two groups of real users drawn at random (seed 42), 200 each:
  - users with at least one liked rating in the model's training window,
    who get `personalized` recommendations;
  - the other users of the training data, who get `popularity_fallback`;
then 50 POST /recommend with 5-20 random catalogue movies each.
Latency is reported per group. Writes reports/api_latency.json.

The default URL uses 127.0.0.1, not localhost: on Docker Desktop for Windows,
a POST sent to `localhost` goes through the IPv6 loopback proxy and takes
about 50 ms longer, which has nothing to do with the API. --compare-localhost
measures the POST requests through `localhost` as well and stores both.
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from src.export_champion import SERVING_DIR, state_path

REPORT_PATH = os.path.join("reports", "api_latency.json")
SEED = 42
WARMUP_REQUESTS = 20
GET_REQUESTS = 200
POST_REQUESTS = 50
TOP_N = 10


def summarise(seconds: List[float]) -> Dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {"requests": int(len(ms)), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean())}


def timed(client: httpx.Client, method: str, url: str, strategies: Optional[Dict[str, int]] = None, **kwargs) -> float:
    """Seconds taken by one request; the strategy of the answer is tallied in `strategies`."""
    started = time.perf_counter()
    response = client.request(method, url, **kwargs)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    if strategies is not None:
        strategy = response.json()["strategy"]
        strategies[strategy] = strategies.get(strategy, 0) + 1
    return elapsed


def total_ram_gb() -> Optional[float]:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def machine_info() -> Dict[str, Any]:
    return {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "ram_gb": total_ram_gb(), "python": platform.python_version()}


def split_users(user_ids: np.ndarray, liked_indptr: np.ndarray) -> Dict[str, np.ndarray]:
    """Users with at least one liked rating inside the training window, and the others."""
    has_likes = np.diff(liked_indptr) > 0
    return {"personalized": user_ids[has_likes], "popularity_fallback": user_ids[~has_likes]}


def run(url: str, model_dir: str, server: str, compare_localhost: bool = False) -> Dict[str, Any]:
    with np.load(state_path(model_dir)) as state:
        user_ids, movie_ids = state["user_ids"], state["item_ids"]
        groups = split_users(user_ids, state["liked_indptr"])
    rng = np.random.default_rng(SEED)
    warmup_users = rng.choice(user_ids, size=min(WARMUP_REQUESTS, len(user_ids)), replace=False).tolist()
    # A group smaller than GET_REQUESTS (as in a tiny test model) is measured with all its users.
    sampled = {name: rng.choice(ids, size=min(GET_REQUESTS, len(ids)), replace=False).tolist()
               for name, ids in groups.items()}
    bodies = [{"liked_movie_ids": rng.choice(movie_ids, size=int(rng.integers(5, 21)), replace=False).tolist(),
               "n": TOP_N} for _ in range(POST_REQUESTS)]

    get_by_group: Dict[str, Optional[Dict[str, Any]]] = {}
    post_strategies: Dict[str, int] = {}
    with httpx.Client(base_url=url, timeout=30.0) as client:
        health = client.get("/health").raise_for_status().json()
        for user in warmup_users:
            timed(client, "GET", f"/recommend/{user}", params={"n": TOP_N})
        for name, users in sampled.items():
            strategies: Dict[str, int] = {}
            seconds = [timed(client, "GET", f"/recommend/{user}", strategies, params={"n": TOP_N}) for user in users]
            # `strategies` records what the API answered, which should be the group's name.
            get_by_group[name] = {**summarise(seconds), "strategies": strategies} if users else None
        post_seconds = [timed(client, "POST", "/recommend", post_strategies, json=body) for body in bodies]

    via_localhost = None
    if compare_localhost:
        with httpx.Client(base_url=url.replace("127.0.0.1", "localhost"), timeout=30.0) as client:
            for body in bodies[:5]:
                timed(client, "POST", "/recommend", json=body)
            via_localhost = summarise([timed(client, "POST", "/recommend", json=body) for body in bodies])

    return {
        "url": url,
        "server": server,
        "model": {"version": health["model_version"], "cutoff": health["cutoff"],
                  "users": int(len(user_ids)),
                  "users_with_liked_ratings_in_window": int(len(groups["personalized"])),
                  "share_of_users_with_liked_ratings_in_window": float(len(groups["personalized"]) / len(user_ids)),
                  "movies": int(len(movie_ids))},
        "protocol": {"seed": SEED, "warmup_requests_not_counted": WARMUP_REQUESTS, "n": TOP_N,
                     "sequential": True, "measured": "client side, one keep-alive connection"},
        "get_recommend_user": get_by_group,
        "post_recommend": {**summarise(post_seconds), "strategies": post_strategies},
        "post_recommend_via_localhost": via_localhost,
        "machine": machine_info(),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure the API's latency from the client side.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", default=SERVING_DIR, help="where the served model's user and movie ids are read from")
    parser.add_argument("--server", default="Docker container on the same machine",
                        help="a short description of where the API runs, stored in the report")
    parser.add_argument("--compare-localhost", action="store_true",
                        help="also measure POST through the host name 'localhost' (see the module docstring)")
    args = parser.parse_args()

    report = run(args.url, args.model_dir, args.server, args.compare_localhost)
    rows = {f"get_recommend_user ({group})": stats for group, stats in report["get_recommend_user"].items()}
    rows.update({name: report[name] for name in ("post_recommend", "post_recommend_via_localhost")})
    for name, s in rows.items():
        if s is None:
            continue
        print(f"  {name:<45} n={s['requests']:>3}  p50 {s['p50_ms']:7.2f} ms | p95 {s['p95_ms']:7.2f} ms | "
              f"p99 {s['p99_ms']:7.2f} ms")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
