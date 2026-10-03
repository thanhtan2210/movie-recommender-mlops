"""Latency benchmark of a text query on the real database.

Each of the 200 queries in scripts/benchmark_queries.txt goes through the
same steps as SemanticSearchEngine.search_by_description: embedding, LanceDB
search, rerank, output validation. Each step is timed separately. The cold
start (download, unpack, model load, table load) is timed once.

The LLM call to Groq is not measured: it depends on the network and quota.

Run from the repo root:  python -m scripts.benchmark
Writes reports/latency.json.
"""
import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from importlib import metadata
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from src.serving import storage
from src.serving.semantic_search import Reranker, SemanticSearchEngine

REPORT_DIR = "reports"
LATENCY_PATH = os.path.join(REPORT_DIR, "latency.json")
QUERIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_queries.txt")
TOP_K = 10
WARMUP_QUERIES = 5
STEPS = ["embedding", "search", "rerank", "validation", "total"]


def load_queries(path: str = QUERIES_PATH) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def timed_query(engine: SemanticSearchEngine, query: str, top_k: int = TOP_K):
    """Run one query step by step; returns (results, seconds per step).

    Mirrors search_by_description with the reranker on. A test checks that
    both return the same results.
    """
    t0 = time.perf_counter()
    vector = engine.embed_query(query)
    t1 = time.perf_counter()
    candidates = engine.retrieve_candidates(vector, top_k * 2)
    t2 = time.perf_counter()
    ranked = Reranker.rerank(candidates)[:top_k]
    t3 = time.perf_counter()
    results = engine._validate_candidates(ranked, "rerank")
    t4 = time.perf_counter()
    return results, {"embedding": t1 - t0, "search": t2 - t1, "rerank": t3 - t2,
                     "validation": t4 - t3, "total": t4 - t0}


def summarise(seconds: List[float]) -> Dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean())}


def run_queries(engine: SemanticSearchEngine, queries: List[str], warmup: int = WARMUP_QUERIES) -> Dict[str, Any]:
    for query in queries[:warmup]:
        timed_query(engine, query)
    timings: Dict[str, List[float]] = {step: [] for step in STEPS}
    for query in queries:
        _, seconds = timed_query(engine, query)
        for step in STEPS:
            timings[step].append(seconds[step])
    return {step: summarise(values) for step, values in timings.items()}


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


def version_of(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def machine_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "python": platform.python_version(),
        "device": "cpu",
        "versions": {name: version_of(name) for name in ("lancedb", "sentence-transformers", "torch", "pandas")},
    }


def cold_start(db_path: str) -> Dict[str, Any]:
    """Time what a cold start does, once. Returns the timings and a ready engine."""
    timings: Dict[str, Any] = {"download_s": None, "unpack_s": None}
    if storage.has_r2_credentials():
        # Download into a scratch directory so the measurement does not depend
        # on (or disturb) a database that is already on disk.
        workdir = tempfile.mkdtemp(prefix="coldstart_")
        zip_path = os.path.join(workdir, storage.DB_ZIP)
        db_path = os.path.join(workdir, storage.DB_PATH)
        t0 = time.perf_counter()
        storage.download_object(storage.DB_ZIP, zip_path)
        t1 = time.perf_counter()
        shutil.unpack_archive(zip_path, db_path)
        t2 = time.perf_counter()
        timings.update(download_s=t1 - t0, unpack_s=t2 - t1, zip_size_mb=os.path.getsize(zip_path) / 1024 ** 2)
    else:
        timings["note"] = "no R2 credentials: download and unpack were not measured"

    engine = SemanticSearchEngine(lancedb_uri=db_path)
    t0 = time.perf_counter()
    engine.load_model()
    t1 = time.perf_counter()
    engine.load_table()
    t2 = time.perf_counter()
    timings.update(load_model_s=t1 - t0, load_table_s=t2 - t1)
    measured = [timings[k] for k in ("download_s", "unpack_s", "load_model_s", "load_table_s") if timings[k] is not None]
    timings["total_measured_s"] = float(sum(measured))
    return {"timings": timings, "engine": engine}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=storage.DB_PATH, help="LanceDB directory used when R2 is not configured")
    args = parser.parse_args()
    load_dotenv()

    if not storage.has_r2_credentials() and not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found and no R2 credentials are set: nothing to benchmark.")

    queries = load_queries()
    started = cold_start(args.db)
    engine = started["engine"]
    print(f"Cold start: {json.dumps(started['timings'], indent=2)}")

    print(f"Running {len(queries)} queries ({WARMUP_QUERIES} warm-up queries first)...")
    latency = run_queries(engine, queries)
    for step in STEPS:
        s = latency[step]
        print(f"  {step:<10} p50 {s['p50_ms']:8.2f} ms | p95 {s['p95_ms']:8.2f} ms | mean {s['mean_ms']:8.2f} ms")

    report = {
        "database": {"movies": int(len(engine.catalog)), "vector_dimension": int(engine._vectors.shape[1])},
        "queries": len(queries),
        "warmup_queries": WARMUP_QUERIES,
        "top_k": TOP_K,
        "candidates": TOP_K * 2,
        "not_measured": "the Groq LLM call (network and quota dependent)",
        "latency": latency,
        "cold_start": started["timings"],
        "machine": machine_info(),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(LATENCY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved {LATENCY_PATH}")


if __name__ == "__main__":
    main()
