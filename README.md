# Movie Recommender MLOps Pipeline

Recommend movies each user is likely to enjoy next, with a model that is retrained, evaluated and promoted as new ratings arrive.

> **Rebuild in progress** — the previous RAG version lives on branch `archive/before-cleanup`.

Planned pipeline:

```text
MovieLens (R2: raw/) → prepare → R2: processed/<cutoff>/ → train PureSVD → MLflow on DagsHub (metrics + registry + promotion gate) → FastAPI + Docker → monthly retraining
```
