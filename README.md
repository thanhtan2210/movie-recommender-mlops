# Movie Recommender MLOps Pipeline

Recommend movies each user is likely to enjoy next, with a model that is retrained, evaluated and promoted as new ratings arrive.

> **Rebuild in progress** — the previous RAG version lives on branch `archive/before-cleanup`.

Planned pipeline:

```text
MovieLens (R2: raw/) → prepare → R2: processed/<cutoff>/ → train PureSVD → MLflow on DagsHub (metrics + registry + promotion gate) → FastAPI + Docker → monthly retraining
```

Built so far: the `prepare` step. Training, tracking, serving and retraining do not exist yet.

## Data preparation

`src/prepare.py` turns the raw MovieLens ratings into a training set and a test set for one cutoff date:

- **train**: every rating before the cutoff, for movies with at least `min_item_ratings` ratings in the training set;
- **test**: liked ratings (`rating >= positive_threshold`) in the `test_window_days` after the cutoff, for users with at least `min_user_train_positives` liked movies in train and for movies in the training catalogue.

Every filter is computed on the training set only, so nothing from the test window leaks into what the model sees. The cutoff is midnight UTC. Parameters are in [configs/data.yaml](configs/data.yaml).

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.prepare --cutoff 2019-06-01            # reads raw/ from R2, writes data/processed/2019-06-01/
python -m src.prepare --cutoff 2019-06-01 --upload   # also uploads to R2 at processed/2019-06-01/
```

`--source local:<directory>` reads `ratings.csv` and `movies.csv` from disk instead of R2. R2 credentials go in a `.env` file (see [.env.example](.env.example)).

Output: `train.parquet`, `test.parquet`, `movies.parquet` and `stats.json` (row, user and movie counts, rows removed by each test filter, sha256 of each parquet file, git commit). Running the same cutoff again produces byte-identical parquet files. An upload never overwrites: files with the same sha256 are skipped, and a different file under the same cutoff stops the upload.
