# Movie Recommender MLOps Pipeline

Recommend movies each user is likely to enjoy next, with a model that is retrained, evaluated and promoted as new ratings arrive.

> **Rebuild in progress** — the previous RAG version lives on branch `archive/before-cleanup`.

Planned pipeline:

```text
MovieLens (R2: raw/) → prepare → R2: processed/<cutoff>/ → train PureSVD → MLflow on DagsHub (metrics + registry + promotion gate) → FastAPI + Docker → monthly retraining
```

Built so far: `prepare`, `train` and the offline evaluation. Experiment tracking, serving and retraining do not exist yet.

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
python -m src.train --cutoff 2019-06-01              # choose k on validation, fit, save artifacts/2019-06-01/
python -m src.evaluate --cutoff 2019-06-01           # test metrics -> reports/2019-06-01/
```

`--source local:<directory>` reads `ratings.csv` and `movies.csv` from disk instead of R2. R2 credentials go in a `.env` file (see [.env.example](.env.example)).

Output: `train.parquet`, `test.parquet`, `movies.parquet` and `stats.json` (row, user and movie counts, rows removed by each test filter, sha256 of each parquet file, git commit). Running the same cutoff again produces byte-identical parquet files. An upload never overwrites: files with the same sha256 are skipped, and a different file under the same cutoff stops the upload.

## Model and offline evaluation

**Model.** PureSVD: a truncated SVD of the binary user × movie matrix (1 where the rating is at least 4.0). Only the movie factors are stored; a user is scored from the movies they liked, and movies they already rated are never recommended.

**Baseline.** Popularity: the movies with the most liked ratings in the last 90 days of training, minus the movies the user already rated. It has no parameter to tune.

**Protocol.** Cutoff 2019-06-01. The number of factors k is chosen on a validation window inside train (liked ratings from 2019-05-01 to 2019-05-31, model fit on ratings before 2019-05-01), then the model is refit on all of train and evaluated on the test window (2019-06-01 to 2019-06-30). Intervals are 95% bootstrap intervals over users (1,000 resamples).

Choosing k on validation (1,489 users) - `python -m src.train`, [reports/2019-06-01/validation.csv](reports/2019-06-01/validation.csv):

| k | HitRate@10 | Recall@10 | NDCG@10 | Catalogue coverage |
| --- | --- | --- | --- | --- |
| 32 | 20.3% | 6.9% | 0.0568 [0.0494, 0.0642] | 6.2% |
| **64** | 22.0% | 7.3% | **0.0624** [0.0545, 0.0701] | 7.2% |
| 128 | 21.7% | 7.2% | 0.0609 [0.0535, 0.0689] | 8.2% |
| 256 | 20.8% | 6.6% | 0.0558 [0.0485, 0.0630] | 9.4% |

k = 64 has the highest NDCG@10, but the intervals of all four values overlap: the choice of k matters little here.

Test set (1,407 users, 9,727 liked ratings) - `python -m src.evaluate`, [reports/2019-06-01/test_metrics.json](reports/2019-06-01/test_metrics.json):

| | PureSVD (k = 64) | Popularity | PureSVD − Popularity (paired) |
| --- | --- | --- | --- |
| HitRate@10 | 21.6% [19.6, 23.7] | 22.8% [20.5, 25.0] | −1.2 pt [−3.9, +1.4] |
| Recall@10 | 7.4% [6.5, 8.3] | 9.1% [7.9, 10.2] | −1.7 pt [−3.0, −0.3] |
| NDCG@10 | 0.0596 [0.0524, 0.0662] | 0.0674 [0.0589, 0.0751] | −0.0078 [−0.0175, +0.0016] |
| Catalogue coverage | 7.1% (920 movies) | 1.5% (195 movies) | +5.6 pt |
| Long-tail share | 0.2% [0.1, 0.3] | 17.7% [17.2, 18.2] | −17.5 pt [−18.1, −17.0] |

What this says:

- **PureSVD does not beat the popularity baseline.** It is lower on HitRate@10 and NDCG@10 by a margin that is not distinguishable from zero, and lower on Recall@10 by a small but real margin.
- **It recommends a wider set of movies**: 920 distinct movies across the test users against 195.
- **It almost never leaves the most rated movies.** The long tail is defined as everything outside the 20% of movies with the most ratings in train. PureSVD's recommendations are 0.2% long tail; the popularity baseline's are 17.7%, because it ranks by the last 90 days and so surfaces recent releases that have not yet accumulated many ratings.

Three example users, with their recent likes and both top-10 lists, are in [reports/2019-06-01/examples.json](reports/2019-06-01/examples.json).

Limits of this evaluation: one cutoff and one 30-day window; 1,407 users, all of whom already had at least 5 liked movies (new users are not evaluated); offline metrics on ratings, which are not viewing behaviour and do not replace an online test.
