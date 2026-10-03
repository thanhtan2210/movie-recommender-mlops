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
python -m src.diagnose --cutoff 2019-06-01           # diagnostics by release year
python -m src.blend --cutoff 2019-06-01              # choose the popularity blend on validation, score on test
```

`--source local:<directory>` reads `ratings.csv` and `movies.csv` from disk instead of R2. R2 credentials go in a `.env` file (see [.env.example](.env.example)).

Output: `train.parquet`, `test.parquet`, `movies.parquet` and `stats.json` (row, user and movie counts, rows removed by each test filter, sha256 of each parquet file, git commit). Running the same cutoff again produces byte-identical parquet files. An upload never overwrites: files with the same sha256 are skipped, and a different file under the same cutoff stops the upload.

## Results

Cutoff 2019-06-01. Train: ratings before the cutoff. Test: the 9,727 liked ratings of 1,407 users in June 2019. Intervals are 95% bootstrap intervals over users (1,000 resamples); differences are paired (the same resampled users for both recommenders). All files are in [reports/2019-06-01/](reports/2019-06-01/).

**The recommenders**

- **PureSVD**: a truncated SVD of the binary user × movie matrix (1 where the rating is at least 4.0). Only the movie factors are stored; a user is scored from the movies they liked. Movies a user already rated are never recommended.
- **Popularity** (baseline): the movies with the most liked ratings in the last 90 days of training. No parameter to tune.
- **PureSVD + recent popularity**: `score = z(SVD score of the user) + weight × z(log(1 + likes in the last 90 days))`, with the SVD fitted on a recent window of training ratings (`train_window`).

**Test set** - `python -m src.blend`, [test_metrics_v2.json](reports/2019-06-01/test_metrics_v2.json):

| | PureSVD (k = 64) | PureSVD + recent popularity (1 year, weight 4) | Popularity | Blend − Popularity (paired) |
| --- | --- | --- | --- | --- |
| HitRate@10 | 21.6% [19.6, 23.7] | **27.2%** [24.7, 29.4] | 22.8% [20.5, 25.0] | +4.4 pt [+2.0, +6.5] |
| Recall@10 | 7.4% [6.5, 8.3] | **11.3%** [10.1, 12.5] | 9.1% [7.9, 10.2] | +2.3 pt [+1.1, +3.5] |
| NDCG@10 | 0.0596 [0.0524, 0.0662] | **0.0846** [0.0749, 0.0937] | 0.0674 [0.0589, 0.0751] | +0.0172 [+0.0084, +0.0260] |
| Catalogue coverage | 7.1% (920 movies) | 3.3% (430 movies) | 1.5% (195 movies) | +1.8 pt |
| Long-tail share | 0.2% [0.1, 0.3] | 28.2% [26.6, 29.7] | 17.7% [17.2, 18.2] | +10.5 pt [+9.0, +12.1] |

**How to read this table.** The blend was selected on the May validation window; the June test window has now been looked at twice (first for plain PureSVD, then for the blend). The blend's test numbers are therefore optimistic to an unknown degree. An independent evaluation will use July-October 2019, which no decision has touched.

### How the model was chosen

1. **k** - `python -m src.train`, [validation.csv](reports/2019-06-01/validation.csv). On the validation window (liked ratings in May 2019, model fitted on ratings before 2019-05-01, 1,489 users) k = 64 had the highest NDCG@10 (0.0624), with intervals overlapping those of k = 32, 128 and 256: k matters little.

2. **First test evaluation** - `python -m src.evaluate`, [test_metrics.json](reports/2019-06-01/test_metrics.json). Plain PureSVD did not beat popularity: NDCG@10 0.0596 against 0.0674, Recall@10 lower by 1.7 points [−3.0, −0.3].

3. **Diagnostics** - `python -m src.diagnose`, [diagnostics.json](reports/2019-06-01/diagnostics.json). PureSVD has no notion of time, and what users like in the test month leans towards recent releases:

   | | Released 2018 or later | Median release year |
   | --- | --- | --- |
   | Liked ratings in train, all time | 0.2% | 1996 |
   | Liked ratings in train, last 90 days | 5.5% | 2004 |
   | Movies liked in the test window | 12.2% | 2007 |
   | PureSVD top-10 | 0.1% | 2002 |
   | Popularity top-10 | 22.7% | 2006 |

   | Target movies | Users | PureSVD HitRate@10 | Popularity HitRate@10 |
   | --- | --- | --- | --- |
   | Released 2018 or later | 614 | 0.3% [0.0, 0.8] | 28.5% [24.9, 31.8] |
   | Released before 2018 | 1,220 | 24.8% [22.5, 27.3] | 14.0% [12.1, 16.0] |

   PureSVD is the better recommender for older movies and almost never finds a new one. The popularity baseline's long-tail share (17.7%) also comes from new releases, which have few ratings in total: without movies from 2018 onwards it is 0.3%.

4. **Blend** - `python -m src.blend`, [validation_blend.csv](reports/2019-06-01/validation_blend.csv). Grid of `train_window` ∈ {all, 3 years, 1 year} × weight ∈ {0, 0.25, 0.5, 1, 2, 4} with k = 64, scored on the same validation window. Best weight per window:

   | train_window | weight | HitRate@10 | Recall@10 | NDCG@10 | Coverage |
   | --- | --- | --- | --- | --- | --- |
   | all | 4 | 23.2% | 8.3% | 0.0673 [0.0594, 0.0758] | 3.1% |
   | 3 years | 4 | 24.7% | 8.9% | 0.0729 [0.0648, 0.0816] | 3.4% |
   | **1 year** | **4** | 26.5% | 10.1% | **0.0791** [0.0704, 0.0881] | 3.4% |
   | Popularity | - | 22.1% | 8.4% | 0.0667 [0.0585, 0.0745] | 1.5% |

   The selected configuration beats popularity on validation by 0.0124 NDCG@10 [+0.0039, +0.0214]. Two things to note. The training window does most of the work: with weight 0, NDCG@10 is 0.0624 for all of train, 0.0661 for 3 years and 0.0682 for 1 year. And the best weight is the largest one in the grid for every window, so the optimum may lie beyond it; no weight outside the grid was tried.

### Accuracy against variety

Test set, `train_window` = 1 year - [tradeoff.csv](reports/2019-06-01/tradeoff.csv). Descriptive only: the weight was chosen on validation, not from this table.

| Weight | NDCG@10 | HitRate@10 | Catalogue coverage |
| --- | --- | --- | --- |
| 0 | 0.0741 | 24.0% | 6.7% |
| 0.25 | 0.0803 | 26.5% | 5.1% |
| 0.5 | 0.0812 | 26.9% | 4.9% |
| 1 | 0.0814 | 27.0% | 4.5% |
| 2 | 0.0824 | 27.1% | 4.0% |
| 4 | 0.0846 | 27.2% | 3.3% |

More weight on recent popularity buys a little accuracy and costs variety: from weight 0 to 4 the number of distinct movies recommended is halved.

### Limits of this evaluation

- The June test window has been used twice (see above).
- One cutoff and 30-day windows; 1,407 test users, all with at least 5 liked movies before the cutoff. New users are not evaluated.
- With a 1-year window the SVD is fitted on 673,894 liked ratings out of 12.1 million, and 70 of the 1,407 test users have no liked rating inside the window; they receive the popularity ranking.
- Offline metrics on ratings, which are not viewing behaviour and do not replace an online test.

Three example users with their recent likes and the PureSVD and popularity top-10 lists: [examples.json](reports/2019-06-01/examples.json).
