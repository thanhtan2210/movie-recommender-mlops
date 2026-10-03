# Movie Recommender with Vector Search & RAG

"I liked these movies (or: this is the kind of movie I feel like watching). What should I watch next?"

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/thanhtanphan/ai-movie-resys)

## Problem → Approach → Results

**Problem.** A viewer wants ten movies to watch next, starting from a few movies they liked or from a short description.

**Approach.** Each movie in MovieLens 25M with at least 50 ratings is turned into one text (title, genres, overview, tags, rating) and embedded with a sentence-embedding model. The app builds a query vector from the liked movies or from the description, retrieves the nearest movies from LanceDB, reranks them by similarity, popularity and quality, and shows the top 10. A chat tab lets an LLM explain the picks using the same search as a tool.

**Results.** Not reported yet. `python -m scripts.evaluate_offline` (HitRate@10, NDCG@10, coverage and long-tail share against a popularity baseline) and `python -m scripts.benchmark` (latency per step) write their results to `reports/`; they have not been run on the real database in this branch. Earlier versions reported Recall@10 = 88.56% and NDCG@10 = 0.81; no code in the repo computed those numbers, so they have been removed.

## How it works

```
liked movies or a description → embedding → LanceDB search → rerank → top 10 → (optional) LLM explanation
```

| Component | Role |
| --- | --- |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | 384-d embeddings of movies and of text queries |
| LanceDB | Embedded vector database, cosine search |
| Groq (`llama-3.3-70b-versatile`) | Chat with three tools: search by description, similar movies by title, highly rated movies |
| Streamlit | App with three tabs: Recommend, Chat, Evaluation |
| Cloudflare R2 | Stores the database artifact `lancedb_movies.zip` |
| Hugging Face Spaces | Hosts the app; downloads the database on a cold start |

The vector database was built once on Google Colab (`notebooks/colab_pipeline.ipynb`). Apache Spark was used once on Colab to aggregate 25M ratings into a per-movie average and count; it is not part of the running app.

Details: [docs/architecture.md](docs/architecture.md). Problems met along the way: [docs/troubleshooting.md](docs/troubleshooting.md).

## How to run

Python 3.10 or 3.11.

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q
```

The app reads these variables from the environment or a `.env` file:

```env
AWS_ACCESS_KEY_ID=...                  # Cloudflare R2
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
S3_BUCKET_NAME=movie-mlops
GROQ_API_KEY=...                       # only needed for the Chat tab
```

On its first run the app downloads `lancedb_movies.zip` from the R2 bucket and unpacks it into `lancedb_movies/`. The tests need neither network access nor keys.

## Limitations

- **Content-based only.** Recommendations come from text similarity between movies; the app does not learn from what other users with similar taste watched.
- **The quality signal is counted twice.** The average rating and vote count are written into the embedded text and are also used by the reranker.
- **MovieLens ratings are not viewing behaviour.** They record what people chose to rate, often long after watching.
- **The database is a fixed snapshot** of MovieLens 25M built once; new movies are not added.
- **No measured accuracy or latency yet** (see Results above).

## Repository notes

- `_archive/` holds earlier experiments that are not used by the app.
- `docs/report.pdf` is the original course report. It describes earlier versions of the system, and its metrics are superseded by this README.

---
*Developed as a Graduation Project Report focusing on Big Data and MLOps.*
