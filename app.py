import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.serving import storage
from src.serving.chatbot import MovieChatbot, set_search_engine
from src.serving.semantic_search import SemanticSearchEngine

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

CHART_COLOR = "#636EFA"
TOP_K = 10
REPORT_DIR = "reports"


# ==========================================
# 1. SERVICES
# ==========================================


@st.cache_resource
def init_lancedb():
    """Download the vector database from Cloudflare R2 (first run only)."""
    if not os.path.exists(storage.DB_PATH):
        if not storage.has_r2_credentials():
            st.warning("R2 environment variables are missing. Check your .env file.")
            return None

        try:
            storage.download_database()
        except Exception as e:
            st.error(f"Could not download the database from R2: {e}")
            return None

    engine = SemanticSearchEngine(lancedb_uri=storage.DB_PATH)
    engine.load_table()
    engine.load_model()
    return engine


@st.cache_resource
def init_chatbot():
    """The chat tab needs a Groq key; the rest of the app works without it."""
    try:
        return MovieChatbot()
    except ValueError:
        return None


engine = init_lancedb()
if engine is not None:
    # The chatbot tools use the same engine, so the table and the model are loaded once.
    set_search_engine(engine)
chatbot = init_chatbot()


# ==========================================
# 2. RECOMMEND
# ==========================================


def display_recommendations(recommendations):
    for start in range(0, len(recommendations), 5):
        cols = st.columns(5)
        for col, rec in zip(cols, recommendations[start:start + 5]):
            with col:
                poster = rec.get('poster_path', '')
                if poster and poster not in ("nan", "None"):
                    st.image(f"https://image.tmdb.org/t/p/w300{poster}")
                st.write(rec.get('title', 'Unknown'))
                st.caption(
                    f"⭐ {rec.get('avg_rating', 0.0):.1f} · {rec.get('rating_count', 0):,} ratings")


def render_recommend():
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
        st.session_state.recommendation_source = ""

    catalog = engine.catalog
    title_to_id = dict(zip(catalog['title'], catalog['movieId']))

    left, right = st.columns(2)
    with left:
        st.subheader("Movies you liked")
        selected_titles = st.multiselect(
            "Pick one or more movies", options=list(title_to_id))
        if st.button("Recommend from these movies", use_container_width=True):
            if selected_titles:
                ratings = {int(title_to_id[title]): 5.0 for title in selected_titles}
                try:
                    user_vec = engine.get_user_vector(ratings)
                    st.session_state.recommendations = engine.personalized_recommend(
                        user_vec, top_k=TOP_K, exclude_ids=list(ratings))
                    st.session_state.recommendation_source = "Because you liked: " + \
                        ", ".join(selected_titles)
                except Exception as e:
                    st.error(f"Could not build recommendations: {e}")
            else:
                st.warning("Pick at least one movie.")

    with right:
        st.subheader("Or describe what you want to watch")
        description = st.text_input(
            "Description", placeholder="e.g. a slow-burn heist thriller with a twist ending")
        if st.button("Recommend from this description", use_container_width=True):
            if description.strip():
                try:
                    st.session_state.recommendations = engine.search_by_description(
                        description, top_k=TOP_K)
                    st.session_state.recommendation_source = f"Matching: {description}"
                except Exception as e:
                    st.error(f"Search failed: {e}")
            else:
                st.warning("Type a short description first.")

    st.divider()
    if st.session_state.recommendations:
        st.subheader(f"Top {TOP_K} for you")
        st.caption(st.session_state.recommendation_source)
        display_recommendations(st.session_state.recommendations)
    else:
        st.subheader("Not sure where to start? Highly rated movies")
        display_recommendations(engine.get_trending_by_rating(
            min_rating=4.0, min_votes=10000, top_k=TOP_K))


# ==========================================
# 3. CHAT
# ==========================================


def render_chat():
    if chatbot is None:
        st.info("The chat needs a Groq API key (GROQ_API_KEY). The other tabs work without it.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = {
            "liked_genres": set(), "mentioned_movies": set()}

    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about movies (e.g. a tense action movie with a smart villain)"):
        # Remember genres the user mentions, to personalise later answers.
        if "hành động" in prompt.lower() or "action" in prompt.lower():
            st.session_state.entity_memory["liked_genres"].add("Action")
        if "khoa học viễn tưởng" in prompt.lower() or "sci-fi" in prompt.lower():
            st.session_state.entity_memory["liked_genres"].add("Sci-Fi")
        if "kinh dị" in prompt.lower() or "horror" in prompt.lower():
            st.session_state.entity_memory["liked_genres"].add("Horror")

        with chat_container:
            st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            bot_reply = chatbot.chat(
                prompt, history=st.session_state.messages[:-1], entity_memory=st.session_state.entity_memory)
        except Exception as e:
            bot_reply = f"Sorry, something went wrong: {e}"

        with chat_container:
            st.chat_message("assistant").markdown(bot_reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply})


# ==========================================
# 4. EVALUATION
# ==========================================


def load_report(name):
    """A file written by the scripts in scripts/, or None if it has not been generated."""
    path = os.path.join(REPORT_DIR, name)
    if not os.path.exists(path):
        return None
    if name.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def with_interval(metric, fmt):
    return f"{metric['value']:{fmt}} [{metric['ci95_low']:{fmt}}, {metric['ci95_high']:{fmt}}]"


def render_offline_eval(report):
    data, protocol, results = report["data"], report["protocol"], report["results"]
    st.subheader("Offline evaluation: can it find the next movie a user liked?")
    st.caption(
        f"{data['evaluated_users']:,} MovieLens users with at least "
        f"{protocol['min_liked_movies_per_user']} liked movies (rating ≥ "
        f"{protocol['liked_means_rating_at_least']}). The last liked movie is hidden; the earlier "
        f"ones are the input. Everything the user had already rated is excluded. "
        f"Square brackets: 95% bootstrap interval over users."
    )
    labels = {
        "popularity": "Popularity baseline (most rated movies)",
        "content_default": f"Content-based + reranker ({results['content_default']['candidates']} candidates)",
        "content_as_served": f"Content-based + reranker ({results['content_as_served']['candidates']} "
                             "candidates, as served on the Recommend tab)",
    }
    st.dataframe(
        pd.DataFrame([
            {
                "Recommender": labels[name],
                "HitRate@10": with_interval(result["hit_rate_at_10"], ".2%"),
                "NDCG@10": with_interval(result["ndcg_at_10"], ".4f"),
                "Catalogue coverage": f"{result['catalog_coverage']['value']:.2%}",
                "Long-tail share": with_interval(result["long_tail_share"], ".1%"),
            }
            for name, result in results.items()
        ]),
        hide_index=True, use_container_width=True,
    )
    diff = report["hit_rate_difference_default_minus_popularity"]
    st.caption(
        f"HitRate@10 difference, content-based minus popularity: {diff['value']:+.2%} "
        f"[{diff['ci95_low']:+.2%}, {diff['ci95_high']:+.2%}]. Long-tail share is the share of "
        f"recommendations outside the {report['head_movies']:,} most rated movies. Offline "
        f"metrics on MovieLens ratings do not replace an A/B test."
    )


def render_tradeoff(tradeoff):
    st.subheader("Popularity weight: accuracy against variety")
    left, right = st.columns([3, 2])
    with left:
        points = tradeoff.assign(label="pop_weight " + tradeoff["pop_weight"].astype(str))
        fig = px.line(points, x="long_tail_share", y="hit_rate_at_10", text="label", markers=True)
        fig.update_traces(line_color=CHART_COLOR, textposition="top center")
        fig.update_layout(xaxis_title="Long-tail share of recommendations", yaxis_title="HitRate@10",
                          xaxis_tickformat=".0%", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(
            pd.DataFrame({
                "pop_weight": tradeoff["pop_weight"],
                "HitRate@10": tradeoff["hit_rate_at_10"].map("{:.2%}".format),
                "NDCG@10": tradeoff["ndcg_at_10"].map("{:.4f}".format),
                "Coverage": tradeoff["catalog_coverage"].map("{:.2%}".format),
                "Long tail": tradeoff["long_tail_share"].map("{:.1%}".format),
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption("Quality weight fixed at 0.1; similarity weight = 0.9 − pop_weight.")


def render_latency(report):
    st.subheader("Latency of a text query")
    names = {"embedding": "Embed the query", "search": "LanceDB search", "rerank": "Rerank",
             "validation": "Validate output", "total": "Total"}
    st.dataframe(
        pd.DataFrame([
            {"Step": names.get(step, step), "p50 (ms)": f"{stats['p50_ms']:.1f}", "p95 (ms)": f"{stats['p95_ms']:.1f}"}
            for step, stats in report["latency"].items()
        ]),
        hide_index=True, use_container_width=True,
    )
    machine, cold = report["machine"], report["cold_start"]
    cold_parts = [f"{label} {cold[key]:.1f} s" for key, label in
                  [("download_s", "download"), ("unpack_s", "unpack"), ("load_model_s", "load model"),
                   ("load_table_s", "load table")] if cold.get(key) is not None]
    st.caption(
        f"{report['queries']} queries on {report['database']['movies']:,} movies, "
        f"{machine['device'].upper()} only ({machine['processor'] or machine['platform']}, "
        f"{machine['cpu_count']} logical cores). Cold start, measured once: {', '.join(cold_parts)}. "
        f"The LLM call is not included."
    )


def render_evaluation():
    offline_eval = load_report("offline_eval.json")
    tradeoff = load_report("tradeoff.csv")
    latency = load_report("latency.json")
    if offline_eval is not None:
        render_offline_eval(offline_eval)
    if tradeoff is not None:
        render_tradeoff(tradeoff)
    if latency is not None:
        render_latency(latency)
    if offline_eval is None and tradeoff is None and latency is None:
        st.info("No evaluation reports yet. Run `python -m scripts.evaluate_offline` and "
                "`python -m scripts.benchmark` to generate them.")

    catalog = engine.catalog
    st.subheader("The catalogue")
    st.caption(f"{len(catalog):,} movies in the vector database.")

    left, right = st.columns(2)
    with left:
        genres = catalog['genres'].fillna("").str.split('|').str[0].replace("", "Unknown")
        by_genre = genres.value_counts().rename_axis("genre").reset_index(name="movies")
        fig = px.bar(by_genre.sort_values("movies"), x="movies", y="genre", orientation="h",
                     title="Movies by first listed genre")
        fig.update_traces(marker_color=CHART_COLOR)
        fig.update_layout(yaxis_title=None, xaxis_title="Movies")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = np.sort(catalog['rating_count'].to_numpy())[::-1]
        curve = pd.DataFrame({"rank": np.arange(1, len(counts) + 1), "ratings": counts})
        fig = px.line(curve, x="rank", y="ratings", log_y=True,
                      title="Popularity is long-tailed: ratings per movie, by rank")
        fig.update_traces(line_color=CHART_COLOR)
        fig.update_layout(xaxis_title="Movie rank by number of ratings",
                          yaxis_title="Ratings (log scale)")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 5. MAIN
# ==========================================


def main():
    st.title("🎬 Movie Recommender")
    st.caption("Tell it which movies you liked, or describe what you feel like watching.")

    if engine is None:
        st.error("The vector database is not available, so recommendations cannot be shown.")
        return

    tab_recommend, tab_chat, tab_evaluation = st.tabs(["Recommend", "Chat", "Evaluation"])
    with tab_recommend:
        render_recommend()
    with tab_chat:
        render_chat()
    with tab_evaluation:
        render_evaluation()


if __name__ == "__main__":
    main()
