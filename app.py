import os
import shutil

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.serving.chatbot import MovieChatbot
from src.serving.semantic_search import SemanticSearchEngine

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

CHART_COLOR = "#636EFA"
TOP_K = 10
DB_PATH = "lancedb_movies"
DB_ZIP = "lancedb_movies.zip"


# ==========================================
# 1. SERVICES
# ==========================================


def get_s3_client():
    return boto3.client('s3',
                        endpoint_url=os.environ.get('AWS_ENDPOINT_URL'),
                        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.environ.get(
                            'AWS_SECRET_ACCESS_KEY')
                        )


@st.cache_resource
def init_lancedb():
    """Download the vector database from Cloudflare R2 (first run only)."""
    if not os.path.exists(DB_PATH):
        if not os.environ.get('AWS_ACCESS_KEY_ID'):
            st.warning("R2 environment variables are missing. Check your .env file.")
            return None

        try:
            s3 = get_s3_client()
            bucket = os.environ.get('S3_BUCKET_NAME', 'movie-mlops')
            s3.download_file(bucket, DB_ZIP, DB_ZIP)
            shutil.unpack_archive(DB_ZIP, DB_PATH)
            os.remove(DB_ZIP)
        except Exception as e:
            st.error(f"Could not download the database from R2: {e}")
            return None

    engine = SemanticSearchEngine(lancedb_uri=DB_PATH)
    engine.load_table()
    return engine


@st.cache_resource
def init_chatbot():
    """The chat tab needs a Groq key; the rest of the app works without it."""
    try:
        return MovieChatbot()
    except ValueError:
        return None


engine = init_lancedb()
chatbot = init_chatbot()


@st.cache_data(ttl=3600)
def get_catalog():
    """Movie metadata from LanceDB (no vectors)."""
    df = engine.table.to_pandas()
    return df.drop(columns=["vector"])


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

    catalog = get_catalog()
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
                        user_vec, top_k=TOP_K)
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


def render_evaluation():
    catalog = get_catalog()
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
