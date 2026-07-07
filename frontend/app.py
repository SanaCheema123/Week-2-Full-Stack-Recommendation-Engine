
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from pathlib import Path

API = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="RecoFlix AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f8fafc,#fff7ed); font-family:Segoe UI, sans-serif;}
.block-container {padding-top:1.4rem; max-width:1500px;}
#MainMenu, footer, header {display:none!important;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#111827,#7f1d1d);}
section[data-testid="stSidebar"] * {color:white!important;}
.hero {background:linear-gradient(135deg,#111827,#dc2626,#f97316); padding:38px 42px; border-radius:28px; color:white; box-shadow:0 20px 45px rgba(220,38,38,.22); margin-bottom:26px;}
.hero h1 {color:white!important; font-size:40px; margin:0 0 10px 0;}
.hero p {color:rgba(255,255,255,.92)!important; font-size:17px;}
.card {background:white; padding:24px; border-radius:22px; border:1px solid #fee2e2; box-shadow:0 12px 30px rgba(15,23,42,.07);}
.card p {color:#64748b; margin:0; font-weight:700;}
.card h2 {color:#111827; margin:12px 0 0 0; font-size:30px;}
.stButton > button {background:linear-gradient(135deg,#dc2626,#f97316); color:white; border:none; border-radius:14px; font-weight:800; width:100%;}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)

def card(title, value, icon):
    st.markdown(f"<div class='card'><div style='font-size:28px'>{icon}</div><p>{title}</p><h2>{value}</h2></div>", unsafe_allow_html=True)

def get(path):
    try:
        r = requests.get(API + path, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def rec_table(path):
    data = get(path)
    if "error" in data:
        st.error("Backend API is not running. Start: py -m uvicorn backend.main:app --reload")
        return
    df = pd.DataFrame(data.get("recommendations", []))
    if df.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with st.sidebar:
    st.markdown("## 🎬 RecoFlix AI")
    st.caption("FastAPI + Streamlit Recommendation Engine")
    page = st.radio("Navigation", ["Dashboard","Movie Search","Content Recommendation","User Recommendation","Hybrid Recommendation","Analytics","About"])
    st.info("Run backend first:\npy -m uvicorn backend.main:app --reload")

if page == "Dashboard":
    hero("Recommendation Engine Dashboard", "Netflix/Amazon-style recommendation system using content-based filtering, collaborative filtering, hybrid logic, FastAPI, and Streamlit.")
    stats = get("/movies/stats/summary")
    if "error" in stats:
        st.error("Backend API is not running. Start backend first.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1: card("Movies", stats["total_movies"], "🎬")
        with c2: card("Users", stats["total_users"], "👥")
        with c3: card("Ratings", stats["total_ratings"], "⭐")
        with c4: card("Average Rating", stats["average_rating"], "📊")
        movies = pd.DataFrame(get("/movies/?limit=100"))
        st.subheader("Movie Dataset Preview")
        st.dataframe(movies.head(20), use_container_width=True, hide_index=True)
        genre_counts = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")["genres"].value_counts().reset_index()
        genre_counts.columns = ["Genre","Count"]
        st.plotly_chart(px.bar(genre_counts, x="Genre", y="Count", title="Genre Distribution"), use_container_width=True)

elif page == "Movie Search":
    hero("Movie Search", "Search movies by title.")
    q = st.text_input("Movie title", "Toy")
    if st.button("Search"):
        df = pd.DataFrame(get(f"/movies/search/{q}"))
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Content Recommendation":
    hero("Content-Based Recommendation", "Recommend movies similar to a selected movie using genres, tags, TF-IDF, and cosine similarity.")
    title = st.text_input("Movie title", "Toy Story")
    n = st.slider("Top N", 3, 15, 10)
    if st.button("Get Similar Movies"):
        rec_table(f"/recommend/content/{title}?top_n={n}")

elif page == "User Recommendation":
    hero("Collaborative Filtering", "Personalized recommendations using similar users and rating behavior.")
    user = st.number_input("User ID", 1, 80, 1)
    n = st.slider("Top N", 3, 15, 10)
    if st.button("Get User Recommendations"):
        rec_table(f"/recommend/user/{user}?top_n={n}")

elif page == "Hybrid Recommendation":
    hero("Hybrid Recommendation", "Combine collaborative filtering and content-based recommendation.")
    user = st.number_input("User ID", 1, 80, 1)
    title = st.text_input("Movie title", "Toy Story")
    n = st.slider("Top N", 3, 15, 10)
    if st.button("Get Hybrid Recommendations"):
        rec_table(f"/recommend/hybrid/{user}?movie_title={title}&top_n={n}")

elif page == "Analytics":
    hero("Recommendation Analytics", "Analyze rating distribution, popular movies, and genre trends.")
    movies = pd.read_csv(ROOT/"data/movies.csv")
    ratings = pd.read_csv(ROOT/"data/ratings.csv")
    merged = ratings.merge(movies, on="movieId")
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(ratings, x="rating", title="Rating Distribution"), use_container_width=True)
    with c2:
        pop = merged.groupby("title", as_index=False).agg(count=("rating","count"), avg_rating=("rating","mean")).sort_values("count", ascending=False).head(10)
        st.plotly_chart(px.bar(pop, x="title", y="count", title="Most Rated Movies"), use_container_width=True)
    genres = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")["genres"].value_counts().reset_index()
    genres.columns = ["Genre","Count"]
    st.plotly_chart(px.pie(genres, names="Genre", values="Count", title="Genre Share"), use_container_width=True)

else:
    hero("About Project", "Week 2 Recommendation Engine with FastAPI backend and Streamlit dashboard.")
    st.markdown("""
    ### Run Backend
    `py -m uvicorn backend.main:app --reload`

    ### Run Dashboard
    `py -m streamlit run frontend/app.py`

    ### Swagger API
    `http://127.0.0.1:8000/docs`

    ### Methods
    - Content-Based Recommendation
    - Collaborative Filtering
    - Hybrid Recommendation
    """)
