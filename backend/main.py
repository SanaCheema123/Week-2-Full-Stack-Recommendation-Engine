
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

app = FastAPI(
    title="Week 2 Recommendation Engine API",
    description="Content-based, collaborative filtering, and hybrid recommendation system.",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def movies_df():
    return pd.read_csv(DATA / "movies.csv")

def ratings_df():
    return pd.read_csv(DATA / "ratings.csv")

def tags_df():
    p = DATA / "tags.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["userId","movieId","tag","timestamp"])

def content_recs(title, top_n=10):
    movies = movies_df()
    tags = tags_df()
    if not tags.empty:
        tag_text = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index()
        movies = movies.merge(tag_text, on="movieId", how="left")
    else:
        movies["tag"] = ""
    movies["tag"] = movies["tag"].fillna("")
    movies["content"] = movies["title"] + " " + movies["genres"].str.replace("|"," ", regex=False) + " " + movies["tag"]
    matches = movies[movies["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        return []
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(movies["content"])
    sim = cosine_similarity(mat)
    idx = matches.index[0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x:x[1], reverse=True)[1:top_n+1]
    return [{"movieId":int(movies.iloc[i]["movieId"]), "title":movies.iloc[i]["title"], "genres":movies.iloc[i]["genres"], "score":round(float(s),4)} for i,s in scores]

def popular_recs(top_n=10):
    movies = movies_df()
    ratings = ratings_df()
    pop = ratings.groupby("movieId").agg(avg_rating=("rating","mean"), count=("rating","count")).reset_index()
    pop["score"] = pop["avg_rating"] * np.log1p(pop["count"])
    pop = pop.sort_values("score", ascending=False).head(top_n).merge(movies,on="movieId",how="left")
    return [{"movieId":int(r["movieId"]), "title":r["title"], "genres":r["genres"], "score":round(float(r["score"]),4)} for _,r in pop.iterrows()]

def user_recs(user_id, top_n=10):
    movies = movies_df()
    ratings = ratings_df()
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    if user_id not in matrix.index:
        return popular_recs(top_n)
    sim = cosine_similarity(matrix)
    simdf = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    similar = simdf[user_id].sort_values(ascending=False).drop(user_id).head(10)
    scores = pd.Series(dtype=float)
    for other, s in similar.items():
        scores = scores.add(matrix.loc[other] * s, fill_value=0)
    already = ratings[ratings["userId"]==user_id]["movieId"].tolist()
    scores = scores.drop(labels=already, errors="ignore").sort_values(ascending=False).head(top_n)
    out=[]
    for mid, score in scores.items():
        row = movies[movies["movieId"]==mid].iloc[0]
        out.append({"movieId":int(row["movieId"]), "title":row["title"], "genres":row["genres"], "score":round(float(score),4)})
    return out

@app.get("/")
def home():
    return {"project":"Full-Stack Recommendation Engine", "docs":"http://127.0.0.1:8000/docs", "dashboard":"Run Streamlit frontend/app.py"}

@app.get("/health")
def health():
    return {"status":"running"}

@app.get("/movies/")
def list_movies(limit:int=20):
    return movies_df().head(limit).to_dict(orient="records")

@app.get("/movies/stats/summary")
def stats():
    m=movies_df(); r=ratings_df(); t=tags_df()
    return {"total_movies":int(m.movieId.nunique()),"total_users":int(r.userId.nunique()),"total_ratings":int(len(r)),"total_tags":int(len(t)),"average_rating":round(float(r.rating.mean()),2)}

@app.get("/movies/search/{title}")
def search(title:str):
    m=movies_df()
    return m[m["title"].str.contains(title, case=False, na=False)].head(20).to_dict(orient="records")

@app.get("/recommend/popular")
def recommend_popular(top_n:int=10):
    return {"method":"popular","query":"global","recommendations":popular_recs(top_n)}

@app.get("/recommend/content/{movie_title}")
def recommend_content(movie_title:str, top_n:int=10):
    return {"method":"content_based","query":movie_title,"recommendations":content_recs(movie_title, top_n)}

@app.get("/recommend/user/{user_id}")
def recommend_user(user_id:int, top_n:int=10):
    return {"method":"collaborative_filtering","query":str(user_id),"recommendations":user_recs(user_id, top_n)}

@app.get("/recommend/hybrid/{user_id}")
def recommend_hybrid(user_id:int, movie_title:str="", top_n:int=10):
    collab = user_recs(user_id, top_n)
    content = content_recs(movie_title, top_n) if movie_title else []
    combined = {}
    for x in collab:
        combined[x["movieId"]] = {**x, "score": x["score"]*0.6}
    for x in content:
        if x["movieId"] in combined:
            combined[x["movieId"]]["score"] += x["score"]*0.4
        else:
            combined[x["movieId"]] = {**x, "score": x["score"]*0.4}
    recs = sorted(combined.values(), key=lambda x:x["score"], reverse=True)[:top_n] if combined else popular_recs(top_n)
    return {"method":"hybrid","query":f"user={user_id}, movie={movie_title}","recommendations":recs}
