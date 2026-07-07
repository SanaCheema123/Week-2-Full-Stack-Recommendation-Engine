# 🎬 Week 2 Full-Stack Recommendation Engine
[screen-capture (2).webm](https://github.com/user-attachments/assets/518e46d2-cc46-4d93-99f9-0fb9019e4e42)

## Project Overview

This project is a complete **Recommendation Engine** delivered alongside **Week 2**. It includes both:

- **FastAPI backend**
- **Streamlit dashboard frontend**

The system mirrors real-world applications such as Netflix-style and Amazon-style recommendation systems.

## Dataset

Recommended Kaggle dataset:

```text
https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset
```

This project includes a sample MovieLens-style dataset:

```text
data/movies.csv
data/ratings.csv
data/tags.csv
```

## Techniques Used

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation
- TF-IDF Vectorization
- Cosine Similarity
- User-Item Matrix
- FastAPI Deployment
- Streamlit Dashboard

## Folder Structure

```text
week2_recommendation_engine_fullstack/
│
├── backend/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
│
├── models/
├── docs/
├── train.py
├── requirements.txt
├── run_backend.bat
├── run_dashboard.bat
└── README.md
```

## Installation

```bash
py -m pip install -r requirements.txt
```

## Run Backend

Open PowerShell in the project folder:

```bash
py -m uvicorn backend.main:app --reload
```

Swagger API:

```text
http://127.0.0.1:8000/docs
```

## Run Dashboard

Open a second PowerShell window:

```bash
py -m streamlit run frontend\app.py
```

## Windows Quick Run

First double-click:

```text
run_backend.bat
```

Then double-click:

```text
run_dashboard.bat
```

## Dashboard Pages

- Dashboard
- Movie Search
- Content Recommendation
- User Recommendation
- Hybrid Recommendation
- Analytics
- About

## API Endpoints

```text
GET /
GET /health
GET /movies/
GET /movies/stats/summary
GET /movies/search/{title}
GET /recommend/popular
GET /recommend/content/{movie_title}
GET /recommend/user/{user_id}
GET /recommend/hybrid/{user_id}
```

## Learning Outcomes

Participants learn how to:

- Build a recommendation system
- Use content-based filtering
- Use collaborative filtering
- Combine methods into a hybrid recommender
- Deploy APIs with FastAPI
- Build dashboards with Streamlit
- Test APIs using Swagger UI
