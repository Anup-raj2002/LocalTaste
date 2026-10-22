import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))
from services import gemini_service

st.set_page_config(page_title="LocalTaste", layout="centered")


@st.cache_data
def load_restaurants():
    csv_path = BASE_DIR / "data" / "processed" / "restaurants_clean.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["cuisines", "location"]).reset_index(drop=True)
    return df


@st.cache_resource
def build_recommender(df):
    vectorizer = TfidfVectorizer(token_pattern=r"[^,]+")
    cuisine_matrix = vectorizer.fit_transform(df["cuisines"])

    locality_models = {}
    for locality, group in df.groupby("location"):
        indices = group.index.to_numpy()
        vectors = cuisine_matrix[indices]
        k = min(6, len(indices))
        model = NearestNeighbors(metric="cosine", n_neighbors=k)
        model.fit(vectors)
        locality_models[locality] = (model, indices)

    # Universal city-wide KNN model
    city_k = min(10, len(df))
    city_model = NearestNeighbors(metric="cosine", n_neighbors=city_k)
    city_model.fit(cuisine_matrix)

    return vectorizer, cuisine_matrix, locality_models, city_model


@st.cache_resource
def load_sentiment_model():
    model = joblib.load(BASE_DIR / "models" / "sentiment_model.pkl")
    vec = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
    return model, vec


df = load_restaurants()
vectorizer, cuisine_matrix, locality_models, city_model = build_recommender(df)
sentiment_model, sentiment_vectorizer = load_sentiment_model()


def recommend(user_locality, liked_names, top_k=5):
    liked_rows = df[df["name"].isin(liked_names)]
    if liked_rows.empty:
        return pd.DataFrame()

    liked_indices = liked_rows.index.to_numpy()
    taste_vector = np.asarray(cuisine_matrix[liked_indices].mean(axis=0))

    if user_locality and user_locality in locality_models:
        model, indices = locality_models[user_locality]
        k = min(top_k + len(liked_indices), len(indices))
        distances, neighbor_pos = model.kneighbors(taste_vector, n_neighbors=k)
        result_indices = indices[neighbor_pos[0]]
    else:
        k = min(top_k + len(liked_indices), len(df))
        distances, neighbor_pos = city_model.kneighbors(taste_vector, n_neighbors=k)
        result_indices = neighbor_pos[0]

    results = df.loc[result_indices, ["name", "location", "cuisines", "rate"]].copy()
    results["similarity"] = 1 - distances[0]
    results = results[~results["name"].isin(liked_names)]
    return results.sort_values("similarity", ascending=False).head(top_k)


st.title("LocalTaste")
st.caption("Locality-aware & City-wide restaurant recommendations, backed by review sentiment.")

localities = ["All Bangalore (City-wide)"] + sorted(df["location"].unique())
selected_locality = st.selectbox("Your locality", localities)

if selected_locality == "All Bangalore (City-wide)":
    restaurants_in_locality = sorted(df["name"].unique().tolist())
else:
    restaurants_in_locality = sorted(df[df["location"] == selected_locality]["name"].tolist())

liked = st.multiselect("Restaurants you've liked before", restaurants_in_locality)

if st.button("Get Recommendations"):
    if not liked:
        st.warning("Pick at least one restaurant you've liked.")
    else:
        recs = recommend(selected_locality, liked, top_k=5)
        if recs.empty:
            st.error("No recommendations found for this combination.")
        else:
            st.subheader("Recommended for you")
            for _, row in recs.iterrows():
                st.markdown(
                    f"**{row['name']}** ({row['location']}) — {row['cuisines']}  \n"
                    f"Rating: {row['rate']} | Similarity: {row['similarity']:.2f}"
                )

st.divider()

st.subheader("Check review sentiment")
review_input = st.text_area("Paste a review to classify:")
if st.button("Classify Sentiment"):
    if review_input.strip():
        vec = sentiment_vectorizer.transform([review_input])
        pred = str(sentiment_model.predict(vec)[0])
        probs = sentiment_model.predict_proba(vec)[0]
        conf = float(np.max(probs)) * 100
        st.write(f"Predicted sentiment: **{pred.upper()}** (Confidence: **{conf:.1f}%**)")
        reply = gemini_service.generate_review_response(review_input, pred)
        if pred == "positive":
            st.success(reply)
        else:
            st.error(reply)
    else:
        st.warning("Enter some review text first.")
