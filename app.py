import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="LocalTaste", layout="centered")

@st.cache_data
def load_restaurants():
    df = pd.read_csv("restaurants_clean.csv")
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

    return vectorizer, cuisine_matrix, locality_models

@st.cache_resource
def load_sentiment_model():
    model = joblib.load("sentiment_model.pkl")
    vec = joblib.load("tfidf_vectorizer.pkl")
    return model, vec

df = load_restaurants()
vectorizer, cuisine_matrix, locality_models = build_recommender(df)
sentiment_model, sentiment_vectorizer = load_sentiment_model()

def recommend(user_locality, liked_names, top_k=5):
    liked_rows = df[df["name"].isin(liked_names)]
    if liked_rows.empty or user_locality not in locality_models:
        return pd.DataFrame()

    liked_indices = liked_rows.index.to_numpy()
    taste_vector = np.asarray(cuisine_matrix[liked_indices].mean(axis=0))

    model, indices = locality_models[user_locality]
    k = min(top_k + len(liked_indices), len(indices))
    distances, neighbor_pos = model.kneighbors(taste_vector, n_neighbors=k)

    result_indices = indices[neighbor_pos[0]]
    results = df.loc[result_indices, ["name", "location", "cuisines", "rate"]].copy()
    results["similarity"] = 1 - distances[0]
    results = results[~results["name"].isin(liked_names)]
    return results.sort_values("similarity", ascending=False).head(top_k)


st.title("LocalTaste")
st.caption("Locality-aware restaurant recommendations, backed by review sentiment.")

localities = sorted(df["location"].unique())
selected_locality = st.selectbox("Your locality", localities)

restaurants_in_locality = df[df["location"] == selected_locality]["name"].tolist()
liked = st.multiselect("Restaurants you've liked before", sorted(restaurants_in_locality))

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
                st.markdown(f"**{row['name']}** — {row['cuisines']}  \n"
                            f"Rating: {row['rate']} | Similarity: {row['similarity']:.2f}")

st.divider()

st.subheader("Check review sentiment")
review_input = st.text_area("Paste a review to classify:")
if st.button("Classify Sentiment"):
    if review_input.strip():
        vec = sentiment_vectorizer.transform([review_input])
        pred = sentiment_model.predict(vec)[0]
        st.write(f"Predicted sentiment: **{pred}**")
    else:
        st.warning("Enter some review text first.")