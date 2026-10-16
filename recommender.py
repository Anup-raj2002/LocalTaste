import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

print("Loading restaurants_clean.csv...")
df = pd.read_csv("restaurants_clean.csv")
df = df.dropna(subset=["cuisines", "location"]).reset_index(drop=True)

print("Total restaurants (already deduplicated at source):", len(df))

vectorizer = TfidfVectorizer(token_pattern=r"[^,]+")
cuisine_matrix = vectorizer.fit_transform(df["cuisines"])

print("\nTF-IDF vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Sample cuisine terms:", vectorizer.get_feature_names_out()[:10])

locality_models = {}

for locality, group in df.groupby("location"):
    indices = group.index.to_numpy()
    vectors = cuisine_matrix[indices]

    k = min(6, len(indices))
    model = NearestNeighbors(metric="cosine", n_neighbors=k)
    model.fit(vectors)

    locality_models[locality] = (model, indices)

print(f"\nBuilt {len(locality_models)} locality-specific KNN models.")

print("\nSmallest localities after dedup:")
print(df["location"].value_counts().tail(15))


def recommend(user_locality, liked_restaurant_names, top_k=5):
    if user_locality not in locality_models:
        print(f"No data for locality '{user_locality}'")
        return pd.DataFrame()

    liked_rows = df[df["name"].isin(liked_restaurant_names)]
    if liked_rows.empty:
        print("None of the liked restaurant names were found.")
        return pd.DataFrame()

    liked_indices = liked_rows.index.to_numpy()
    liked_vectors = cuisine_matrix[liked_indices]
    taste_vector = np