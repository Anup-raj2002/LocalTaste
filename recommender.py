import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

print("Loading restaurants_clean.csv...")
df = pd.read_csv("restaurants_clean.csv")
df = df.dropna(subset=["cuisines", "location"]).reset_index(drop=True)

print("Total restaurants (already deduplicated at source):", len(df))

# ---------- Build one shared TF-IDF vocabulary over cuisines ----------
vectorizer = TfidfVectorizer(token_pattern=r"[^,]+")  # split on commas, not spaces
cuisine_matrix = vectorizer.fit_transform(df["cuisines"])

print("\nTF-IDF vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Sample cuisine terms:", vectorizer.get_feature_names_out()[:10])

# ---------- Fit one KNN model per locality ----------
locality_models = {}   # locality -> (NearestNeighbors model, row indices in df)

for locality, group in df.groupby("location"):
    indices = group.index.to_numpy()
    vectors = cuisine_matrix[indices]

    k = min(6, len(indices))  # fetch one extra, in case we need to filter liked ones out
    model = NearestNeighbors(metric="cosine", n_neighbors=k)
    model.fit(vectors)

    locality_models[locality] = (model, indices)

print(f"\nBuilt {len(locality_models)} locality-specific KNN models.")

# ---------- Check locality sizes post-dedup (some may now be very small) ----------
print("\nSmallest localities after dedup:")
print(df["location"].value_counts().tail(15))


# ---------- Recommend, excluding restaurants the user already liked ----------
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
    taste_vector = np.asarray(liked_vectors.mean(axis=0))

    model, indices = locality_models[user_locality]
    k = min(top_k + len(liked_indices), len(indices))  # fetch extra to survive exclusion
    distances, neighbor_pos = model.kneighbors(taste_vector, n_neighbors=k)

    result_indices = indices[neighbor_pos[0]]
    results = df.loc[result_indices, ["name", "location", "cuisines", "rate"]].copy()
    results["similarity"] = 1 - distances[0]

    # exclude restaurants the user already liked/rated
    results = results[~results["name"].isin(liked_restaurant_names)]

    return results.sort_values("similarity", ascending=False).head(top_k)


# ---------- Try it on Banashankari ----------
sample_recs = recommend(
    user_locality="Banashankari",
    liked_restaurant_names=["Jalsa"],
    top_k=5
)
print("\nSample recommendations for a user in Banashankari who liked 'Jalsa':")
print(sample_recs)