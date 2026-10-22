import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from config import DB_PATH

RERANK_POOL = 10
CITY_RERANK_POOL = 100
RERANK_ALPHA = 0.7

print("Loading restaurant data from SQLite and building recommender...")
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_df = pd.read_sql_query("SELECT * FROM restaurants", _conn)
_df = _df.dropna(subset=["cuisines", "location"]).reset_index(drop=True)

_vectorizer = TfidfVectorizer(
    tokenizer=lambda s: [c.strip() for c in s.split(",")],
    token_pattern=None,
)
_cuisine_matrix = _vectorizer.fit_transform(_df["cuisines"])

_locality_models = {}
for _locality, _group in _df.groupby("location"):
    _indices = _group.index.to_numpy()
    _vectors = _cuisine_matrix[_indices]
    _k = min(max(6, RERANK_POOL), len(_indices))
    _model = NearestNeighbors(metric="cosine", n_neighbors=_k)
    _model.fit(_vectors)
    _locality_models[_locality] = (_model, _indices)

# Universal city-wide KNN model across all Bangalore restaurants
_city_k = min(max(6, CITY_RERANK_POOL), len(_df))
_city_model = NearestNeighbors(metric="cosine", n_neighbors=_city_k)
_city_model.fit(_cuisine_matrix)

# Universal average taste vector across all Bangalore restaurants (used when user doesn't specify a cuisine)
_city_mean_taste_vector = np.asarray(_cuisine_matrix.mean(axis=0))

print(f"Recommender ready: {len(_df)} restaurants, {len(_locality_models)} localities + universal city KNN.")

# Computed once at startup instead of on every request
_VALID_LOCALITIES = sorted(_df["location"].unique().tolist())
_VALID_CUISINES = sorted(set(c.strip() for row in _df["cuisines"] for c in row.split(",")))


def get_localities():
    return ["All Bangalore (City-wide)"] + _VALID_LOCALITIES


def get_restaurants_in_locality(locality: str):
    if not locality or locality in ("All Bangalore (City-wide)", "Bangalore", "All Bangalore"):
        return sorted(_df["name"].unique().tolist())
    return sorted(_df[_df["location"] == locality]["name"].tolist())


def get_valid_cuisines():
    return _VALID_CUISINES


def _top_dishes(dish_liked_value):
    if not isinstance(dish_liked_value, str) or not dish_liked_value.strip():
        return []
    return [d.strip() for d in dish_liked_value.split(",")][:5]


def _rerank(results, top_k, rerank):
    """Order candidates by similarity only, or by a similarity/rating blend."""
    if rerank:
        rating = results["rate"].fillna(0).clip(0, 5) / 5.0
        results = results.assign(
            _score=RERANK_ALPHA * results["similarity"] + (1 - RERANK_ALPHA) * rating
        ).sort_values(["_score", "rate"], ascending=[False, False])
    else:
        results = results.sort_values(["similarity", "rate"], ascending=[False, False])
    return results.head(top_k)


def recommend(user_locality: str, liked_names: list, top_k: int = 5):
    liked_rows = _df[_df["name"].isin(liked_names)]
    if liked_rows.empty:
        return []

    liked_indices = liked_rows.index.to_numpy()
    taste_vector = np.asarray(_cuisine_matrix[liked_indices].mean(axis=0))

    if user_locality and user_locality in _locality_models:
        model, indices = _locality_models[user_locality]
        k = min(top_k + len(liked_indices), len(indices))
        distances, neighbor_pos = model.kneighbors(taste_vector, n_neighbors=k)
        result_indices = indices[neighbor_pos[0]]
    else:
        # Fallback to universal city-wide KNN
        k = min(max(CITY_RERANK_POOL, top_k + len(liked_indices)), len(_df))
        distances, neighbor_pos = _city_model.kneighbors(taste_vector, n_neighbors=k)
        result_indices = neighbor_pos[0]

    results = _df.loc[result_indices].copy()
    results["similarity"] = 1 - distances[0]
    results = results[~results["name"].isin(liked_names)]
    results = _rerank(results, top_k, rerank=True)

    return _format_results(results)


def recommend_by_cuisines(user_locality: str | None, preferred_cuisines: list,
                          top_k: int = 5, rerank: bool = True):
    if preferred_cuisines:
        query_text = ", ".join(preferred_cuisines)
        taste_vector = _vectorizer.transform([query_text]).toarray()
    else:
        if user_locality and user_locality in _locality_models:
            _, indices = _locality_models[user_locality]
            taste_vector = np.asarray(_cuisine_matrix[indices].mean(axis=0))
        else:
            taste_vector = _city_mean_taste_vector

    if user_locality and user_locality in _locality_models:
        model, indices = _locality_models[user_locality]
        pool = min(max(RERANK_POOL, top_k), len(indices)) if rerank else min(top_k, len(indices))
        distances, neighbor_pos = model.kneighbors(taste_vector, n_neighbors=pool)
        result_indices = indices[neighbor_pos[0]]
        results = _df.loc[result_indices].copy()
        results["similarity"] = 1 - distances[0]

        # If the selected locality has 0 cuisine match (e.g. Kengeri has no Gujarati/Dhokla spot),
        # or fewer than top_k matching restaurants, supplement with Universal City-wide KNN matches!
        matching_local = results[results["similarity"] > 0]
        if len(matching_local) < top_k and preferred_cuisines:
            city_pool = min(max(CITY_RERANK_POOL, top_k), len(_df))
            c_dist, c_pos = _city_model.kneighbors(taste_vector, n_neighbors=city_pool)
            c_indices = c_pos[0]
            city_results = _df.loc[c_indices].copy()
            city_results["similarity"] = 1 - c_dist[0]
            city_results = city_results[~city_results.index.isin(matching_local.index)]
            city_ranked = _rerank(city_results, top_k - len(matching_local), rerank)
            local_ranked = _rerank(matching_local, len(matching_local), rerank)
            results = pd.concat([local_ranked, city_ranked], ignore_index=True)
            return _format_results(results)
    else:
        # Universal city-wide KNN across all of Bangalore
        pool = min(max(CITY_RERANK_POOL, top_k), len(_df)) if rerank else min(top_k, len(_df))
        distances, neighbor_pos = _city_model.kneighbors(taste_vector, n_neighbors=pool)
        result_indices = neighbor_pos[0]
        results = _df.loc[result_indices].copy()
        results["similarity"] = 1 - distances[0]

    results = _rerank(results, top_k, rerank)

    return _format_results(results)


def _format_results(results_df):
    output = []
    for _, row in results_df.iterrows():
        output.append({
            "name": row["name"],
            "location": row["location"],
            "cuisines": row["cuisines"],
            "rate": float(row["rate"]) if pd.notna(row["rate"]) else None,
            "similarity": round(float(row["similarity"]), 3),
            "top_dishes": _top_dishes(row.get("dish_liked", None)),
        })
    return output
