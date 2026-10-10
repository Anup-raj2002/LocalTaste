import sqlite3
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "localtaste.db"
OUT_CSV = BASE_DIR / "data" / "processed" / "recommender_eval.csv"

K = 5
RATING_THRESHOLD = 3.5
MIN_LOCALITY_SIZE = 10
random.seed(42)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT name, location, cuisines, rate FROM restaurants", conn)
df = df.dropna(subset=["cuisines", "location", "rate"]).reset_index(drop=True)
df["cuisine_set"] = df["cuisines"].apply(lambda s: {c.strip().lower() for c in s.split(",")})

vectorizer = TfidfVectorizer(
    tokenizer=lambda s: [c.strip() for c in s.split(",")], token_pattern=None
)
matrix = vectorizer.fit_transform(df["cuisines"])


def metrics(rows, query_set):
    match = rows["cuisine_set"].apply(lambda s: len(s & query_set) > 0)
    ok = rows["rate"] >= RATING_THRESHOLD
    return match.mean(), ok.mean(), (match & ok).mean()


records = []
for locality, group in df.groupby("location"):
    idx = group.index.to_numpy()
    n = len(idx)

    counts = Counter(c for s in group["cuisine_set"] for c in s)
    top = [c for c, _ in counts.most_common(5)]
    queries = [[c] for c in top] + [list(p) for p in combinations(top[:4], 2)]

    model = NearestNeighbors(metric="cosine", n_neighbors=min(K, n)).fit(matrix[idx])

    for q in queries:
        q_set = set(q)
        vec = vectorizer.transform([", ".join(q)])
        dist, pos = model.kneighbors(vec, n_neighbors=min(K, n))
        knn_rows = group.iloc[pos[0]]

        rec = {"locality": locality, "n_restaurants": n, "query": ", ".join(q),
               "returned": len(knn_rows), "top1_similarity": 1 - dist[0][0]}

        if n >= MIN_LOCALITY_SIZE:
            rec["knn_match"], rec["knn_rating"], rec["knn_p"] = metrics(knn_rows, q_set)
            rand_rows = group.sample(min(K, n), random_state=random.randint(0, 10**6))
            rec["rand_match"], rec["rand_rating"], rec["rand_p"] = metrics(rand_rows, q_set)
            top_rows = group.sort_values("rate", ascending=False).head(K)
            rec["top_match"], rec["top_rating"], rec["top_p"] = metrics(top_rows, q_set)
        records.append(rec)

res = pd.DataFrame(records)
res.to_csv(OUT_CSV, index=False)
evald = res.dropna(subset=["knn_p"])

print(f"Queries run: {len(res)} across {res['locality'].nunique()} localities")
print(f"Coverage (localities returning >=1 result): "
      f"{res.groupby('locality')['returned'].max().gt(0).mean():.1%}")
print(f"Localities evaluated for precision (n>={MIN_LOCALITY_SIZE}): "
      f"{evald['locality'].nunique()}  |  queries: {len(evald)}")
print(f"Mean top-1 similarity: {evald['top1_similarity'].mean():.3f}\n")

table = pd.DataFrame({
    "Cuisine match@5": [evald["knn_match"].mean(), evald["rand_match"].mean(), evald["top_match"].mean()],
    f"Rating>={RATING_THRESHOLD}@5": [evald["knn_rating"].mean(), evald["rand_rating"].mean(), evald["top_rating"].mean()],
    "Precision@5 (both)": [evald["knn_p"].mean(), evald["rand_p"].mean(), evald["top_p"].mean()],
}, index=["KNN (ours)", "Random in locality", "Top-rated in locality"])
print(table.round(3))
print(f"\nPer-query results saved to {OUT_CSV}")