import pandas as pd

print("Loading original clean files...")
restaurants = pd.read_csv("restaurants_clean.csv")
reviews = pd.read_csv("reviews_labeled.csv")

print("Restaurants before dedup:", len(restaurants))
print("Reviews before dedup:", len(reviews))

# ---------- Dedup restaurants: same name + location = same physical place ----------
restaurants = restaurants.drop_duplicates(subset=["name", "location"], keep="first").reset_index(drop=True)
print("Restaurants after dedup:", len(restaurants))

# ---------- Dedup reviews: same restaurant + same review text = duplicate scrape ----------
reviews = reviews.drop_duplicates(
    subset=["restaurant_name", "locality", "review_text"], keep="first"
).reset_index(drop=True)
print("Reviews after dedup:", len(reviews))
print(reviews["sentiment_label"].value_counts())

# ---------- Overwrite the clean files with deduplicated versions ----------
restaurants.to_csv("restaurants_clean.csv", index=False)
reviews.to_csv("reviews_labeled.csv", index=False)
print("\nOverwrote restaurants_clean.csv and reviews_labeled.csv with deduplicated data")