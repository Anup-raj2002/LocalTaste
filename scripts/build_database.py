import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DB_PATH = DATA_PROCESSED / "localtaste.db"

print("Loading CSVs...")
restaurants = pd.read_csv(DATA_PROCESSED / "restaurants_clean.csv")
reviews = pd.read_csv(DATA_PROCESSED / "reviews_labeled.csv")

print(f"Restaurants: {len(restaurants)}, Reviews: {len(reviews)}")

print("Creating SQLite database...")
conn = sqlite3.connect(DB_PATH)

restaurants.to_sql("restaurants", conn, if_exists="replace", index=True, index_label="restaurant_id")
reviews.to_sql("reviews", conn, if_exists="replace", index=True, index_label="review_id")

# Index the columns we filter/join on most, for real query performance
cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants(location)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_name ON restaurants(name)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_restaurant_name ON reviews(restaurant_name)")
conn.commit()

# Sanity check
print("\nTables created:")
for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(" -", row[0])

print("\nRestaurant count in DB:", cur.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0])
print("Review count in DB:", cur.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])

conn.close()
print(f"\nSaved database to {DB_PATH}")