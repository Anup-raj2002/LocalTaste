import pandas as pd
import ftfy
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Localtaste/
RAW_CSV = BASE_DIR / "data" / "raw" / "zomato.csv"
CLEAN_CSV = BASE_DIR / "data" / "processed" / "restaurants_clean.csv"

print("Loading original zomato.csv for dish_liked column...")
raw = pd.read_csv(RAW_CSV)
raw["location"] = raw["location"].str.strip()

dish_data = raw[["name", "location", "dish_liked"]].drop_duplicates(subset=["name", "location"], keep="first")

print("Loading current restaurants_clean.csv...")
restaurants = pd.read_csv(CLEAN_CSV)

merged = restaurants.merge(dish_data, on=["name", "location"], how="left")
print("Restaurants with dish_liked available:", merged["dish_liked"].notna().sum(), "/", len(merged))

# ---------- Fix mojibake in name and location (same issue we fixed in review_text) ----------
def clean_text_field(text):
    if not isinstance(text, str):
        return text
    text = ftfy.fix_text(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning name and location encoding...")
merged["name"] = merged["name"].apply(clean_text_field)
merged["location"] = merged["location"].apply(clean_text_field)

merged.to_csv(CLEAN_CSV, index=False)
print("Saved restaurants_clean.csv with dish_liked restored and encoding fixed")