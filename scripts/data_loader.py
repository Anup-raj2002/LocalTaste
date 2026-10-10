import pandas as pd
import ast
import re
import ftfy

# ---------- 1. Load ----------
df = pd.read_csv("zomato.csv")

# ---------- 2. Clean 'rate' ----------
def clean_rate(val):
    if pd.isna(val):
        return None
    val = val.strip()
    if val in ("NEW", "-"):
        return None
    val = val.replace(" /5", "/5")  # fix the stray-space case
    try:
        return float(val.split("/")[0])
    except (ValueError, IndexError):
        return None

df["rate"] = df["rate"].apply(clean_rate)

# ---------- 3. Clean 'approx_cost' ----------
def clean_cost(val):
    if pd.isna(val):
        return None
    return float(str(val).replace(",", ""))

df["approx_cost(for two people)"] = df["approx_cost(for two people)"].apply(clean_cost)

# ---------- 4. Clean 'location' and 'cuisines' ----------
df["location"] = df["location"].str.strip()
df["cuisines"] = df["cuisines"].str.strip()

# ---------- 5. Drop rows with nulls in critical columns ----------
df = df.dropna(subset=["location", "cuisines", "rate"])

# ---------- 6. Drop the unusable column ----------
df = df.drop(columns=["dish_liked", "phone", "url", "address", "menu_item"])

print("Cleaned shape:", df.shape)
print(df[["name", "rate", "location", "cuisines", "approx_cost(for two people)"]].head())

print("\nUnique locality count:", df["location"].nunique())

counts = df["location"].value_counts()
print(counts)
print("\nLocalities with fewer than 5 restaurants:", (counts < 5).sum())
print("\nLocalities with fewer than 10 restaurants:", (counts < 10).sum())

# ---------- 7. Explode reviews_list into individual reviews ----------
def parse_reviews(row):
    """Turn one restaurant's reviews_list string into a list of dicts."""
    try:
        review_tuples = ast.literal_eval(row["reviews_list"])
    except (ValueError, SyntaxError):
        return []

    parsed = []
    for rating_str, text in review_tuples:
        if rating_str is None:
            continue
        try:
            rating = float(rating_str.replace("Rated", "").strip())
        except (ValueError, AttributeError):
            continue
        parsed.append({
            "restaurant_name": row["name"],
            "locality": row["location"],
            "rating": rating,
            "review_text": text.strip() if isinstance(text, str) else ""
        })
    return parsed

print("\nExploding reviews_list into individual reviews (this is the slow step)...")
all_reviews = []
for i, (_, row) in enumerate(df.iterrows()):
    all_reviews.extend(parse_reviews(row))
    if i % 5000 == 0:
        print(f"Processed {i} / {len(df)} restaurants...")

reviews_df = pd.DataFrame(all_reviews)
print("\nTotal individual reviews extracted:", len(reviews_df))
print(reviews_df.head())
print("\nReviews rating distribution:")
print(reviews_df["rating"].value_counts().sort_index())

# ---------- 8. Clean review text (strip 'RATED' prefix, fix mojibake via ftfy) ----------
def clean_review_text(text):
    text = re.sub(r"^RATED\s*", "", text).strip()
    text = ftfy.fix_text(text)               # repairs repeated/nested mojibake
    text = re.sub(r"\?{2,}", "", text)        # strip runs of lost-emoji '?' marks
    text = re.sub(r"\s+", " ", text).strip()  # collapse newlines/extra whitespace
    return text

print("\nCleaning review text (fixing encoding issues)...")
reviews_df["review_text"] = reviews_df["review_text"].apply(clean_review_text)

# ---------- 9. Derive sentiment labels from rating ----------
def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return None  # 2.5-3.5 dropped as ambiguous

reviews_df["sentiment_label"] = reviews_df["rating"].apply(label_sentiment)

labeled_df = reviews_df.dropna(subset=["sentiment_label"])
print("\nLabeled reviews for training:", len(labeled_df))
print(labeled_df["sentiment_label"].value_counts())
print("\nSample cleaned text:")
print(labeled_df["review_text"].iloc[4])

# ---------- 10. Save cleaned outputs for the next steps ----------
df.to_csv("restaurants_clean.csv", index=False)
labeled_df.to_csv("reviews_labeled.csv", index=False)
print("\nSaved restaurants_clean.csv and reviews_labeled.csv")