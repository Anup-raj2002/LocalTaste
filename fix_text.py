import pandas as pd
import re

print("Loading reviews_labeled.csv...")
df = pd.read_csv("reviews_labeled.csv")

def final_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\?{1,}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text column...")
df["review_text"] = df["review_text"].apply(final_clean)

df = df[df["review_text"].str.len() > 5]

print("Final labeled review count:", len(df))
print(df["sentiment_label"].value_counts())
print("\nSample cleaned text:")
print(df["review_text"].iloc[4])

df.to_csv("reviews_labeled.csv", index=False)
print("\nOverwrote reviews_labeled.csv with fully cleaned text")