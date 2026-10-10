import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# ---------- Paths (matches the current project structure) ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # Localtaste/
REVIEWS_CSV = BASE_DIR / "data" / "processed" / "reviews_labeled.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("Loading reviews_labeled.csv...")
df = pd.read_csv(REVIEWS_CSV)
df = df.dropna(subset=["review_text", "sentiment_label"])

X = df["review_text"]
y = df["sentiment_label"]

# ---------- Train/test split, stratified by label ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", len(X_train), "Test size:", len(X_test))

# ---------- TF-IDF vectorization ----------
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", min_df=5)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


def evaluate_model(name, model):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    print("\nConfusion Matrix (rows=actual, cols=predicted), labels=[positive, negative]:")
    print(confusion_matrix(y_test, y_pred, labels=["positive", "negative"]))

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred))

    return model


# ---------- Model 1: plain MultinomialNB (baseline) ----------
nb_model = evaluate_model("MultinomialNB (baseline)", MultinomialNB())

# ---------- Model 2: ComplementNB (designed for imbalanced text data) ----------
cnb_model = evaluate_model("ComplementNB (imbalance-corrected)", ComplementNB())

print("\nCompare the 'negative' row recall between the two models above.")
print("ComplementNB should show noticeably better negative-class recall.")

# ---------- Save the chosen model (ComplementNB) + vectorizer for reuse ----------
print("\nSaving chosen model (ComplementNB) and TF-IDF vectorizer to disk...")
joblib.dump(cnb_model, MODELS_DIR / "sentiment_model.pkl")
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
print(f"Saved sentiment_model.pkl and tfidf_vectorizer.pkl to {MODELS_DIR}")