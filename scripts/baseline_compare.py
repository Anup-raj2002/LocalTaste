import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "reviews_labeled.csv")
df = df.dropna(subset=["review_text", "sentiment_label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"],
    df["sentiment_label"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment_label"]
)

print("Train size:", len(X_train), "Test size:", len(X_test))

vec = TfidfVectorizer(max_features=10000, stop_words="english", min_df=5)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

models = {
    "MultinomialNB": MultinomialNB(),
    "ComplementNB (selected)": ComplementNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Logistic Regression (balanced)": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(),
    "Linear SVM (balanced)": LinearSVC(class_weight="balanced"),
}

rows = []

for name, m in models.items():
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)

    acc = accuracy_score(y_test, pred)

    p, r, f, _ = precision_recall_fscore_support(
        y_test,
        pred,
        labels=["negative"],
        zero_division=0
    )

    wp, wr, wf, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        pred,
        labels=["positive", "negative"]
    )

    rows.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Weighted F1": round(wf * 100, 2),
        "Neg precision": round(p[0] * 100, 1),
        "Neg recall": round(r[0] * 100, 1),
        "Neg F1": round(f[0] * 100, 1),
        "Confusion [[TP_pos,FN_pos],[FN_neg,TP_neg]]": cm.tolist(),
    })

    print(f"done: {name}")

out = pd.DataFrame(rows)

pd.set_option(
    "display.width", 250,
    "display.max_columns", None,
    "display.max_colwidth", 80
)

print("\n" + out.to_string(index=False))

out.to_csv(BASE_DIR / "baseline_results.csv", index=False)

print("\nSaved baseline_results.csv")