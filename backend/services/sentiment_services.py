import joblib
from config import SENTIMENT_MODEL_PATH, TFIDF_VECTORIZER_PATH

print("Loading sentiment model...")
_model = joblib.load(SENTIMENT_MODEL_PATH)
_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
print("Sentiment model ready.")


def predict_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    vec = _vectorizer.transform([text])
    return _model.predict(vec)[0]