from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (Localtaste/)

DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = DATA_PROCESSED / "localtaste.db"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CORS_ORIGINS = ["http://localhost:5173"]