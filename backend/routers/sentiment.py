from fastapi import APIRouter
from pydantic import BaseModel
from services import sentiment_services, gemini_service

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.post("/sentiment")
def classify_sentiment(req: SentimentRequest):
    if not req.text or not req.text.strip():
        return {"label": "unknown", "message": ""}

    # Gemini normalizes spelling/negation and generates the appreciation or apology message in one fast call
    gemini_result = gemini_service.analyze_review_with_gemini(req.text)

    text_for_model = (
        gemini_result.get("normalized_text")
        if gemini_result and gemini_result.get("normalized_text")
        else req.text
    )

    # Classification is performed by the trained ComplementNB + TF-IDF model
    label = sentiment_services.predict_sentiment(text_for_model)

    message = gemini_result.get("message", "") if gemini_result else ""

    return {
        "label": label,
        "message": message,
    }
