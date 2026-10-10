from fastapi import APIRouter
from pydantic import BaseModel
from services import sentiment_services

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.post("/sentiment")
def classify_sentiment(req: SentimentRequest):
    label = sentiment_services.predict_sentiment(req.text)
    return {"label": label}