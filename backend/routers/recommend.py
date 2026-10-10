import time
from fastapi import APIRouter
from pydantic import BaseModel
from services import recommender_services, gemini_service

router = APIRouter()


class RecommendRequest(BaseModel):
    locality: str
    liked_names: list
    top_k: int = 5
    use_gemini: bool = True


class ParseRequest(BaseModel):
    text: str


@router.get("/localities")
def localities():
    return {"localities": recommender_services.get_localities()}


@router.get("/restaurants")
def restaurants(locality: str):
    return {"restaurants": recommender_services.get_restaurants_in_locality(locality)}


@router.post("/recommend")
def get_recommendations(req: RecommendRequest):
    results = recommender_services.recommend(req.locality, req.liked_names, req.top_k)
    if req.use_gemini and results:
        explanations = gemini_service.explain_recommendations(req.liked_names, results)
        for r, e in zip(results, explanations):
            r["explanation"] = e
    return {"recommendations": results}


@router.post("/parse-and-recommend")
def parse_and_recommend(req: ParseRequest, use_gemini_explain: bool = True):
    t0 = time.perf_counter()
    valid_localities = recommender_services.get_localities()
    valid_cuisines = recommender_services.get_valid_cuisines()

    parsed = gemini_service.parse_preferences(req.text, valid_localities, valid_cuisines)
    t1 = time.perf_counter()

    if not parsed.get("locality"):
        if parsed.get("error"):
            msg = "The language service is unavailable right now. Please try again in a moment."
        else:
            msg = ("I couldn't tell which area of Bangalore you're in. "
                   "Try adding a locality, for example: 'in Indiranagar' or 'near Koramangala'.")
        return {"error": msg, "parsed": parsed}

    results = recommender_services.recommend_by_cuisines(
        parsed["locality"], parsed["cuisines"], top_k=5
    )
    t2 = time.perf_counter()

    if use_gemini_explain and results:
        explanations = gemini_service.explain_recommendations(parsed["cuisines"], results)
        for r, e in zip(results, explanations):
            r["explanation"] = e
    t3 = time.perf_counter()

    timing = {
        "parse_ms": round((t1 - t0) * 1000),
        "rank_ms": round((t2 - t1) * 1000),
        "explain_ms": round((t3 - t2) * 1000),
        "total_ms": round((t3 - t0) * 1000),
    }
    print("parse-and-recommend timing:", timing)
    return {"parsed": parsed, "recommendations": results, "timing": timing}