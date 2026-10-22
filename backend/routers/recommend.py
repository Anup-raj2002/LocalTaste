import time
from fastapi import APIRouter
from pydantic import BaseModel
from services import recommender_services, gemini_service

router = APIRouter()


class RecommendRequest(BaseModel):
    locality: str | None = None
    liked_names: list
    top_k: int = 5
    use_gemini: bool = True


class ParseRequest(BaseModel):
    text: str


@router.get("/localities")
def localities():
    return {"localities": recommender_services.get_localities()}


@router.get("/restaurants")
def restaurants(locality: str = ""):
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
    # Pass only specific neighborhoods to Gemini so it returns null when user says "Bangalore" or no locality
    valid_localities = [
        loc for loc in recommender_services.get_localities()
        if loc != "All Bangalore (City-wide)"
    ]
    valid_cuisines = recommender_services.get_valid_cuisines()

    parsed = gemini_service.parse_preferences(req.text, valid_localities, valid_cuisines)
    t1 = time.perf_counter()

    if parsed.get("error"):
        return {
            "error": "The language service is unavailable right now. Please try again in a moment.",
            "parsed": parsed,
        }

    # If no specific locality was mentioned (or user just said Bangalore), use Universal City-wide KNN
    selected_locality = parsed.get("locality")
    if not selected_locality:
        parsed["locality"] = "All Bangalore (City-wide)"
        parsed["is_city_wide"] = True
    else:
        parsed["is_city_wide"] = False

    cuisines = parsed.get("cuisines") or []
    results = recommender_services.recommend_by_cuisines(
        selected_locality, cuisines, top_k=5
    )

    # Display tag in UI when user asked for general good food without specifying a cuisine
    if not cuisines:
        parsed["cuisines"] = ["Chef's Popular Picks"]

    t2 = time.perf_counter()

    if use_gemini_explain and results:
        explanations = gemini_service.explain_recommendations(cuisines, results)
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
