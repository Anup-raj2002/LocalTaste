import json
from config import GEMINI_API_KEY

_enabled = bool(GEMINI_API_KEY)
_MODEL_NAME = "gemini-3.5-flash-lite"

if _enabled:
    from google import genai
    from google.genai import types
    # 20 s cap per call so a slow or rate-limited request can't hang the app
    _client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=20000),
    )


def _strip_fences(text: str) -> str:
    return text.strip().replace("```json", "").replace("```", "").strip()


def _fallback(preferred_cuisines, n):
    msg = f"Recommended because it matches your preference for {', '.join(preferred_cuisines)}."
    return [msg] * n


def explain_recommendation(preferred_cuisines, recommended_name, cuisines) -> str:
    """Single-restaurant explanation (kept for the /recommend endpoint)."""
    out = explain_recommendations(
        preferred_cuisines, [{"name": recommended_name, "cuisines": cuisines}]
    )
    return out[0]


def explain_recommendations(preferred_cuisines, recs) -> list:
    """One API call for ALL recommendations. Returns one sentence per rec, in order.
    Falls back to a plain sentence if the API is off, slow, or returns bad output."""
    if not recs:
        return []
    fallback = _fallback(preferred_cuisines, len(recs))
    if not _enabled:
        return fallback

    items = [{"name": r["name"], "cuisines": r["cuisines"]} for r in recs]
    prompt = (
        f"A user said they like: {', '.join(preferred_cuisines)}.\n"
        f"For each restaurant below, write ONE short, friendly sentence explaining why it "
        f"suits them. Do not mention TF-IDF, cosine similarity, or any technical terms.\n"
        f"Restaurants: {json.dumps(items)}\n\n"
        f"Respond with ONLY a JSON array of exactly {len(recs)} strings, in the same order, "
        f"with no markdown and no extra text."
    )
    try:
        response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        arr = json.loads(_strip_fences(response.text))
        if isinstance(arr, list) and len(arr) == len(recs):
            return [str(x).strip() for x in arr]
    except Exception:
        pass
    return fallback


def parse_preferences(user_text: str, valid_localities: list, valid_cuisines: list):
    if not _enabled:
        return {"locality": None, "cuisines": [], "error": "Gemini not configured"}

    prompt = (
        f"A user wrote this about where they are and what they like to eat:\n"
        f"\"{user_text}\"\n\n"
        f"Valid localities (pick the closest match, or null if none fits): {valid_localities}\n"
        f"Valid cuisines (pick 1-4 closest matches from this list only): {valid_cuisines}\n\n"
        f"Respond with ONLY valid JSON, no explanation, no markdown, in this exact shape:\n"
        f'{{"locality": "<one value from the valid list or null>", "cuisines": ["<value1>", "<value2>"]}}'
    )
    try:
        response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        parsed = json.loads(_strip_fences(response.text))
        return {
            "locality": parsed.get("locality"),
            "cuisines": parsed.get("cuisines", []),
        }
    except Exception as e:
        return {"locality": None, "cuisines": [], "error": str(e)}