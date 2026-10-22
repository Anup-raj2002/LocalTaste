import json
import time
from config import GEMINI_API_KEY

_enabled = bool(GEMINI_API_KEY)
_MODEL_NAME = "gemini-3.5-flash-lite"

if _enabled:
    from google import genai
    from google.genai import types
    # 30 s cap per call so a slow or rate-limited request doesn't hang the app
    _client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=30000),
    )


def _call_gemini(prompt: str) -> str:
    """Call gemini-3.5-flash-lite with retry on temporary 503 UNAVAILABLE spikes."""
    last_err = None
    for attempt in range(4):
        try:
            response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise last_err


def _strip_fences(text: str) -> str:
    return text.strip().replace("```json", "").replace("```", "").strip()


def _fallback(preferred_cuisines, n):
    if preferred_cuisines:
        msg = f"Recommended because it matches your preference for {', '.join(preferred_cuisines)}."
    else:
        msg = "One of the top-rated popular dining spots recommended for you."
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
    pref_text = ", ".join(preferred_cuisines) if preferred_cuisines else "popular, highly rated local favourites"
    prompt = (
        f"A user is looking for: {pref_text}.\n"
        f"For each restaurant below, write ONE short, friendly sentence explaining why it "
        f"is a great choice for them. Do not mention TF-IDF, cosine similarity, or any technical terms.\n"
        f"Restaurants: {json.dumps(items)}\n\n"
        f"Respond with ONLY a JSON array of exactly {len(recs)} strings, in the same order, "
        f"with no markdown and no extra text."
    )
    try:
        raw = _call_gemini(prompt)
        arr = json.loads(_strip_fences(raw))
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
        f"Valid localities (pick the closest specific Bangalore neighborhood match from this list "
        f"even if misspelled, or null if the user only mentions Bangalore/city-wide or doesn't mention a specific neighborhood): {valid_localities}\n"
        f"Valid cuisines (pick 1-4 closest matches from this list if the user mentions cuisines/dishes/cravings; "
        f"if the user asks generally for 'good food', 'best places', or doesn't specify any cuisine, return an empty list []): {valid_cuisines}\n\n"
        f"Respond with ONLY valid JSON, no explanation, no markdown, in this exact shape:\n"
        f'{{"locality": "<one value from the valid list or null>", "cuisines": ["<value1>", "<value2>"]}}'
    )
    try:
        raw = _call_gemini(prompt)
        parsed = json.loads(_strip_fences(raw))
        return {
            "locality": parsed.get("locality"),
            "cuisines": parsed.get("cuisines", []),
        }
    except Exception as e:
        return {"locality": None, "cuisines": [], "error": str(e)}


def analyze_review_with_gemini(raw_text: str) -> dict | None:
    """Single Gemini call using gemini-3.5-flash-lite that:
    1. Rewrites/normalizes customer review text (fixing spelling errors and converting negated
       phrases like 'didnt liked' into direct sentiment words like 'disliked') so the unigram
       TF-IDF + ComplementNB model can classify it accurately.
    2. Generates a dynamic appreciation or apology response with suggestions.
    """
    if not _enabled or not raw_text or not raw_text.strip():
        return None

    prompt = (
        f"You are the language assistant and friendly host for LocalTaste in Bangalore.\n"
        f"A customer submitted this feedback:\n"
        f"\"{raw_text}\"\n\n"
        f"Return ONLY valid JSON (no markdown, no extra text) with two fields:\n"
        f"1. \"normalized_text\": Rewrite the customer's feedback with corrected spelling and grammar, "
        f"and express any negated phrases (such as \"didnt liked\" or \"not good\") using direct single-word "
        f"sentiment equivalents (such as \"disliked disappointing\" or \"bad\") so a unigram Bag-of-Words "
        f"classifier accurately captures their true sentiment.\n"
        f"2. \"message\": Write a warm, natural 2-3 sentence response to the customer:\n"
        f"   - If the customer's feedback is positive: Greet them warmly, thank them for taking the time to share "
        f"their feedback, express appreciation for what they enjoyed, and suggest a dish or spot to try next time.\n"
        f"   - If the customer's feedback is negative: Thank them for taking the time to share their feedback, "
        f"sincerely apologize for their disappointing experience (referencing what they mentioned), and suggest "
        f"how we can make it right or a better option next time.\n\n"
        f'Respond with ONLY valid JSON in this exact format:\n'
        f'{{"normalized_text": "...", "message": "..."}}'
    )
    try:
        raw = _call_gemini(prompt)
        cleaned = _strip_fences(raw)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None
