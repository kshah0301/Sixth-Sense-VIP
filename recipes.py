import json
import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env file.")

client = genai.Client(api_key=api_key)

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _strip_trailing_commas(s: str) -> str:
    # Remove JSON trailing commas like {"a":1,} or [1,2,]
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s


def _loads_json_relaxed(text: str) -> dict:
    """
    Gemini responses sometimes wrap JSON in ```json fences or include extra text.
    This extracts the JSON object and parses it, with a small trailing-comma fixup.
    """
    if not text or not text.strip():
        raise ValueError("Empty model response (expected JSON).")

    t = text.strip()
    m = _FENCED_JSON_RE.search(t)
    if m:
        candidate = m.group(1).strip()
    else:
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        candidate = t[start : end + 1].strip()

    candidate = _strip_trailing_commas(candidate)
    return json.loads(candidate)


def get_recipe_ingredients(meal_description: str) -> list[str]:
    prompt = f"""
    The user wants to cook: "{meal_description}".

    1. Infer a simple, practical recipe.
    2. Return a JSON object with a single key "ingredients",
       whose value is a list of 5–12 grocery items that correspond
       to United States supermarket product labels that can be found in Open Food facts with brand names.

    Only output the JSON, nothing else.
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = resp.text
    print(text)
    data = _loads_json_relaxed(text)
    ingredients = data.get("ingredients", [])
    if not isinstance(ingredients, list):
        raise ValueError("Model returned JSON but 'ingredients' was not a list.")
    return [str(x) for x in ingredients if str(x).strip()]
