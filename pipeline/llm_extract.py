"""
LLM extraction: call OpenAI with structured output (ingredients + reasoning).
Output structure matches the required JSON schema.
"""
import json
import os
from typing import List, Dict, Any

# Optional: openai only when used
def _get_client():
    from openai import OpenAI
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        import config as _cfg
        api_key = getattr(_cfg, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
    except ImportError:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set; create config.py with OPENAI_API_KEY or set env")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://aiapi-prod.stanford.edu/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_extraction_schema() -> dict:
    """JSON schema for LLM response (strict, machine-readable)."""
    return {
        "format": {
            "type": "json_schema",
            "name": "results",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "item_id": {"type": "string"},
                                "item_name": {"type": "string"},
                                "restaurant_id": {"type": "string"},
                                "ingredients": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "name": {"type": "string"},
                                            "confidence": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                        },
                                        "required": ["name", "confidence"],
                                    },
                                },
                                "reasoning": {"type": "string"},
                            },
                            "required": [
                                "item_id",
                                "item_name",
                                "restaurant_id",
                                "ingredients",
                                "reasoning",
                            ],
                        },
                    }
                },
                "required": ["results"],
            },
        }
    }


def get_assistant_instruction() -> str:
    """Prompt for ingredient extraction (matches notebook + singular ingredient names)."""
    return """
You are a food menu analysis assistant.
Task:
Decompose each restaurant menu item into its likely ingredients.

Input:
You will be given a list of menu items. Each item contains:
- item_id
- item_name
- restaurant_id
- optional description

Output requirements (STRICT):
- Output MUST be machine-readable.
- Output MUST follow the provided JSON schema exactly.
- Do NOT include extra commentary, explanations, or markdown.
- Do NOT invent fields that are not requested.
- Each ingredient must be a short noun phrase.
- Each ingredient name must be in singular form (e.g., "chicken wing" not "chicken wings").
- Ingredients must be ordered from most prominent to least prominent.
- Each ingredient must include a confidence score between 0 and 1.
- The confidence represents how likely the ingredient is to be present in the dish.

For each menu item, infer:
1) The most likely ingredients commonly used in this dish
2) A confidence score (0–1) for each ingredient
3) The reasoning explaining WHY those ingredients are present, based on:
   - culinary conventions
   - dish name semantics
   - regional cooking practices
   - typical restaurant preparation methods

Constraints:
- If an ingredient is uncertain, still include it but give it a lower confidence.
- Do NOT include cooking utensils or heat
- Avoid overly granular items (e.g., "Himalayan pink salt" → "salt").
""".strip()


def extract_ingredients(
    menu_items_text: str,
    model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """
    Call LLM and return list of item dicts with item_id, item_name, restaurant_id,
    ingredients (list of {name, confidence}), reasoning.
    """
    client = _get_client()
    schema = get_extraction_schema()
    instruction = get_assistant_instruction()
    response = client.responses.create(
        model=model,
        input=menu_items_text,
        instructions=instruction,
        text=schema,
    )
    raw = response.output_text
    if not raw:
        raise ValueError("LLM returned empty output_text")
    data = json.loads(raw)
    if "results" not in data or not isinstance(data["results"], list):
        raise ValueError("LLM output missing or invalid 'results' array")
    return data["results"]
