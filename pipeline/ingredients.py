"""
Ingredient name normalization for pipeline.
- canonicalize_ingredient(name) -> canonical_name for ingredients.canonical_name
- display_name: preserve original or title case (consistent)
"""
import re


def canonicalize_ingredient(name: str) -> str:
    """
    Generate ingredients.canonical_name from raw name.
    - Trim leading/trailing whitespace
    - Lowercase
    - Collapse consecutive spaces to one
    - Remove trailing punctuation (simple)
    - No complex NLP / synonym DB
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!?]+$", "", s).strip()
    return s


def display_name_for_ingredient(name: str) -> str:
    """
    ingredients.display_name: preserve original or title case (consistent).
    Using title case for consistency.
    """
    if not name or not isinstance(name, str):
        return ""
    return name.strip().title()
