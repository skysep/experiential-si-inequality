import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple

def keep_new(old, new):
    """Always prefer the new value when present."""
    return new if new is not None else old

def keep_old_if_accept(old: str | None, new: str | None):
    """For decisions: keep old if it's 'accept' (case/space-insensitive), else take new."""
    if isinstance(old, str) and old.strip().lower() == "accept":
        return old
    return new if new is not None else old

def keep_old_if_nonempty_else_new(old: str | None, new: str | None):
    """For rationale strings: keep old if it is non-empty after strip, else take new."""
    if isinstance(old, str) and old.strip() != "":
        return old
    return new if new is not None else old

def max_counter(old: int | None, new: int | None):
    """Merge counters by taking the max (None-safe)."""
    if old is None:
        return new
    if new is None:
        return old
    return max(old, new)

def absa_fields_merge(old: List[Dict[str, str]] | None, new: List[Dict[str, str]] | None):
    """
    Keep old if it already has items; otherwise take new.
    Ensures the merged value is a list (never None).
    """
    old_list = old or []
    new_list = new or []
    return old_list if len(old_list) > 0 else new_list

def overall_sentiment_merge(
    old: Literal["positive", "negative", "neutral"] | None,
    new: Literal["positive", "negative", "neutral"] | None
) -> Literal["positive", "negative", "neutral"] | None:
    """
    Keep old if it's decisively not 'neutral'; otherwise prefer new.
    If both are None, returns None (caller should normalize later).
    """
    if isinstance(old, str) and old.strip().lower() in {"positive", "negative"}:
        return old  # keep decisive old
    return new if new is not None else old

AllowedSent = Literal["positive", "negative", "neutral"]

def normalize_sent(s: Optional[str]) -> Optional[AllowedSent]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    return s if s in {"positive", "negative", "neutral"} else None

def decisive_first(old: Optional[AllowedSent], new: Optional[AllowedSent]) -> Optional[AllowedSent]:
    """
    Keep old if it's decisive (positive/negative). Otherwise take new if valid.
    """
    old_n = normalize_sent(old)
    new_n = normalize_sent(new)
    if old_n in {"positive", "negative"}:
        return old_n
    return new_n if new_n is not None else old_n

def merge_aspects(old: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    """
    Union + stable order for aspect strings. If old has items, keep them and append any new unseen.
    """
    old_list = list(old or [])
    new_list = list(new or [])
    return old_list if len(old_list) > 0 else new_list

def merge_opinions(old: Optional[List[Dict[str, str]]], new: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Union + stable order for opinions. Uses ('aspect','opinion') as identity.
    """
    old_list = list(old or [])
    new_list = list(new or [])

    return old_list if len(old_list) > 0 else new_list

def merge_sentiment_items(
    old: Optional[List["SentimentItem"]],
    new: Optional[List["SentimentItem"]],
) -> List["SentimentItem"]:
    """
    Union + stable order for SentimentItem.
    Identity: (aspect, opinion).
    If duplicate pair exists, prefer the item with decisive sentiment (pos/neg) over neutral.
    """
    old_list = list(old or [])
    new_list = list(new or [])
    return old_list if len(old_list) > 0 else new_list

def parse_aspect_list(v: Any) -> List[str]:
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                aspect = str(item.get("aspect", "")).strip()
                category = str(item.get("category", "")).strip()
                out.append({"aspect": aspect, "category": category})
            elif isinstance(item, str):
                # maybe a dict encoded as string
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        aspect = str(parsed.get("aspect", "")).strip()
                        category = str(parsed.get("category", "")).strip()
                        out.append({"aspect": aspect, "category": category})
                except Exception:
                    pass
        return out
    
    if isinstance(v, dict):
        return [{
            "aspect": str(v.get("aspect", "")).strip(),
            "category": str(v.get("category", "")).strip()
        }]

    if isinstance(v, str):
        s = v.strip()

        # remove code fences
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]

        # Try JSON
        try:
            parsed = json.loads(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # fallback: treat as one opinion with missing aspect
        return [{"aspect": "", "category": s}]

    # Unknown type
    return []

def parse_opinions_list(v: Any) -> List[Dict[str, str]]:
    """
    Parse/normalize LLM outputs into a list of {"aspect":..., "opinion":...}.
    Accepts:
      - list of dicts
      - dict
      - stringified list
      - JSON or Python literal
      - code fences
      - junk (ignored)
    Always returns a clean list[dict].
    """

    # Handle None
    if v is None:
        return []

    # If it's already a list
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                aspect = str(item.get("aspect", "")).strip()
                opinion = str(item.get("opinion", "")).strip()
                out.append({"aspect": aspect, "opinion": opinion})
            elif isinstance(item, str):
                # maybe a dict encoded as string
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        aspect = str(parsed.get("aspect", "")).strip()
                        opinion = str(parsed.get("opinion", "")).strip()
                        out.append({"aspect": aspect, "opinion": opinion})
                except Exception:
                    pass
        return out

    # If it's a dict
    if isinstance(v, dict):
        return [{
            "aspect": str(v.get("aspect", "")).strip(),
            "opinion": str(v.get("opinion", "")).strip()
        }]

    # If it is a string
    if isinstance(v, str):
        s = v.strip()

        # remove code fences
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]

        # Try JSON
        try:
            parsed = json.loads(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # fallback: treat as one opinion with missing aspect
        return [{"aspect": "", "opinion": s}]

    # Unknown type
    return []


def parse_sentiments_list(v: Any) -> List[Dict[str, str]]:
    """
    Parse anything into a list of SentimentItem dicts:
        {"aspect":..., "opinion":..., "sentiment": ...}
    sentiment must be one of {'positive','negative','neutral'}.
    """

    allowed = {"positive", "negative", "neutral"}

    # Handle None
    if v is None:
        return []

    def clean_item(item: Dict[str, Any]) -> Dict[str, str]:
        aspect = str(item.get("aspect", "")).strip()
        category = str(item.get("category", "")).strip()
        opinion = str(item.get("opinion", "")).strip()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if sentiment not in allowed:
            sentiment = "neutral"
        return {
            "aspect": aspect,
            "category": category,
            "opinion": opinion,
            "sentiment": sentiment
        }

    # If it's already a list
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                out.append(clean_item(item))
            elif isinstance(item, str):
                # maybe a dict as string
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        out.append(clean_item(parsed))
                except Exception:
                    pass
        return out

    # If it's a dict
    if isinstance(v, dict):
        return [clean_item(v)]

    # If string
    if isinstance(v, str):
        s = v.strip()

        # strip code fences
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]

        # Try JSON
        try:
            parsed = json.loads(s)
            return parse_sentiments_list(parsed)
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            return parse_sentiments_list(parsed)
        except Exception:
            pass

        # Unknown string → treat as "neutral" with no aspect/opinion
        return [{"aspect": "", "opinion": s, "sentiment": "neutral"}]

    # Unknown type
    return []