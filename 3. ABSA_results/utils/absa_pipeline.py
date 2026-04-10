import pandas as pd
import json
import ast
import dspy

import threading
from absa_dspy import *

GPT_MODEL = 'openai/gpt-4.1-nano'
GPT_API_KEY = ''


lm = dspy.LM(GPT_MODEL, api_key=GPT_API_KEY, temperature=1.0, max_tokens=16000)
dspy.configure(lm=lm)

ABSA = dspy.Predict(ABSA_Signature)


daily_limit_hit = False
daily_limit_lock = threading.Lock()


def is_daily_limit_error(e):
    """Check if the exception looks like a quota/daily-limit error."""
    msg = str(e).lower()
    # Adjust these substrings if needed after you see the real error text
    return (
        "quota" in msg
        or "exceeded your current quota" in msg
        or "rate limit" in msg and "daily" in msg
    )


def run_absa(review):
    global daily_limit_hit

    with daily_limit_lock:
        if daily_limit_hit:
            # Stop doing real calls once flag is set
            raise RuntimeError("Daily limit already hit")

    try:
        out = ABSA(review=review)
    except Exception as e:
        # If this looks like a daily limit / quota error, flip the global flag
        if is_daily_limit_error(e):
            with daily_limit_lock:
                daily_limit_hit = True
            raise RuntimeError("Daily limit reached") from e
        # Other errors: either re-raise or return empty result
        raise

    items = safe_parse(out.sentiments) or []
    cleaned = []
    for item in items:
        cleaned.append({
            # "aspect": item.get("aspect",""),
            "category": item.get("category",""),
            "sentiment": item.get("sentiment", "neutral")
        })
    return cleaned


def run_absa_row(row): 
    return row["ind"], run_absa(row["text"])


