import pandas as pd
import json
import ast

import dspy
from typing_extensions import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple


def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []


class SentimentItem(TypedDict):
    # aspect: str
    category: Literal['staff_attitude', 'service_process', 'food_drink', 'pricing_value', 'location_access', 'facilities_amenities', 'cleanliness_maintenance', 'atmosphere_vibe', 'natural_environment', 'activities_events', 'safety_security', 'policies_governance', 'inclusivity_community', 'overall_experience', 'other']    
    sentiment: Literal['positive', 'negative', 'neutral']



# class ABSA_Signature(dspy.Signature):
#     """
#     Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entities (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity.
#     You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA) task. 
#     Your task is to:
#     - Extract aspect terms from the review (explicit or implicit) and assign a category to each aspect term. categories are: 'staff_attitude', 'service_process', 'food_drink', 'pricing_value', 'location_access', 'facilities_amenities', 'cleanliness_maintenance', 'atmosphere_vibe', 'natural_environment', 'activities_events', 'safety_security', 'policies_governance', 'inclusivity_community', 'overall_experience', 'other'
#     - Avoid assigning items to the “other” category unless absolutely necessary.
#     - For each aspect category, classify the sentiment regarding that aspect as 'positive', 'negative', or 'neutral'.
#     - Ensure all extracted aspects are unique (no duplicates).
#     """
#     review: str = dspy.InputField(desc="A google review for a place.")

#     sentiments: list[SentimentItem] = dspy.OutputField(desc="An array of JSONs containing aspects, their category, and the sentiment towards the aspect ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect': '<entity>', 'category': '<entity>', 'sentiment': 'positive' or 'negative' or 'neutral'}, ...] without any other text or explanation.")


class ABSA_Signature(dspy.Signature):
    """
    Extract aspects in the review. For each aspect, assign one category:
    staff_attitude, service_process, food_drink, pricing_value, location_access, facilities_amenities, cleanliness_maintenance, atmosphere_vibe, natural_environment, activities_events, safety_security, policies_governance, inclusivity_community, overall_experience, other.
    For each category, classify the sentiment regarding that category as 'positive' | 'negative' | 'neutral'.
    Avoid “other” when possible.
    Output unique items only.
    """
    review: str = dspy.InputField(desc='A google review for a place.')

    sentiments: list[SentimentItem] = dspy.OutputField(
        desc="List of {'category', 'sentiment'} only."
    )