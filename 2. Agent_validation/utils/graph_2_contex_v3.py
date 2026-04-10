import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple


class ATE_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entities (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Aspect Term Extraction (ATE) task for Aspect-Based Sentiment Analysis (ABSA). 
    Your task is to:
    - Extract aspect terms from the review (each aspect term must be two words or fewer).
    - Assign a category to each aspect term. categories are: 'staff_attitude', 'service_process', 'food_drink', 'pricing_value', 'location_access', 'facilities_amenities', 'cleanliness_maintenance', 'atmosphere_vibe', 'natural_environment', 'activities_events', 'safety_security', 'policies_governance', 'inclusivity_community', 'overall_experience', 'other'
    - Perform a step-by-step reasoning process explaining how each aspect was identified.
    - Ensure all extracted aspects are unique (no duplicates).
    """

    place_name: str = dspy.InputField(desc="The name of the place being analyzed.")
    review: str = dspy.InputField(desc="A google review for a place.")
    previous_aspects: list[str] = dspy.InputField(desc="A list of aspects proposed by you in the previous iteration if any.")
    revised_aspects_rationale: str = dspy.InputField(desc="A short rational justifying why the supervisor did not accept the proposed aspects.")
    
    aspects: list[Dict[str, str]] = dspy.OutputField(desc="An array of aspects for the place after considering proposed aspects and the rationale of the supervisor. Example Output format: [{'aspect': '<entity>'}, 'category': '<entity>', ...] without any other text or explanation.")




class OTE_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entities (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Opinion Term Extraction (OTE) task for Aspect-Based Sentiment Analysis (ABSA). 
    Your task is to:
    - For each aspect, extract one short opinion phrase (≤3 words, verbatim).
    - If an aspect provided in the input does NOT have a corresponding opinion in the text, do NOT invent (hallucinate) one. Instead, tell the ATE agent what is wrong with the proposed set of aspects.
    """
    
    review: str = dspy.InputField(desc="A google review for the place.")
    aspects: list[str] = dspy.InputField(desc="A list of aspects that are mentioned in the review.")
    previous_opinions: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion proposed by you in the previous iteration, if any.")
    revised_opinions_rationale: str = dspy.InputField(desc="A set of revisions to be applied to the opinion list. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    

    aspects_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspects are reasonable and you can find explicit opinions for them in the review. Else return 'revise'.")
    aspects_rationale: str = dspy.OutputField(desc="A short reason, explaining what is wrong with aspects and why you can not assign an explicit opinion to the aspect.")
    
    aspect_opinions: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion in the review. output schema: [{'aspect': '<entity>', 'opinion': '<opinion>'}, ...] without any other text or explanation.")



class ALSC_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entities (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Aspect-level Sentiment Classification (ALSC) task for Aspect-Based Sentiment Analysis (ABSA). Your task is to:
    - For each aspect, classify the sentiment of the opinion as 'positive', 'negative', or 'neutral'.
    - Avoid duplicate aspects: each aspect in a review must have only one corresponding opinion annotated.
    - If the candidate opinion for an aspect is purely descriptive and not evaluative, do **not** infer or fabricate positive/negative sentiment.
    """

    review: str = dspy.InputField(desc="A google review for a place.")
    aspect_category: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding category.")
    aspect_opinions: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding opinion.")
    previous_sentiments: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by another agent.")
    revised_sentiments_rationale: str = dspy.InputField(desc="A set of revisions to be applied to the sentiments for each aspect-opinion pair. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
        
    opinions_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if all of the proposed opinions are sentiment-bearing and correctly linked to the aspect. Else return 'revise'.")
    opinions_rationale: str = dspy.OutputField(desc="A short reason, explaining what is wrong with opinions (e.g., 'not sentiment bearing', 'fact not opinion').")

    sentiments: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects, their category, their corresponding opinion and the sentiment of the opinion ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect': '<entity>', 'category': '<entity>', 'opinion': '<opinion>', 'sentiment': 'positive' or 'negative' or 'neutral'}, ...] without any other text or explanation.")



class ABSA_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying entities (such as people, products, services, or experiences) mentioned in a text and determining the sentiment (positive, negative, or neutral) expressed toward each entity.
    You serve as a critical supervisor for an Aspect-Based Sentiment Analysis (ABSA) pipeline, which operates in three stages:
    - Aspect Term Extraction (ATE)
    - Opinion Term Extraction (OTE)
    - Aspect-Level Sentiment Classification (ALSC)

    You are given:
    - The original review text.
    - The current list of aspect–opinion–sentiment items.

    Your tasks are to rigorously verify the correctness of each triplet:
    - Ensure there are no duplicates or near-duplicates aspects.
    - Check each aspect, opinion, and sentiment triplet and make sure the sentiment belongs to the opinion that corresponds to the element.
    - Ensure each aspect is categorized correctly.
    - Ensure every opinion is tied to an aspect in the review and is not hallucinated.
    - Ensure every sentiment is correctly classified as positive, negative, or neutral.
    - If you find errors, inconsistencies, or missing items, revise the triplets to produce the most accurate and complete output.
    - Only give mandatory revisions, all at once.
    """

    place_name: str = dspy.InputField(desc="The name of the place being analyzed.")
    review: str = dspy.InputField(desc="The original Google review text for a place.")
    proposed_sentiments: list[dict] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by an agent.")



    aspects_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspects for all aspect-opinion pairs are completely correct and need no edit. Else return 'revise'.")
    opinions_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed opinions for all aspect-opinion pairs are completely correct and need no edit. Else return 'revise'.")
    sentiments_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed sentiments for all aspect-opinion pairs are completely correct and need no edit. Else return 'revise'.")


    aspects_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the aspects. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    opinions_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the opinions corresponding to each aspect. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    sentiments_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the sentiments corresponding to each aspect-opinion pair. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")

