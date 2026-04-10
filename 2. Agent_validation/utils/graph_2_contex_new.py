import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple


class ATE_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Aspect Term Extraction (ATE) task for Aspect-Based Sentiment Analysis (ABSA). Your task is to:
    - Extract explicit aspect terms mentioned in the review (≤2 words each).
    - Provide a reasoning process for how you identified the entities.
    - Only consider items which the aspect and sentiment are explicitly stated in the review.
    """
    
    review: str = dspy.InputField(desc="A google review for a place.")
    previous_aspects: list[str] = dspy.InputField(desc="A list of aspects proposed by you in the previous iteration if any.")
    revised_aspects_rationale: str = dspy.InputField(desc="A short rational justifying why the supervisor did not accept the proposed aspects.")
    
    aspects: list[str] = dspy.OutputField(desc="An array of aspects for the place after considering proposed aspects and the rationale of the supervisor. Example Output format: [{'aspect': '<entity>'}, ...] without any other text or explanation.")




class OTE_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Opinion Term Extraction (OTE) task for Aspect-Based Sentiment Analysis (ABSA). Your task is to:
    - For each aspect, extract one short opinion phrase (≤3 words, verbatim).
    - Provide a reasoning process for how you identified the opinion phrases.
    - Only consider items which the aspect and opinion are explicitly stated in the review.
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    aspects: list[str] = dspy.InputField(desc="A list of aspects.")
    previous_opinions: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion proposed by you in the previous iteration if any.")
    # revised_opinions = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion in the review proposed by another agent.")
    revised_opinions_rationale: str = dspy.InputField(desc="A set of revisions to be applied to the opinion list. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    
    aspect_opinions: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion in the review. output schema: [{'aspect': '<entity>', 'opinion': '<opinion>'}, ...] without any other text or explanation.")





class ALSC_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator for the Aspect-level Sentiment Classification (ALSC) task for Aspect-Based Sentiment Analysis (ABSA). Your task is to:
    - For each aspect, classify the sentiment of the opinion as 'positive', 'negative', or 'neutral'.
    - Provide a reasoning process for how you identified the sentiment of the opinion.
    - Only consider items which the aspect and opinion are explicitly stated in the review.
    Only use explicit evidence. Output JSON array:
        [{"aspect": '<entity>', "opinion": '<opinion>', "sentiment": 'positive' or 'negative' or 'neutral'}, ...]
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    aspect_opinions: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding opinion. ")
    previous_sentiments: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by another agent.")
    revised_sentiments_rationale: str = dspy.InputField(desc="A set of revisions to be applied to the sentiments for each aspect-opinion pair. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    
    sentiments: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect': '<entity>', 'opinion': '<opinion>', 'sentiment': 'positive' or 'negative' or 'neutral'}, ...] without any other text or explanation.")




class ABSA_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are a critical supervisor for an Aspect-Based Sentiment Analysis (ABSA) pipeline.
    The pipeline has three stages:
    - Aspect Term Extraction (ATE)
    - Opinion Term Extraction (OTE)
    - Aspect-Level Sentiment Classification (ALSC)

    You are given:
    - The original review text.
    - The current list of aspect–opinion–sentiment items.

    Your job is to:
    1. First check the **ATE level** (aspects only).
       - All proposed aspects must be explicitly stated in the review with some sentiment.
       - Each aspect must be no more than 2 words.
       - There must be no duplicates or near-duplicates.
       - All important aspects mentioned in the review (with sentiment) should be included.

    2. If the aspects are acceptable, then check the **OTE level** (aspect–opinion pairs).
       - All proposed aspect–opinion pairs must be explicitly stated in the review with a sentiment.
       - Each opinion phrase must be no more than 3 words.
       - There must be no duplicates or near-duplicates.
       - All important aspect–opinion pairs mentioned in the review should be included.
       - Keep only one best opinion phrase per aspect.

    3. If both ATE and OTE are acceptable, then check the **ALSC level** (sentiments).
       - All proposed sentiment labels must be explicitly supported by the review (no hallucinations).
       - Sentiments must be consistent with the evidence in the text.
       - Avoid duplicates or near-duplicates.
       - All important sentiment-bearing aspect–opinion pairs should be covered.

    """
    review: str = dspy.InputField(desc="The original Google review text for a place.")
    proposed_sentiments: list[dict] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by an agent.")

    aspects_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspects for all aspect-opinion pairs is completely correct and needs no edit. Else return 'revise'.")
    opinions_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed opinions for all aspect-opinion pairs is completely correct and needs no edit. Else return 'revise'.")
    sentiments_decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed sentiments for all aspect-opinion pairs is completely correct and needs no edit. Else return 'revise'.")


    aspects_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the aspects. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    opinions_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the opinions. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    sentiments_rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the sentiments. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")

