import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple




class ABSA_Do(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are an expert annotator performing Aspect-Based Sentiment Analysis (ABSA) on google reviews for places. 
    Your task is to:
    - Extract aspects and their corresponding opinions and sentiments.
    - Provide a reasoning process for how you identified the aspects, opinions, and sentiments.
    - Only consider items which the aspect and opinion are explicitly stated in the review with some sentiment.
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    previous_absa_fields: list[Dict[str, str]] = dspy.InputField(desc="A list of aspects and their corresponding opinions and sentiments proposed by you in the previous iteration if any.")
    revised_absa_rationale: str = dspy.InputField(desc="A set of revisions to be applied to the aspects and their corresponding opinions and sentiments. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
    
    absa_fields: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion and sentiment in the review. output schema: [{'aspect': '<entity>', 'opinion': '<opinion>', 'sentiment': '<sentiment>'}, ...] without any other text or explanation.")




class ABSA_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].
    You are a critical supervisor for an Aspect-Based Sentiment Analysis (ABSA) pipeline.
    You are given:
    - The original review text.
    - The current list of aspects, opinions, and sentiments.

    Your job is to:
    1. First check the **aspects level**.
       - All proposed aspects must be explicitly stated in the review with some sentiment.
       - Each aspect must be no more than 2 words.
       - There must be no duplicates or near-duplicates.
       - All important aspects mentioned in the review (with sentiment) should be included.

    2. If the aspects are acceptable, then check the **opinions level**.
       - All proposed aspect–opinion pairs must be explicitly stated in the review with a sentiment.
       - There must be no duplicates or near-duplicates.
       - All important aspect–opinion pairs mentioned in the review should be included.

    3. If both aspects and opinions are acceptable, then check the **sentiments level**.
       - All proposed sentiment labels must be explicitly supported by the review (no hallucinations).
       - Sentiments must be consistent with the evidence in the text.
       - Avoid duplicates or near-duplicates.
    """
    review: str = dspy.InputField(desc="The original Google review text for a place.")
    proposed_absa_fields: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and sentiment proposed by an agent.")

    decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspects and their corresponding opinion and sentiment are completely correct and needs no edit. Else return 'revise'.")
    rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the aspects and their corresponding opinion and sentiment. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")