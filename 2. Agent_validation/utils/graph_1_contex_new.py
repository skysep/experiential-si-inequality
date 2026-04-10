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


class ATE_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral]. 
    You are a critical supervisor for the Aspect Term Extraction (ATE) task for Aspect-Based Sentiment Analysis (ABSA). Critically evaluate the proposed aspect list.
    Check the following:
    - All of the proposed aspects are explicitly stated in the review with a sentiment.
    - Each aspect is not more than 2 words.
    - There are no duplicates and near-duplicates.
    - All of the aspects that are mentioned in the review are included in the proposed aspects.
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    proposed_aspects: list[str] = dspy.InputField(desc="A list of aspects proposed by the agent.")
    
    decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspects are completely correct and need no edit. Else return 'revise'.")
    # revised_aspects = dspy.OutputField(desc="An array containing the revised aspects. Example Output format: [{'entity': '<entity>'}, ...] without any other text or explanation.")
    rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the aspect list. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")



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


class OTE_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].    
    You are a critical supervisor for the Opinion Term Extraction (OTE) task for Aspect-Based Sentiment Analysis (ABSA). Critically evaluate the proposed aspect-opinion pairs.
    Check the following:
    - All of the proposed aspect-opinion pairs are explicitly stated in the review with a sentiment.
    - Each opinion is not more than 3 words.
    - There are no duplicates and near-duplicates.
    - All of the aspect-opinion pairs that are mentioned in the review are included in the proposed aspect-opinion pairs.
    Keep one best phrase per aspect.
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    proposed_aspect_opinions: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding opinion proposed by another agent.")

    decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed aspect-opinion pairs are completely correct and need no edit. Else return 'revise'.")
    # revised_opinions: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding revised opinion. Output schema: [{'aspect':..., 'opinion':...}, ...]")
    rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the aspect-opinion pairs. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")






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


class ALSC_Supervise(dspy.Signature):
    """
    Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience, etc.) mentioned in a text and determining the sentiment expressed toward each entity. Each entity is associated with a sentiment that can be [positive, negative, or neutral].    
    You are a critical supervisor for the Aspect-level Sentiment Classification (ALSC) task for Aspect-Based Sentiment Analysis (ABSA). Critically evaluate the proposed sentiments for each aspect-opinion pair.
    Check the following:
    - All of the proposed sentiments are explicitly stated in the review with a sentiment.
    - Each sentiment is not more than 3 words.
    - There are no duplicates and near-duplicates.
    - All of the sentiments that are mentioned in the review are included in the proposed sentiments.
    - The sentiments are aligned with the explicit evidence (not assumptions and hallucinations).
    """
    review: str = dspy.InputField(desc="A google review for a place.")
    proposed_sentiments: list[Dict[str, str]] = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by an agent.")
    
    decision: Literal['accept', 'revise'] = dspy.OutputField(desc="Retune 'accept' if the proposed sentiment for all aspect-opinion pairs is completely correct and needs no edit. Else return 'revise'.")
    # revised_sentiments: list[Dict[str, str]] = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion and the revised sentiment of the opinion ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect':..., 'opinion':..., 'sentiment': 'positive' or 'negative' or 'neutral'}, ...]")
    rationale: str = dspy.OutputField(desc="A set of revisions to be applied to the sentiments for each aspect-opinion pair. And reasons, explaining the rationale for the revisions (if the decision is 'revise').")
