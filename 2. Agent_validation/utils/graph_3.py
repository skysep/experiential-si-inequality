import os
import pandas as pd
import json
import ast

import dspy
from ollama import Client, generate
import requests, json, time
from typing import Any, Dict, List, TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END


max_debate_loops = 3

class ABSA_Supervise(dspy.Signature):
    """
    You are the supervisor for the Aspect-Based Sentiment Analysis (ABSA) task. Critically evaluate the proposed aspects, and their corresponding sentiment.
    Rules:
    - Only consider items which the aspect and sentiment are explicitly stated in the review.
    - Identify distinct aspects mentioned in the review.
    - For each aspect, provide sentiment (positive/negative/neutral).
    - If the review is empty or non-linguistic, return aspects=[].
    - Do NOT include extra fields. Keep opinion phrase short and exactly quoted from the review text.
    - Try to return maximum 2 words as aspect.
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_absa_fields = dspy.InputField(desc="A dictionary of proposed aspects and sentiments.")

    decision = dspy.OutputField(desc="Retune 'accept' if the proposed aspects and sentiments are completely correct and need no edit. Else return 'revise'.")
    revised_absa_fields = dspy.OutputField(desc="A dictionary containing the revised aspects and sentiments. Output schema: [{'aspect': ..., 'sentiment': ...}, ...]")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised aspects and sentiments.")

class ABSA_Do(dspy.Signature):
    """
    You are an expert annotator performing Aspect-Based Sentiment Analysis (ABSA) on google reviews for places. Your task is to extract aspects and their corresponding sentiment.
    """
    
    review = dspy.InputField(desc="A google review for a place.")
    revised_absa_fields = dspy.InputField(desc="A dictionary of proposed aspects and sentiments proposed by another agent.")
    revised_absa_fields_rationale = dspy.InputField(desc="A short reason for the proposed aspects and sentiments by the other agent.")
    
    absa_fields = dspy.OutputField(desc="A dictionary of aspects and sentiments for the place after considering proposed aspects and sentiments and rationale of the other agent. Output schema: [{'aspect': ..., 'sentiment': ...}, ...]")






class Overall_Do(dspy.Signature):
    """
    You are an expert in the sentiment analysis task. Return the overall sentiment for the review. Sentiments: 'positive' or 'negative' or 'neutral'.
    """
    review = dspy.InputField(desc="A google review for a place.")
    revised_overall_sentiment = dspy.InputField(desc="A string expressing the overal sentiment of the review proposed by another agent. Sentiments: 'positive' or 'negative' or 'neutral'.")
    revised_overall_sentiment_rationale = dspy.InputField(desc="A short reason for the proposed overall sentiment by another agent.")
    
    overall_sentiment = dspy.OutputField(desc="A string expressing the overal sentiment of the review. Sentiments: 'positive' or 'negative' or 'neutral'")


class Overall_Supervise(dspy.Signature):
    """
    As the supervisor of a sentiment analysis task, ensure the proposed overall sentiment reflects the full tone (not just the average) of the review.
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_overall_sentiment = dspy.InputField(desc="A string expressing the overall sentiment of the review proposed by an agent.")
    
    decision = dspy.OutputField(desc="Retune 'accept' if the proposed sentiment for the review is completely correct and needs no edit. Else return 'revise'.")
    revised_overall_sentiment = dspy.OutputField(desc="A string expressing the revised overall sentiment of the review. Sentiments: 'positive' or 'negative' or 'neutral'.")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised sentiment for the review. Sentiments: 'positive' or 'negative' or 'neutral'.")










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



class ABSAState(TypedDict, total=False):
    # Inputs / passthrough
    review: Annotated[str, keep_new]
    nlp_absa_fields: Annotated[str, keep_new] 

    absa_agent: Annotated[dspy.ChainOfThought, keep_new]
    absa_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]

    absa_fields: Annotated[List[Dict[str, str]], absa_fields_merge]
    absa_decision: Annotated[str, keep_old_if_accept]
    absa_fields_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    absa_fields_counter: Annotated[int, max_counter]

    sa_agent: Annotated[dspy.ChainOfThought, keep_new]
    sa_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]
    overall_sentiment: Annotated[Literal["positive", "negative", "neutral"], overall_sentiment_merge]
    overall_sentiment_decision: Annotated[str, keep_old_if_accept]
    overall_sentiment_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    overall_sentiment_counter: Annotated[int, max_counter]




def node_initiation(state: ABSAState) -> ABSAState:
    """ This node sets the counters to 0. """
    print("node_initiation")
    # state.setdefault("nlp_absa_fields", "")
    state.setdefault("absa_fields", [])
    state.setdefault("absa_decision", "revise")
    state.setdefault("absa_fields_rationale", "")
    
    
    state.setdefault("overall_sentiment", "neutral")
    state.setdefault("overall_sentiment_decision", "revise")
    state.setdefault("overall_sentiment_rationale", "")

    state["absa_fields_counter"] = 0
    state["overall_sentiment_counter"] = 0
    return state




def node_nlp_extract_pairs(state: ABSAState) -> ABSAState:
    """ Extracts the aspects and sentiments from the NLP-annotated fields."""
    if not state["nlp_absa_fields"]:
        state["absa_fields"] = []
        return state
    else:
        parsed = ast.literal_eval(state["nlp_absa_fields"])
        aspects = parsed.get("aspect", [])
        sentiments = parsed.get("sentiment", [])
        pairs = [
            {"aspect": str(a), "sentiment": str(sentiments[i]).lower().strip()}
            for i, a in enumerate(aspects)
        ]
        state["absa_fields"] = pairs
        return state

def node_absa(state: ABSAState) -> ABSAState:
    """ Does the ABSA task."""
    # print("node_absa")
    ABSA_agent = state["absa_agent"]
    out = ABSA_agent(review=state["review"],
                   revised_absa_fields=state["absa_fields"],
                   revised_absa_fields_rationale=state["absa_fields_rationale"])
    absa_fields = out.get("absa_fields", [])
    state["absa_fields"] = absa_fields
    return state

def node_absa_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ABSA task done by another agent."""
    ABSA_supervisor = state["absa_supervision_agent"]
    out = ABSA_supervisor(review=state["review"],
                        proposed_absa_fields=state["absa_fields"])

    state["absa_decision"] = out.get("decision", "").strip().lower()
    state["absa_fields"] = out.get("revised_absa_fields", [])
    state["absa_fields_rationale"] = out.get("rationale", "")
    state["absa_fields_counter"] += 1
    return state
        
    
def should_continue_absa(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["absa_fields_counter"] < max_debate_loops) and (state["absa_decision"].strip().lower() != "accept"):
        return "ABSA"  # Continue looping
    else:
        return "END"  # Exit the loop




def node_sa(state: ABSAState) -> ABSAState:
    """ Does the sentiment analysis task."""
    SA_agent = state["sa_agent"]
    out = SA_agent(review=state["review"],
                   revised_overall_sentiment=state["overall_sentiment"],
                   revised_overall_sentiment_rationale=state["overall_sentiment_rationale"])
    overall_sentiment = out.get("overall_sentiment", state["overall_sentiment"]).strip().lower()
    state["overall_sentiment"] = overall_sentiment
    return state

def node_sa_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the sentiment analysis task done by another agent."""
    SA_supervisor = state["sa_supervision_agent"]
    out = SA_supervisor(review=state["review"],
                        proposed_overall_sentiment=state["overall_sentiment"])

    state["overall_sentiment_decision"] = out.get("decision", "").strip().lower()
    state["overall_sentiment"] = out.get("revised_overall_sentiment", state["overall_sentiment"]).strip().lower()
    state["overall_sentiment_rationale"] = out.get("rationale", "")
    state["overall_sentiment_counter"] += 1
    return state

def should_continue_sa(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["overall_sentiment_counter"] < max_debate_loops) and (state["overall_sentiment_decision"].strip().lower() != "accept"):
        return "SA"  # Continue looping
    else:
        return "END"  # Exit the loop
