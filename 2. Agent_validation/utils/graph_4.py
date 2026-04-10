import os
import pandas as pd
import json
import ast

import dspy
from ollama import Client, generate
import requests, json, time
from typing import Any, Dict, List, TypedDict, Literal, Optional, Annotated
from langgraph.graph import StateGraph, START, END


max_debate_loops = 3


class ABSA_Do(dspy.Signature):
    """
    You are an expert annotator performing Aspect-Based Sentiment Analysis (ABSA) on google reviews for places. Your task is to extract aspects and their corresponding opinions and sentiments.
    """
    
    review = dspy.InputField(desc="A google review for a place.")
    revised_absa_fields = dspy.InputField(desc="A dictionary of aspects, opinions, and sentiments proposed by another agent.")
    revised_absa_fields_rationale = dspy.InputField(desc="A short reason for the proposed aspects, opinions, and sentiments by the other agent.")
    
    absa_fields = dspy.OutputField(desc="A dictionary of aspects, opinions, and sentiments for the place after considering proposed aspects, opinions, and sentiments and rationale of the other agent. Output schema: [{'aspect': ..., 'opinion': ..., 'sentiment': ...}, ...]")


class ABSA_Supervise(dspy.Signature):
    """
    You are the supervisor for the Aspect-Based Sentiment Analysis (ABSA) task. Critically evaluate the proposed aspects, opinions, and sentiments.
    Rules:
    - Only consider items which the aspect and sentiment are explicitly stated in the review.
    - Identify distinct aspects mentioned in the review.
    - For each aspect, provide: opinion phrase and sentiment (positive/negative/neutral).
    - If the review is empty or non-linguistic, return aspects=[].
    - Do NOT include extra fields. Keep opinion phrase short and exactly quoted from the review text.
    - Try to return maximum 2 words as aspect.
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_absa_fields = dspy.InputField(desc="A dictionary of aspects, opinions, and sentiments proposed by an agent.")

    decision = dspy.OutputField(desc="Retune 'accept' if the proposed aspects, opinions, and sentiments are completely correct and need no edit. Else return 'revise'.")
    revised_absa_fields = dspy.OutputField(desc="A dictionary containing the revised aspects, opinions, and sentiments. Output schema: [{'aspect': ..., 'opinion': ..., 'sentiment': ...}, ...]")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised aspects, opinions, and sentiments.")




class Overall_Do(dspy.Signature):
    """
    You are an expert in the sentiment analysis task. Return the overall sentiment for the review: 'positive' or 'negative' or 'neutral'.
    """
    review = dspy.InputField(desc="A google review for a place.")
    revised_overall_sentiment = dspy.InputField(desc="A string expressing the overal sentiment of the review proposed by another agent.")
    revised_overall_sentiment_rationale = dspy.InputField(desc="A short reason for the proposed overall sentiment by another agent.")
    
    overall_sentiment = dspy.OutputField(desc="A string expressing the overal sentiment of the review. Sentiments: 'positive' or 'negative' or 'neutral'")


class Overall_Supervise(dspy.Signature):
    """
    As the supervisor of a sentiment analysis task, ensure the proposed overall sentiment reflects the full tone (not just the average) of the review. Sentiments: 'positive' or 'negative' or 'neutral'
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_overall_sentiment = dspy.InputField(desc="A string expressing the overall sentiment of the review proposed by an agent. Sentiments: 'positive' or 'negative' or 'neutral'")
    
    decision = dspy.OutputField(desc="Retune 'accept' if the proposed sentiment for the review is completely correct and needs no edit. Else return 'revise'.")
    revised_overall_sentiment = dspy.OutputField(desc="A string expressing the revised overall sentiment of the review. Sentiments: 'positive' or 'negative' or 'neutral'")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised sentiment for the review. Sentiments: 'positive' or 'negative' or 'neutral'")











def keep_new(old, new):
    """Always prefer the new value when present."""
    return new if new is not None else old

def keep_old_if_accept(old: Optional[str], new: Optional[str]):
    """For decision keys: keep old if it's 'accept'; else prefer new."""
    if isinstance(old, str) and old.strip().lower() == "accept":
        return old
    return new if new is not None else old

def keep_old_if_nonempty_else_new(old: Optional[str], new: Optional[str]):
    """Keep old if non-empty after strip; else prefer new."""
    if isinstance(old, str) and old.strip() != "":
        return old
    return new if new is not None else old

def max_counter(old: Optional[int], new: Optional[int]):
    """Merge counters by max (None-safe)."""
    if old is None:
        return new
    if new is None:
        return old
    return max(old, new)

def normalize_sentiment(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    return s if s in {"positive", "negative", "neutral"} else None

def overall_sentiment_merge(
    old: Optional[Literal["positive", "negative", "neutral"]],
    new: Optional[Literal["positive", "negative", "neutral"]],
) -> Optional[Literal["positive", "negative", "neutral"]]:
    """
    Keep decisive old (positive/negative); otherwise take new if valid.
    """
    old_n = normalize_sentiment(old)
    new_n = normalize_sentiment(new)
    if old_n in {"positive", "negative"}:
        return old_n  # keep decisive old
    return new_n if new_n is not None else old_n

def sentiment_items_merge(
    old: Optional[List["SentimentItem"]],
    new: Optional[List["SentimentItem"]],
) -> List["SentimentItem"]:
    """
    Keep old if it already has items; otherwise take new.
    Ensures we always return a list.
    """
    old_list = old or []
    new_list = new or []
    return old_list if len(old_list) > 0 else new_list



class SentimentItem(TypedDict, total=False):
    aspect: Annotated[str, keep_new]
    opinion: Annotated[str, keep_new]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], overall_sentiment_merge]


class ABSAState(TypedDict, total=False):
    review: Annotated[str, keep_new]

    absa_agent: Annotated[dspy.ChainOfThought, keep_new]
    absa_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]

    absa_fields: Annotated[List[SentimentItem], sentiment_items_merge]
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
    state.setdefault("absa_fields", [])
    state.setdefault("absa_decision", "revise")
    state.setdefault("absa_fields_rationale", "")
    
    
    state.setdefault("overall_sentiment", "neutral")
    state.setdefault("overall_sentiment_decision", "revise")
    state.setdefault("overall_sentiment_rationale", "")

    state["absa_fields_counter"] = 0
    state["overall_sentiment_counter"] = 0
    return state



def node_absa(state: ABSAState) -> ABSAState:
    """ Does the ABSA task."""
    # print("node_ate")
    ABSA_agent = state["absa_agent"]
    out = ABSA_agent(review=state["review"],
                   revised_absa_fields=state["absa_fields"],
                   revised_absa_fields_rationale=state["absa_fields_rationale"])
    absa_fields = out.get("absa_fields", [])
    state["absa_fields"] = absa_fields
    return state

def node_absa_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ABSA task done by another agent."""
    # print("node_absa_supervision")
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
    if (state["absa_fields_counter"] < max_debate_loops) and (state["absa_decision"] != "accept"):
        return "ABSA"  # Continue looping
    else:
        return "END"  # Exit the loop




def node_sa(state: ABSAState) -> ABSAState:
    """ Does the sentiment analysis task."""
    # print("node_sa")
    SA_agent = state["sa_agent"]
    out = SA_agent(review=state["review"],
                   revised_overall_sentiment=state["overall_sentiment"],
                   revised_overall_sentiment_rationale=state["overall_sentiment_rationale"])
    overall_sentiment = out.get("overall_sentiment", state["overall_sentiment"]).strip().lower()
    state["overall_sentiment"] = overall_sentiment
    return state

def node_sa_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the sentiment analysis task done by another agent."""
    # print("node_sa_supervision")
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
    if (state["overall_sentiment_counter"] < max_debate_loops) and (state["overall_sentiment_decision"] != "accept"):
        return "SA"  # Continue looping
    else:
        return "END"  # Exit the loop
