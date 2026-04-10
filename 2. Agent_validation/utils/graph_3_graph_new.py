import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple

from graph_3_contex_new import *
from graph_1_reducers import *

max_debate_loops = 2

GPT_MODEL = 'openai/gpt-4o-mini'
GPT_API_KEY = ''

GROK_MODEL = "xai/grok-4-fast-nonreasoning"
GROK_API_KEY = ""
GROK_BASE_URL = "https://api.x.ai/v1"


agents = {
    "absa_agent": dspy.ChainOfThought(ABSA_Do),
    "supervision_agent": dspy.ChainOfThought(ABSA_Supervise),
}



class SentimentItem(TypedDict):
    aspect: str
    opinion: str
    sentiment: Literal['positive', 'negative', 'neutral']
    
class ABSAState(TypedDict):
    review: str
    review_id: int
    
    absa_fields: List[Dict[str, str]]
    absa_decision: str
    absa_rationale: str

    absa_counter: int




def node_initiation(state: ABSAState) -> ABSAState:
    """ This node sets the counters to 0. """
    # print("node_initiation")
    state.setdefault("absa_fields", [])
    state.setdefault("absa_decision", "revise")
    state.setdefault("absa_rationale", "")
    
    state["absa_counter"] = 0
    return state



def node_absa(state: ABSAState) -> ABSAState:
    """ Does the ABSA task."""
    # print("************************")
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY)):
        ABSA_agent = agents["absa_agent"]
        out = ABSA_agent(review=state["review"],
                        previous_absa_fields=state["absa_fields"],
                        revised_absa_rationale=state["absa_rationale"])
        absa_fields = out.get("absa_fields", [])
        # print(absa_fields)
        # print("************************")
        state["absa_fields"] = absa_fields
        # print('ABSA Agent:', state["absa_counter"], ':', state["absa_decision"], '->', state["absa_fields"])
        # print("************************")
        # print(state["absa_rationale"])
        return state




def node_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ABSA task done by another agents."""
    # print("************************")
    with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL)):
        ABSA_supervisor = agents["supervision_agent"]
        out = ABSA_supervisor(review=state["review"],
                              proposed_absa_fields=state["absa_fields"])

        state["absa_decision"] = out.get("absa_decision", "").strip().lower()
        state["absa_rationale"] = out.get("absa_rationale", "")
        state["absa_counter"] += 1

        # print('ABSA Supervisor:', state["absa_counter"], ':', state["absa_decision"], '->', state["absa_fields"])
        # print("************************")
        # print(state["absa_rationale"])
        return state



def should_continue_absa(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["absa_counter"] < max_debate_loops) and (state["absa_decision"] != "accept"):
        return "ABSA"  
    else:
        return "END" 
