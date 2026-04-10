import pandas as pd
import json
import ast
import time

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple

from graph_2_contex_new import *
from graph_1_reducers import *

max_debate_loops = 2

GPT_MODEL = 'openai/gpt-4o-mini'
GPT_API_KEY = ''

GROK_MODEL = "xai/grok-4-fast-nonreasoning"
GROK_API_KEY = ""
GROK_BASE_URL = "https://api.x.ai/v1"

agents = {
    "ate_agent": dspy.ChainOfThought(ATE_Do),
    "ote_agent": dspy.ChainOfThought(OTE_Do),
    "alsc_agent": dspy.ChainOfThought(ALSC_Do),
    "supervision_agent": dspy.ChainOfThought(ABSA_Supervise),
}



class SentimentItem(TypedDict):
    aspect: str
    opinion: str
    sentiment: Literal['positive', 'negative', 'neutral']
    
class ABSAState(TypedDict):
    review: str
    review_id: int
    
    aspects: List[str]
    aspects_decision: str
    aspects_rationale: str

    opinions: List[Dict[str, str]]
    opinions_decision: str
    opinions_rationale: str

    sentiments: List[SentimentItem]
    sentiments_decision: str
    sentiments_rationale: str

    aspects_counter: int
    opinions_counter: int
    sentiments_counter: int




def node_initiation(state: ABSAState) -> ABSAState:
    """ This node sets the counters to 0. """
    # print("node_initiation")
    state.setdefault("aspects", [])
    state.setdefault("aspects_decision", "revise")
    state.setdefault("aspects_rationale", "")
    
    state.setdefault("opinions", [])
    state.setdefault("opinions_decision", "revise")
    state.setdefault("opinions_rationale", "")
    
    state.setdefault("sentiments", [])
    state.setdefault("sentiments_decision", "revise")
    state.setdefault("sentiments_rationale", "")
    
    state["aspects_counter"] = 0
    state["opinions_counter"] = 0
    state["sentiments_counter"] = 0
    return state



def node_ate(state: ABSAState) -> ABSAState:
    """ Does the ATE task."""
    # print("************************")
    time.sleep(0.5)
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY)):
        ATE_agent = agents["ate_agent"]
        out = ATE_agent(review=state["review"],
                        previous_aspects=state["aspects"],
                        revised_aspects_rationale=state["aspects_rationale"])
        aspects = out.get("aspects", [])
        # print(aspects)
        # print("************************")
        state["aspects"] = parse_aspect_list(aspects)
        # print('ATE Agent:', state["aspects_counter"], ':', state["aspects_decision"], '->', state["aspects"])
        # print("************************")
        # print(state["aspects_rationale"])
        return state



   

def node_ote(state: ABSAState) -> ABSAState:
    """ Does the OTE task."""
    # print("************************")
    time.sleep(0.5)
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY)):
        OTE_agent = agents["ote_agent"]
        out = OTE_agent(review=state["review"],
                        aspects=state["aspects"],
                        previous_opinions=state["opinions"],
                        revised_opinions_rationale=state["opinions_rationale"])
        aspect_opinions = out.get("aspect_opinions", [])
        state["opinions"] = parse_opinions_list(aspect_opinions)
        # state["opinions"] = aspect_opinions
        # print('OTE Agent:', state["opinions_counter"], ':', state["opinions_decision"], '->', state["opinions"])
        # print("************************")
        # print(state["opinions_rationale"])
        return state





def node_alsc(state: ABSAState) -> ABSAState:
    """ Does the ALSC task."""
    # print("************************")
    time.sleep(0.5)
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY)):
        ALSC_agent = agents["alsc_agent"]
        out = ALSC_agent(review=state["review"],
                        aspect_opinions=state["opinions"],
                        previous_sentiments=state["sentiments"],
                        revised_sentiments_rationale=state["sentiments_rationale"])
        sentiments = out.get("sentiments", [])
        state["sentiments"] = parse_sentiments_list(sentiments)
        # state["sentiments"] = sentiments
        # print('ALSC Agent:', state["sentiments_counter"], ':', state["sentiments_decision"], '->', state["sentiments"])
        # print("************************")
        # print(state["sentiments_rationale"])
        return state




def node_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ABSA task done by three other agents."""
    # print("************************")
    time.sleep(0.5)
    with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL)):
        ALSC_supervisor = agents["supervision_agent"]
        out = ALSC_supervisor(review=state["review"],
                            proposed_sentiments=state["sentiments"])

        state["aspects_decision"] = out.get("aspects_decision", "").strip().lower()
        state["opinions_decision"] = out.get("opinions_decision", "").strip().lower()
        state["sentiments_decision"] = out.get("sentiments_decision", "").strip().lower()
        # state["sentiments"] = parse_sentiments_list(out.get("revised_sentiments", []))
        # state["sentiments"] = out.get("revised_sentiments", [])
        state["aspects_rationale"] = out.get("aspects_rationale", "")
        state["opinions_rationale"] = out.get("opinions_rationale", "")
        state["sentiments_rationale"] = out.get("sentiments_rationale", "")

        if state["aspects_decision"] != "accept" and state["aspects_counter"] < max_debate_loops:
            state["aspects_counter"] += 1
        elif state["opinions_decision"] != "accept" and state["opinions_counter"] < max_debate_loops:
            state["opinions_counter"] += 1
        elif state["sentiments_decision"] != "accept" and state["sentiments_counter"] < max_debate_loops:
            state["sentiments_counter"] += 1

        # print('ALSC Supervisor:', state["sentiments_counter"], ':', state["sentiments_decision"], '->', state["sentiments"])
        # print("************************")
        # print(state["sentiments_rationale"])
        return state



def should_continue_absa(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["aspects_counter"] < max_debate_loops) and (state["aspects_decision"] != "accept"):
        return "ATE"  
    elif (state["opinions_counter"] < max_debate_loops) and (state["opinions_decision"] != "accept"):
        return "OTE" 
    elif (state["sentiments_counter"] < max_debate_loops) and (state["sentiments_decision"] != "accept"):
        return "ALSC" 
    else:
        return "END" 

