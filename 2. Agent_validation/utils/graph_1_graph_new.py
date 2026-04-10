import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple

from graph_1_contex_new import *
from graph_1_reducers import *

max_debate_loops = 2

GPT_MODEL = 'openai/gpt-4o-mini'
GPT_API_KEY = ''

GROK_MODEL = "xai/grok-4-fast-nonreasoning"
GROK_API_KEY = ""
GROK_BASE_URL = "https://api.x.ai/v1"

agents = {
    "ate_agent": dspy.ChainOfThought(ATE_Do),
    "ate_supervision_agent": dspy.ChainOfThought(ATE_Supervise),
    "ote_agent": dspy.ChainOfThought(OTE_Do),
    "ote_supervision_agent": dspy.ChainOfThought(OTE_Supervise),
    "alsc_agent": dspy.ChainOfThought(ALSC_Do),
    "alsc_supervision_agent": dspy.ChainOfThought(ALSC_Supervise),
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
    aspects_counter: int

    opinions: List[Dict[str, str]]
    opinions_decision: str
    opinions_rationale: str
    opinions_counter: int

    sentiments: List[SentimentItem]
    sentiments_decision: str
    sentiments_rationale: str
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

def node_ate_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ATE task done by another agent."""
    # print("************************")
    with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL)):
        ATE_supervisor = agents["ate_supervision_agent"]
        out = ATE_supervisor(review=state["review"],
                            proposed_aspects=state["aspects"])

        state["aspects_decision"] = out.get("decision", "").strip().lower()
        # state["aspects"] = parse_aspect_list(out.get("revised_aspects", []))
        state["aspects_rationale"] = out.get("rationale", "")
        state["aspects_counter"] += 1
        # print('ATE Supervisor:', state["aspects_counter"], ':', state["aspects_decision"], '->', state["aspects"])
        # print("************************")
        # print(state["aspects_rationale"])
        return state
        
    
def should_continue_ate(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["aspects_counter"] < max_debate_loops) and (state["aspects_decision"] != "accept"):
        return "ATE" 
    # elif (state["aspects_counter"] == max_debate_loops) and (state["aspects_decision"] != "accept"):
    #     return "SECOND ATE AGENT"
    else:
        return "OTE" 


   

def node_ote(state: ABSAState) -> ABSAState:
    """ Does the OTE task."""
    # print("************************")
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

def node_ote_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the OTE task done by another agent."""
    # print("************************")
    with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL)):
        OTE_supervisor = agents["ote_supervision_agent"]
        out = OTE_supervisor(review=state["review"],
                            proposed_aspect_opinions=state["opinions"])

        state["opinions_decision"] = out.get("decision", "").strip().lower()
        # state["opinions"] = parse_opinions_list(out.get("revised_opinions", []))
        # state["opinions"] = out.get("revised_opinions", [])
        state["opinions_rationale"] = out.get("rationale", "")
        state["opinions_counter"] += 1
        # print('OTE Supervisor:', state["opinions_counter"], ':', state["opinions_decision"], '->', state["opinions"])
        # print("************************")
        # print(state["opinions_rationale"])
        return state

def should_continue_ote(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["opinions_counter"] < max_debate_loops) and (state["opinions_decision"] != "accept"):
        return "OTE"  
    # elif (state["opinions_counter"] == max_debate_loops) and (state["opinions_decision"] != "accept"):
    #     return "SECOND OTE AGENT"
    else:
        return "ALSC"  




def node_alsc(state: ABSAState) -> ABSAState:
    """ Does the ALSC task."""
    # print("************************")
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

def node_alsc_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ALSC task done by another agent."""
    # print("************************")
    with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL)):
        ALSC_supervisor = agents["alsc_supervision_agent"]
        out = ALSC_supervisor(review=state["review"],
                            proposed_sentiments=state["sentiments"])

        state["sentiments_decision"] = out.get("decision", "").strip().lower()
        # state["sentiments"] = parse_sentiments_list(out.get("revised_sentiments", []))
        # state["sentiments"] = out.get("revised_sentiments", [])
        state["sentiments_rationale"] = out.get("rationale", "")
        state["sentiments_counter"] += 1
        # print('ALSC Supervisor:', state["sentiments_counter"], ':', state["sentiments_decision"], '->', state["sentiments"])
        # print("************************")
        # print(state["sentiments_rationale"])
        return state

def should_continue_alsc(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["sentiments_counter"] < max_debate_loops) and (state["sentiments_decision"] != "accept"):
        return "ALSC"  
    # elif (state["sentiments_counter"] == max_debate_loops) and (state["sentiments_decision"] != "accept"):
    #     return "SECOND ALSC AGENT"
    else:
        return "END" 

