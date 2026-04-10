import pandas as pd
import json
import ast
import time

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple

from graph_2_contex_v3 import *
from graph_1_reducers import *

max_debate_loops = 2
print_dial = False

GPT_MODEL = 'openai/gpt-4.1-nano'
# GPT_MODEL = 'openai/gpt-4o-mini'
GPT_API_KEY = ''

GROK_MODEL = "xai/grok-4-1-fast-nonreasoning"
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
    category: Literal['staff_attitude', 'service_process', 'food_drink', 'pricing_value', 'location_access', 'facilities_amenities', 'cleanliness_maintenance', 'atmosphere_vibe', 'natural_environment', 'activities_events', 'safety_security', 'policies_governance', 'inclusivity_community', 'overall_experience', 'other']    
    opinion: str
    sentiment: Literal['positive', 'negative', 'neutral']


class AspectCategory(TypedDict):
    aspects: str
    category: Literal['staff_attitude', 'service_process', 'food_drink', 'pricing_value', 'location_access', 'facilities_amenities', 'cleanliness_maintenance', 'atmosphere_vibe', 'natural_environment', 'activities_events', 'safety_security', 'policies_governance', 'inclusivity_community', 'overall_experience', 'other']    


class ABSAState(TypedDict):
    place_name: str
    review: str
    review_id: int
    
    aspects: List[AspectCategory]
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
    supervision_counter: int





def node_initiation(state: ABSAState) -> ABSAState:
    """ This node sets the counters to 0. """
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
    state["supervision_counter"] = 0
    return state



def node_ate(state: ABSAState) -> ABSAState:
    """ Does the ATE task."""
    # print("************************")
    # time.sleep(0.5)
    state["aspects_counter"] += 1
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY, temperature=0.2)):
        ATE_agent = agents["ate_agent"]
        aspects = [item.get("aspect") for item in state["aspects"]]
        out = ATE_agent(place_name=state["place_name"],
                        review=state["review"],
                        previous_aspects=aspects,
                        revised_aspects_rationale=state["aspects_rationale"])
        aspects = out.get("aspects", [])
        # print(aspects)
        # print("************************")
        state["aspects"] = parse_aspect_list(aspects)
        if print_dial:
            print('ATE Agent:', state["aspects_counter"], ':', state["aspects_decision"], '->', state["aspects"])
            print("************************")
            print(state["aspects_rationale"])
        return state



   

def node_ote(state: ABSAState) -> ABSAState:
    """ Does the OTE task."""
    # print("************************")
    # time.sleep(0.5)
    state["opinions_counter"] += 1
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY, temperature=0.2)):
        OTE_agent = agents["ote_agent"]
        aspects = [item.get("aspect") for item in state["aspects"]]
        out = OTE_agent(review=state["review"],
                        aspects=aspects,
                        previous_opinions=state["opinions"],
                        revised_opinions_rationale=state["opinions_rationale"])
        aspect_opinions = out.get("aspect_opinions", [])
        state["opinions"] = parse_opinions_list(aspect_opinions)

        state["aspects_decision"] = out.get("aspects_decision", [])
        state["aspects_rationale"] = out.get("aspects_rationale", [])
        # state["opinions"] = aspect_opinions
        if print_dial:
            print('OTE Agent:', state["opinions_counter"], ':', state["opinions_decision"], '->', state["opinions"])
            print("************************")
            print(state["opinions_rationale"])
        return state





def node_alsc(state: ABSAState) -> ABSAState:
    """ Does the ALSC task."""
    # print("************************")
    # time.sleep(0.5)
    state["sentiments_counter"] += 1
    with dspy.context(lm=dspy.LM(GPT_MODEL, api_key=GPT_API_KEY, temperature=0.2)):
        ALSC_agent = agents["alsc_agent"]
        out = ALSC_agent(review=state["review"],
                        aspect_category=state["aspects"],
                        aspect_opinions=state["opinions"],
                        previous_sentiments=state["sentiments"],
                        revised_sentiments_rationale=state["sentiments_rationale"])
        sentiments = out.get("sentiments", [])
        state["sentiments"] = parse_sentiments_list(sentiments)

        state["opinions_decision"] = out.get("opinions_decision", [])
        state["opinions_rationale"] = out.get("opinions_rationale", [])
        # state["sentiments"] = sentiments
        if print_dial:
            print('ALSC Agent:', state["sentiments_counter"], ':', state["sentiments_decision"], '->', state["sentiments"])
            print("************************")
            print(state["sentiments_rationale"])
        return state




def node_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ABSA task done by three other agents."""
    # print("************************")
    # time.sleep(0.5)
    state['supervision_counter'] += 1
    state["aspects_counter"] = 1
    state["opinions_counter"] = 1
    state["sentiments_counter"] = 1
    if state['supervision_counter'] < max_debate_loops:
        with dspy.context(lm=dspy.LM(GROK_MODEL, api_key=GROK_API_KEY, api_base=GROK_BASE_URL, temperature=0.2)):
            ABSA_supervisor = agents["supervision_agent"]
            out = ABSA_supervisor(place_name=state["place_name"],
                                review=state["review"],
                                proposed_sentiments=state["sentiments"])

            state["aspects_decision"] = out.get("aspects_decision", "").strip().lower()
            state["opinions_decision"] = out.get("opinions_decision", "").strip().lower()
            state["sentiments_decision"] = out.get("sentiments_decision", "").strip().lower()
            # state["sentiments"] = parse_sentiments_list(out.get("revised_sentiments", []))
            # state["sentiments"] = out.get("revised_sentiments", [])
            state["aspects_rationale"] = out.get("aspects_rationale", "")
            state["opinions_rationale"] = out.get("opinions_rationale", "")
            state["sentiments_rationale"] = out.get("sentiments_rationale", "")

            # if state["aspects_decision"] != "accept" and state["aspects_counter"] < max_debate_loops:
            #     state["aspects_counter"] += 1
            # elif state["opinions_decision"] != "accept" and state["opinions_counter"] < max_debate_loops:
            #     state["opinions_counter"] += 1
            # elif state["sentiments_decision"] != "accept" and state["sentiments_counter"] < max_debate_loops:
            #     state["sentiments_counter"] += 1
            if print_dial:
                print('ABSA Supervisor aspects:', state["aspects_counter"], ':', state["aspects_decision"])
                print("************************")
                print(state["aspects_rationale"])
                print('-----------------')
                print('ABSA Supervisor opinions:', state["opinions_counter"], ':', state["opinions_decision"])
                print("************************")
                print(state["opinions_rationale"])
                print('-----------------')
                print('ABSA Supervisor sentiments:', state["sentiments_counter"], ':', state["sentiments_decision"])
                print("************************")
                print(state["sentiments_rationale"])
            return state
    
    else:
        state["aspects_decision"] = 'accept'
        state["opinions_decision"] = 'accept'
        state["sentiments_decision"] = 'accept'
        return state






def should_do_alsc(state: ABSAState) -> str:
    """ Function to decide where to go forward or backward."""
    if state['aspects_counter'] >= max_debate_loops:
        return "ALSC" 
    elif state["aspects"] == []:
        return "ALSC" 
    elif (state["supervision_counter"] >= 1):
        return "ALSC" 
    elif (state["aspects_decision"] == "accept"):
        return "ALSC"  
    else:
        return "ATE" 



def should_do_supervision(state: ABSAState) -> str:
    """ Function to decide where to go forward or backward."""
    if state['opinions_counter'] >= max_debate_loops:
        return "Supervision" 
    elif state["aspects"] == []:
        return "Supervision" 
    elif (state["supervision_counter"] >= 1):
        return "Supervision" 
    elif (state["aspects_decision"] == "accept"):
        return "Supervision"  
    else:
        return "OTE" 


def should_finish_absa(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["supervision_counter"] >= max_debate_loops):
        return "END" 
    elif (state["aspects_decision"] != "accept"):
        return "ATE"  
    elif (state["opinions_decision"] != "accept"):
        return "OTE" 
    elif (state["sentiments_decision"] != "accept"):
        return "ALSC" 
    else:
        return "END" 

