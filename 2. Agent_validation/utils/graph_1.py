import pandas as pd
import json
import ast

import dspy
from typing import Any, Dict, List, TypedDict, Literal, Annotated, Optional, Tuple


max_debate_loops = 2


class ATE_Do(dspy.Signature):
    """
    You are an expert annotator for the Aspect Term Extraction (ATE) task for Aspect-Based Sentiment Analysis (ABSA), and your task is to extract explicit aspect terms mentioned in the review (≤2 words each).
    """
    
    review = dspy.InputField(desc="A google review for a place.")
    revised_aspects = dspy.InputField(desc="A list of aspects proposed by another agent.")
    revised_aspects_rationale = dspy.InputField(desc="A short reason for the proposed aspects by the other agent.")
    
    aspects = dspy.OutputField(desc="An array of aspects for the place after considering proposed aspects and rationale of the other agent.")


class ATE_Supervise(dspy.Signature):
    """
    You are the supervisor for the Aspect Term Extraction (ATE) task for Aspect-Based Sentiment Analysis (ABSA). Critically evaluate the proposed aspect list.
    Rules:
    - Only aspects explicitly mentioned in the text.
    - Keep each aspect to ≤2 words.
    - Remove duplicates and near-duplicates.
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_aspects = dspy.InputField(desc="A list of aspects proposed by an agent.")
    
    decision = dspy.OutputField(desc="Retune 'accept' if the proposed aspects are completely correct and need no edit. Else return 'revise'.")
    revised_aspects = dspy.OutputField(desc="An array containing the revised aspects.")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised aspects.")



class OTE_Do(dspy.Signature):
    """
    You are an expert annotator for Opinion Term Extraction (OTE) for Aspect-Based Sentiment Analysis (ABSA). For each aspect, extract one short opinion phrase (≤3 words, verbatim).
    """
    review = dspy.InputField(desc="A google review for a place.")
    aspects = dspy.InputField(desc="A list of aspects.")
    revised_opinions = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion in the review proposed by another agent.")
    revised_opinions_rationale = dspy.InputField(desc="A short reason for the proposed opinions by the other agent.")
    
    aspect_opinions = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion in the review. output schema: [{'aspect':..., 'opinion':...}, ...]")


class OTE_Supervise(dspy.Signature):
    """
    You are the supervisor for the Opinion Term Extraction (OTE) task for Aspect-Based Sentiment Analysis (ABSA). Check that the proposed aspect-opinion pairs are valid.
    Keep one best phrase per aspect.
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_aspect_opinions = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding opinion proposed by another agent.")

    decision = dspy.OutputField(desc="Retune 'accept' if the proposed aspect-opinion pairs are completely correct and need no edit. Else return 'revise'.")
    revised_opinions = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding revised opinion. Output schema: [{'aspect':..., 'opinion':...}, ...]")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised aspect-opinion pairs.")




class ALSC_Do(dspy.Signature):
    """
    You are an expert annotator for the Aspect-level Sentiment Classification (ALSC) task for Aspect-Based Sentiment Analysis (ABSA). For each aspect, classify the sentiment of the opinion as 'positive', 'negative', or 'neutral'.
    Only use explicit evidence. Output JSON array:
        [{"aspect":..., "opinion":..., "sentiment": 'positive' or 'negative' or 'neutral'}, ...]
    """
    review = dspy.InputField(desc="A google review for a place.")
    aspect_opinions = dspy.InputField(desc="An array of JSONs containing aspects and their corrosponding opinion.")
    revised_sentiments = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by another agent.")
    revised_sentiments_rationale = dspy.InputField(desc="A short reason for the proposed sentiment for each aspect-opinions pair by the other agent.")
    
    sentiments = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect':..., 'opinion':..., 'sentiment': 'positive' or 'negative' or 'neutral'}, ...]")

class ALSC_Supervise(dspy.Signature):
    """
    You are the supervisor for the ALSC task for Aspect-Based Sentiment Analysis (ABSA). Verify labels align with explicit evidence (not assumptions and hallucinations).
    Respond with JSON:
      {"decision": "accept" or "revise",
       "sentiments": ["aspect":..., "opinion":..., "sentiment": 'positive' or 'negative' or 'neutral'}, ...],
       "rationale": ...}
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_sentiments = dspy.InputField(desc="An array of JSONs containing aspects and their corresponding opinion and the sentiment of the opinion proposed by an agent.")
    
    decision = dspy.OutputField(desc="Retune 'accept' if the proposed sentiment for all aspect-opinion pairs is completely correct and needs no edit. Else return 'revise'.")
    revised_sentiments = dspy.OutputField(desc="An array of JSONs containing aspects and their corresponding opinion and the revised sentiment of the opinion ('positive' or 'negative' or 'neutral'). Output schema: [{'aspect':..., 'opinion':..., 'sentiment': 'positive' or 'negative' or 'neutral'}, ...]")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised sentiment for each aspect-opinion pair.")



class Overall_Do(dspy.Signature):
    """
    You are an expert in the sentiment analysis task. Return the overall sentiment for the review: 'positive' or 'negative' or 'neutral'.
    """
    review = dspy.InputField(desc="A google review for a place.")
    revised_overall_sentiment = dspy.InputField(desc="A string expressing the overal sentiment of the review proposed by another agent.")
    revised_overall_sentiment_rationale = dspy.InputField(desc="A short reason for the proposed overall sentiment by another agent.")
    
    overall_sentiment = dspy.OutputField(desc="A string expressing the overal sentiment of the review.")

class Overall_Supervise(dspy.Signature):
    """
    As the supervisor of a sentiment analysis task, ensure the proposed overall sentiment reflects the full tone (not just the average) of the review. Sentiments: 'positive' or 'negative' or 'neutral'
    """
    review = dspy.InputField(desc="A google review for a place.")
    proposed_overall_sentiment = dspy.InputField(desc="A string expressing the overall sentiment of the review proposed by an agent.")
    
    decision = dspy.OutputField(desc="Retune 'accept' if the proposed sentiment for the review is completely correct and needs no edit. Else return 'revise'.")
    revised_overall_sentiment = dspy.OutputField(desc="A string expressing the revised overall sentiment of the review. Sentiments: 'positive' or 'negative' or 'neutral'")
    rationale = dspy.OutputField(desc="A short reason, explaining the rationale for the revised sentiment for the review.")









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

AllowedSent = Literal["positive", "negative", "neutral"]

def normalize_sent(s: Optional[str]) -> Optional[AllowedSent]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    return s if s in {"positive", "negative", "neutral"} else None

def decisive_first(old: Optional[AllowedSent], new: Optional[AllowedSent]) -> Optional[AllowedSent]:
    """
    Keep old if it's decisive (positive/negative). Otherwise take new if valid.
    """
    old_n = normalize_sent(old)
    new_n = normalize_sent(new)
    if old_n in {"positive", "negative"}:
        return old_n
    return new_n if new_n is not None else old_n

def merge_aspects(old: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    """
    Union + stable order for aspect strings. If old has items, keep them and append any new unseen.
    """
    old_list = list(old or [])
    new_list = list(new or [])
    return old_list if len(old_list) > 0 else new_list

def merge_opinions(old: Optional[List[Dict[str, str]]], new: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Union + stable order for opinions. Uses ('aspect','opinion') as identity.
    """
    old_list = list(old or [])
    new_list = list(new or [])

    return old_list if len(old_list) > 0 else new_list

def merge_sentiment_items(
    old: Optional[List["SentimentItem"]],
    new: Optional[List["SentimentItem"]],
) -> List["SentimentItem"]:
    """
    Union + stable order for SentimentItem.
    Identity: (aspect, opinion).
    If duplicate pair exists, prefer the item with decisive sentiment (pos/neg) over neutral.
    """
    old_list = list(old or [])
    new_list = list(new or [])
    return old_list if len(old_list) > 0 else new_list

def parse_aspect_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []

def parse_opinions_list(v: Any) -> List[Dict[str, str]]:
    """
    Parse/normalize LLM outputs into a list of {"aspect":..., "opinion":...}.
    Accepts:
      - list of dicts
      - dict
      - stringified list
      - JSON or Python literal
      - code fences
      - junk (ignored)
    Always returns a clean list[dict].
    """

    # Handle None
    if v is None:
        return []

    # If it's already a list
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                aspect = str(item.get("aspect", "")).strip()
                opinion = str(item.get("opinion", "")).strip()
                out.append({"aspect": aspect, "opinion": opinion})
            elif isinstance(item, str):
                # maybe a dict encoded as string
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        aspect = str(parsed.get("aspect", "")).strip()
                        opinion = str(parsed.get("opinion", "")).strip()
                        out.append({"aspect": aspect, "opinion": opinion})
                except Exception:
                    pass
        return out

    # If it's a dict
    if isinstance(v, dict):
        return [{
            "aspect": str(v.get("aspect", "")).strip(),
            "opinion": str(v.get("opinion", "")).strip()
        }]

    # If it is a string
    if isinstance(v, str):
        s = v.strip()

        # remove code fences
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]

        # Try JSON
        try:
            parsed = json.loads(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            return parse_opinions_list(parsed)
        except Exception:
            pass

        # fallback: treat as one opinion with missing aspect
        return [{"aspect": "", "opinion": s}]

    # Unknown type
    return []


def parse_sentiments_list(v: Any) -> List[Dict[str, str]]:
    """
    Parse anything into a list of SentimentItem dicts:
        {"aspect":..., "opinion":..., "sentiment": ...}
    sentiment must be one of {'positive','negative','neutral'}.
    """

    allowed = {"positive", "negative", "neutral"}

    # Handle None
    if v is None:
        return []

    def clean_item(item: Dict[str, Any]) -> Dict[str, str]:
        aspect = str(item.get("aspect", "")).strip()
        opinion = str(item.get("opinion", "")).strip()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if sentiment not in allowed:
            sentiment = "neutral"
        return {
            "aspect": aspect,
            "opinion": opinion,
            "sentiment": sentiment
        }

    # If it's already a list
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                out.append(clean_item(item))
            elif isinstance(item, str):
                # maybe a dict as string
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        out.append(clean_item(parsed))
                except Exception:
                    pass
        return out

    # If it's a dict
    if isinstance(v, dict):
        return [clean_item(v)]

    # If string
    if isinstance(v, str):
        s = v.strip()

        # strip code fences
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").split("\n", 1)[-1]

        # Try JSON
        try:
            parsed = json.loads(s)
            return parse_sentiments_list(parsed)
        except Exception:
            pass

        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            return parse_sentiments_list(parsed)
        except Exception:
            pass

        # Unknown string → treat as "neutral" with no aspect/opinion
        return [{"aspect": "", "opinion": s, "sentiment": "neutral"}]

    # Unknown type
    return []



class SentimentItem(TypedDict, total=False):
    aspect: Annotated[str, keep_new]
    opinion: Annotated[str, keep_new]
    sentiment: Annotated[Literal['positive', 'negative', 'neutral'], decisive_first]

class ABSAState(TypedDict, total=False):
    review: Annotated[str, keep_new]

    ate_agent: Annotated[dspy.ChainOfThought, keep_new]
    ate_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]
    aspects: Annotated[List[str], merge_aspects]
    aspects_decision: Annotated[str, keep_old_if_accept]
    aspects_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    aspects_counter: Annotated[int, max_counter]

    ote_agent: Annotated[dspy.ChainOfThought, keep_new]
    ote_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]
    opinions: Annotated[List[Dict[str, str]], merge_opinions]   # keys: 'aspect','opinion'
    opinions_decision: Annotated[str, keep_old_if_accept]
    opinions_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    opinions_counter: Annotated[int, max_counter]

    alsc_agent: Annotated[dspy.ChainOfThought, keep_new]
    alsc_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]
    sentiments: Annotated[List[SentimentItem], merge_sentiment_items]
    sentiments_decision: Annotated[str, keep_old_if_accept]
    sentiments_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    sentiments_counter: Annotated[int, max_counter]

    sa_agent: Annotated[dspy.ChainOfThought, keep_new]
    sa_supervision_agent: Annotated[dspy.ChainOfThought, keep_new]
    overall_sentiment: Annotated[AllowedSent, decisive_first]
    overall_sentiment_decision: Annotated[str, keep_old_if_accept]
    overall_sentiment_rationale: Annotated[str, keep_old_if_nonempty_else_new]
    overall_sentiment_counter: Annotated[int, max_counter]


# class SentimentItem(TypedDict):
#     aspect: str
#     opinion: str
#     sentiment: Literal['positive', 'negative', 'neutral']
    
# class ABSAState(TypedDict):
#     review: str
    
#     ate_agent: dspy.ChainOfThought
#     ate_supervision_agent: dspy.ChainOfThought
#     aspects: List[str]
#     aspects_decision: str
#     aspects_rationale: str
#     aspects_counter: int

#     ote_agent: dspy.ChainOfThought
#     ote_supervision_agent: dspy.ChainOfThought
#     opinions: List[Dict[str, str]]
#     opinions_decision: str
#     opinions_rationale: str
#     opinions_counter: int

#     alsc_agent: dspy.ChainOfThought
#     alsc_supervision_agent: dspy.ChainOfThought
#     sentiments: List[SentimentItem]
#     sentiments_decision: str
#     sentiments_rationale: str
#     sentiments_counter: int

#     sa_agent: dspy.ChainOfThought
#     sa_supervision_agent: dspy.ChainOfThought
#     overall_sentiment: Literal['positive', 'negative', 'neutral']
#     overall_sentiment_decision: str
#     overall_sentiment_rationale: str
#     overall_sentiment_counter: int



def node_initiation(state: ABSAState) -> ABSAState:
    """ This node sets the counters to 0. """
    print("node_initiation")
    state.setdefault("aspects", [])
    state.setdefault("aspects_decision", "")
    state.setdefault("aspects_rationale", "")
    
    state.setdefault("opinions", [])
    state.setdefault("opinions_decision", "")
    state.setdefault("opinions_rationale", "")
    
    state.setdefault("sentiments", [])
    state.setdefault("sentiments_decision", "")
    state.setdefault("sentiments_rationale", "")
    
    state.setdefault("overall_sentiment", "neutral")
    state.setdefault("overall_sentiment_decision", "")
    state.setdefault("overall_sentiment_rationale", "")

    state["aspects_counter"] = 0
    state["opinions_counter"] = 0
    state["sentiments_counter"] = 0
    state["overall_sentiment_counter"] = 0
    return state



def node_ate(state: ABSAState) -> ABSAState:
    """ Does the ATE task."""
    # print(state["aspects"])
    # print("************************")
    ATE_agent = state["ate_agent"]
    out = ATE_agent(review=state["review"],
                   revised_aspects=state["aspects"],
                   revised_aspects_rationale=state["aspects_rationale"])
    aspects = out.get("aspects", [])
    # print(aspects)
    # print("************************")
    state["aspects"] = parse_aspect_list(aspects)
    return state

def node_ate_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ATE task done by another agent."""
    # print(state["aspects"])
    # print("************************")
    ATE_supervisor = state["ate_supervision_agent"]
    out = ATE_supervisor(review=state["review"],
                        proposed_aspects=state["aspects"])

    state["aspects_decision"] = out.get("decision", "").strip().lower()
    state["aspects"] = parse_aspect_list(out.get("revised_aspects", []))
    state["aspects_rationale"] = out.get("rationale", "")
    state["aspects_counter"] += 1
    return state
        
    
def should_continue_ate(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["aspects_counter"] < max_debate_loops) and (state["aspects_decision"] != "accept"):
        return "ATE"  
    else:
        return "OTE" 


   

def node_ote(state: ABSAState) -> ABSAState:
    """ Does the OTE task."""
    # print("node_ote")
    OTE_agent = state["ote_agent"]
    out = OTE_agent(review=state["review"],
                   aspects=state["aspects"],
                   revised_opinions=state["opinions"],
                   revised_opinions_rationale=state["opinions_rationale"])
    aspect_opinions = out.get("aspect_opinions", [])
    state["opinions"] = parse_opinions_list(aspect_opinions)
    # state["opinions"] = aspect_opinions
    return state

def node_ote_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the OTE task done by another agent."""
    # print("node_ote_supervision")
    OTE_supervisor = state["ote_supervision_agent"]
    out = OTE_supervisor(review=state["review"],
                        proposed_aspect_opinions=state["opinions"])

    state["opinions_decision"] = out.get("decision", "").strip().lower()
    state["opinions"] = parse_opinions_list(out.get("revised_opinions", []))
    # state["opinions"] = out.get("revised_opinions", [])
    state["opinions_rationale"] = out.get("rationale", "")
    state["opinions_counter"] += 1
    return state

def should_continue_ote(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["opinions_counter"] < max_debate_loops) and (state["opinions_decision"] != "accept"):
        return "OTE"  
    else:
        return "ALSC"  




def node_alsc(state: ABSAState) -> ABSAState:
    """ Does the ALSC task."""
    # print("node_alsc")
    ALSC_agent = state["alsc_agent"]
    out = ALSC_agent(review=state["review"],
                   aspect_opinions=state["opinions"],
                   revised_sentiments=state["sentiments"],
                   revised_sentiments_rationale=state["sentiments_rationale"])
    sentiments = out.get("sentiments", [])
    state["sentiments"] = parse_sentiments_list(sentiments)
    # state["sentiments"] = sentiments
    return state

def node_alsc_supervision(state: ABSAState) -> ABSAState:
    """ Supervise the results of the ALSC task done by another agent."""
    # print("node_alsc_supervision")
    ALSC_supervisor = state["alsc_supervision_agent"]
    out = ALSC_supervisor(review=state["review"],
                        proposed_sentiments=state["sentiments"])

    state["sentiments_decision"] = out.get("decision", "").strip().lower()
    state["sentiments"] = parse_sentiments_list(out.get("revised_sentiments", []))
    # state["sentiments"] = out.get("revised_sentiments", [])
    state["sentiments_rationale"] = out.get("rationale", "")
    state["sentiments_counter"] += 1
    return state

def should_continue_alsc(state: ABSAState) -> str:
    """ Function to decide what to do next."""
    if (state["sentiments_counter"] < max_debate_loops) and (state["sentiments_decision"] != "accept"):
        return "ALSC"  
    else:
        return "END" 



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
        return "SA" 
    else:
        return "END"  
