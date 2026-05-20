import logging
import time
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from services.log_utils.log import log_message


class ContextStateWikidata(TypedDict):
    nlq: str
    retry_count: int
    failed_attempts: List[dict]
    entities: List[str]
    entity_uris: List[dict]
    entity_profile: str
    check_valid: bool
    check_reason: str
    accepted_entity_profile: Optional[str]
    accepted_entity_uris: Optional[List[dict]]
    step_times: dict


def make_context_graph_wikidata(entities_llm, profile_llm, check_llm, log_calls: bool = False) -> StateGraph:
    """
    Builds and compiles the Wikidata context-generation LangGraph sub-graph.

    Flow: extract → el → entity_profile → check → (retry or END)
    No categories step (Wikidata has no dbc: equivalent).
    """
    from services.context_utils.entity_extraction import extract_entities
    from services.context_utils.entity_linking_wikidata import wikidata_el
    from services.context_utils.entity_profile_generation_wikidata import generate_entity_profile_wikidata
    from services.llm_utils import make_context_check_tool
    from prompts.wikidata import context_check_prompt

    _context_check = make_context_check_tool(check_llm, context_prompt=context_check_prompt)

    def extract_node(state: ContextStateWikidata) -> dict:
        _t0 = time.perf_counter()
        try:
            entities = extract_entities(state["nlq"], entities_llm, state["failed_attempts"] or [])
        except Exception as e:
            logging.warning(f"[context_wikidata] extract_entities failed: {e}")
            entities = []
        log_message(step_name="[context_wdt] Extracted entities", color="Cyan", messages=[str(entities)])
        st = state["step_times"]
        st["extraction"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"entities": entities, "step_times": st}

    def el_node(state: ContextStateWikidata) -> dict:
        _t0 = time.perf_counter()
        try:
            entity_uris = wikidata_el(state["nlq"], state["entities"])
        except Exception as e:
            logging.warning(f"[context_wikidata] entity_linking failed: {e}")
            entity_uris = []
        log_message(step_name="[context_wdt] Entity URIs", color="Cyan", messages=[str(entity_uris)])
        st = state["step_times"]
        st["el"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"entity_uris": entity_uris, "step_times": st}

    def entity_profile_node(state: ContextStateWikidata) -> dict:
        _t0 = time.perf_counter()
        try:
            entity_profile = generate_entity_profile_wikidata(
                state["nlq"], state["entity_uris"], profile_llm, log_calls=log_calls
            ) or ""
        except Exception as e:
            logging.warning(f"[context_wikidata] entity_profile_generation failed: {e}")
            entity_profile = ""
        log_message(step_name="[context_wdt] Entity profile generated", color="Cyan", messages=[entity_profile or "(empty)"])
        st = state["step_times"]
        st["entity_profile"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"entity_profile": entity_profile, "step_times": st}

    def check_node(state: ContextStateWikidata) -> dict:
        _t0 = time.perf_counter()
        try:
            result = _context_check.invoke({
                "nlq": state["nlq"],
                "entities": state["entities"],
                "entity_uris": state["entity_uris"],
                "categories": [],
                "entity_profile": state["entity_profile"],
            })
            valid = result.get("valid", False)
            reason = result.get("reason", "")
        except Exception as e:
            logging.warning(f"[context_wikidata] entity_profile_check failed: {e}")
            valid = False
            reason = str(e)

        log_message(
            step_name="[context_wdt] Context check",
            color="Cyan",
            messages=[f"valid={valid}", reason],
        )

        st = state["step_times"]
        st["check"].append(f"{time.perf_counter() - _t0:.2f}s")

        if valid:
            return {
                "check_valid": True,
                "check_reason": reason,
                "accepted_entity_profile": state["entity_profile"],
                "accepted_entity_uris": state["entity_uris"],
                "step_times": st,
            }

        new_retry_count = state["retry_count"] + 1
        new_failed = state["failed_attempts"] + [
            {"entities": state["entities"], "entity_profile": state["entity_profile"], "reason": reason}
        ]
        updates: dict = {
            "check_valid": False,
            "check_reason": reason,
            "failed_attempts": new_failed,
            "retry_count": new_retry_count,
            "step_times": st,
        }
        if new_retry_count >= 3:
            log_message(
                step_name="[context_wdt] Max retries reached — accepting last entity profile",
                color="Yellow",
                messages=[],
            )
            updates["accepted_entity_profile"] = state["entity_profile"]
            updates["accepted_entity_uris"] = state["entity_uris"]
        return updates

    def check_router(state: ContextStateWikidata) -> str:
        if state.get("check_valid") or state.get("retry_count", 0) >= 3:
            return END
        return "extract_node"

    builder = StateGraph(ContextStateWikidata)
    builder.add_node("extract_node", extract_node)
    builder.add_node("el_node", el_node)
    builder.add_node("entity_profile_node", entity_profile_node)
    builder.add_node("check_node", check_node)

    builder.set_entry_point("extract_node")
    builder.add_edge("extract_node", "el_node")
    builder.add_edge("el_node", "entity_profile_node")
    builder.add_edge("entity_profile_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        check_router,
        {"extract_node": "extract_node", END: END},
    )

    return builder.compile()
