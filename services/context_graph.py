import json
import logging
import time
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from services.log_utils.log import log_message


class ContextState(TypedDict):
    nlq: str
    retry_count: int
    failed_attempts: List[dict]   # [{entities, shape, reason}, ...]
    entities: List[str]
    entity_uris: List[dict]
    categories: List[dict]        # [{"uri": str, "label": str}, ...]
    shape: str
    check_valid: bool
    check_reason: str
    accepted_shape: Optional[str]
    accepted_entity_uris: Optional[List[dict]]
    accepted_categories: Optional[List[dict]]
    step_times: dict


def make_context_graph(entities_llm, shapes_llm, check_llm, categories_llm=None, log_calls: bool = False) -> StateGraph:
    """
    Builds and compiles the context-generation LangGraph sub-graph.

    Flow: extract → el → dbc → shape → check → (retry or END)
    On check failure: retry up to 3 times passing failed_attempts back to extract.
    After 3 failures: accept the last shape unconditionally.
    """
    from services.entity_extraction import extract_entities
    from services.entity_linking import dbpedia_el
    from services.category_linking import fetch_categories
    from services.shape_generation import generate_shape
    from services.llm_utils import make_shape_check_tool

    _shape_check = make_shape_check_tool(check_llm)

    def extract_node(state: ContextState) -> dict:
        _t0 = time.perf_counter()
        try:
            entities = extract_entities(
                state["nlq"], entities_llm, state["failed_attempts"] or []
            )
        except Exception as e:
            logging.warning(f"[context_graph] extract_entities failed: {e}")
            entities = []
        log_message(step_name="[context] Extracted entities", color="Cyan", messages=[str(entities)])
        st = state["step_times"]
        st["extraction"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"entities": entities, "step_times": st}

    def el_node(state: ContextState) -> dict:
        _t0 = time.perf_counter()
        try:
            entity_uris = dbpedia_el(state["nlq"], state["entities"])
        except Exception as e:
            logging.warning(f"[context_graph] entity_linking failed: {e}")
            entity_uris = []
        log_message(step_name="[context] Entity URIs", color="Cyan", messages=[str(entity_uris)])
        st = state["step_times"]
        st["el"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"entity_uris": entity_uris, "step_times": st}

    def dbc_node(state: ContextState) -> dict:
        _t0 = time.perf_counter()
        try:
            categories = fetch_categories(state["nlq"], state["entities"], llm=categories_llm)
        except Exception as e:
            logging.warning(f"[context_graph] category_linking failed: {e}")
            categories = []
        st = state["step_times"]
        st["dbc"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"categories": categories, "step_times": st}

    def shape_node(state: ContextState) -> dict:
        _t0 = time.perf_counter()
        try:
            shape = generate_shape(state["nlq"], state["entities"], shapes_llm, log_calls=log_calls) or ""
        except Exception as e:
            logging.warning(f"[context_graph] shape_generation failed: {e}")
            shape = ""
        log_message(step_name="[context] Shape generated", color="Cyan", messages=[shape or "(empty)"])
        st = state["step_times"]
        st["shape"].append(f"{time.perf_counter() - _t0:.2f}s")
        return {"shape": shape, "step_times": st}

    def check_node(state: ContextState) -> dict:
        _t0 = time.perf_counter()
        try:
            result = _shape_check.invoke({"nlq": state["nlq"], "shape": state["shape"]})
            valid = result.get("valid", False)
            reason = result.get("reason", "")
        except Exception as e:
            logging.warning(f"[context_graph] shape_check failed: {e}")
            valid = False
            reason = str(e)

        log_message(
            step_name="[context] Shape check",
            color="Cyan",
            messages=[f"valid={valid}", reason],
        )

        st = state["step_times"]
        st["check"].append(f"{time.perf_counter() - _t0:.2f}s")

        if valid:
            return {
                "check_valid": True,
                "check_reason": reason,
                "accepted_shape": state["shape"],
                "accepted_entity_uris": state["entity_uris"],
                "accepted_categories": state["categories"],
                "step_times": st,
            }

        new_retry_count = state["retry_count"] + 1
        new_failed = state["failed_attempts"] + [
            {
                "entities": state["entities"],
                "shape": state["shape"],
                "reason": reason,
            }
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
                step_name="[context] Max retries reached — accepting last shape",
                color="Yellow",
                messages=[],
            )
            updates["accepted_shape"] = state["shape"]
            updates["accepted_entity_uris"] = state["entity_uris"]
            updates["accepted_categories"] = state["categories"]
        return updates

    def check_router(state: ContextState) -> str:
        if state.get("check_valid") or state.get("retry_count", 0) >= 3:
            return END
        return "extract_node"

    builder = StateGraph(ContextState)
    builder.add_node("extract_node", extract_node)
    builder.add_node("el_node", el_node)
    builder.add_node("dbc_node", dbc_node)
    builder.add_node("shape_node", shape_node)
    builder.add_node("check_node", check_node)

    builder.set_entry_point("extract_node")
    builder.add_edge("extract_node", "el_node")
    builder.add_edge("el_node", "dbc_node")
    builder.add_edge("dbc_node", "shape_node")
    builder.add_edge("shape_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        check_router,
        {"extract_node": "extract_node", END: END},
    )

    return builder.compile()
