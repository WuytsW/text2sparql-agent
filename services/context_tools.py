import json
import logging
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from services.entity_linking import dbpedia_el
from prompts.dbpedia import shape_validation_prompt


class DBpediaELToolInput(BaseModel):
    ne_list: list = Field(description="List of named entity strings to link to DBpedia URIs")


def make_entity_linking_tool(nlq: str):
    """Factory returning a LangChain tool that wraps dbpedia_el, closing over nlq."""
    @tool("dbpedia_el_tool", args_schema=DBpediaELToolInput)
    def dbpedia_el_tool(ne_list: list) -> list:
        """Links named entities to DBpedia URIs via Falcon. Returns [{"label": "URI"}, ...]"""
        return dbpedia_el(nlq, ne_list)
    return dbpedia_el_tool


def validate_shape_with_llm(
    question: str,
    shape: str,
    entity_labels: list,
    linked: list,
    llm
) -> dict:
    """Ask LLM whether the generated shape is useful for answering the question.

    Returns {"useful": bool, "reason": str, "suggestion": str}.
    On parse failure returns {"useful": False, "reason": "...", "suggestion": ""}.
    """
    shape_text = shape if shape else "(no shape generated)"
    prompt_text = shape_validation_prompt["en"].format(
        question=question,
        entity_labels=json.dumps(entity_labels),
        linked_uris=json.dumps(linked),
        shape=shape_text
    )
    raw = ""
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content.strip()
        result = json.loads(raw)
        return {
            "useful": bool(result.get("useful", False)),
            "reason": str(result.get("reason", "")),
            "suggestion": str(result.get("suggestion", ""))
        }
    except Exception as e:
        logging.warning(f"[validate_shape_with_llm] Parse failed: {e}, raw={raw!r}")
        return {"useful": False, "reason": f"Validation parse error: {e}", "suggestion": ""}
