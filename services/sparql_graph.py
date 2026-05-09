import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from services.ld_utils import execute
from services.log_utils.log import log_message
from prompts.dbpedia import generation_prompt, check_result_prompt


class SparqlLoopState(TypedDict):
    question: str
    chat_history: list
    sparql_endpoint: str
    lang: str
    query: str
    exec_result: str
    check_ok: bool
    suggestions: str
    attempt_count: int


def make_sparql_graph(generation_llm, check_llm) -> StateGraph:
    """
    Builds and compiles the SPARQL generation loop as a LangGraph sub-graph.
    Flow: generate_node → execute_node → check_node → (retry or END)
    Up to 3 attempts; suggestions from failed checks are fed back into generate_node.
    """

    def generate_node(state: SparqlLoopState) -> dict:
        suggestions = state["suggestions"]
        log_message(step_name="generate_sparql called", color="Cyan",
                    messages=[f"suggestions: {suggestions}" if suggestions else "first attempt"])
        suggestions_block = f"\nPrior feedback:\n{suggestions}" if suggestions else ""
        messages = state["chat_history"] + [HumanMessage(
            generation_prompt[state["lang"]].format(
                question=state["question"], suggestions_block=suggestions_block
            )
        )]
        query = generation_llm.invoke(messages).content
        log_message(step_name="generate_sparql result", color="Cyan", messages=[query])
        return {"query": query}

    def execute_node(state: SparqlLoopState) -> dict:
        query = state["query"]
        log_message(step_name="execute_sparql called", color="Cyan", messages=[query])
        try:
            raw = execute(query=query, endpoint_url=state["sparql_endpoint"])
            if isinstance(raw, dict) and "error" not in raw:
                bindings = raw.get("results", {}).get("bindings", [])[:3]
                exec_result = json.dumps(bindings)
            else:
                exec_result = json.dumps(raw)
        except Exception as e:
            exec_result = json.dumps({"error": str(e)})
        log_message(step_name="execute_sparql result", color="Cyan", messages=[exec_result])
        return {"exec_result": exec_result}

    def check_node(state: SparqlLoopState) -> dict:
        query = state["query"]
        exec_result = state["exec_result"]
        log_message(step_name="check_result called", color="Cyan",
                    messages=[f"query: {query}", f"result: {exec_result}"])
        prompt_text = check_result_prompt[state["lang"]].format(
            question=state["question"], query=query, execution_result=exec_result
        )
        response = check_llm.invoke([HumanMessage(prompt_text)]).content
        log_message(step_name="check_result response", color="Cyan", messages=[response])
        try:
            check_json = json.loads(response)
        except json.JSONDecodeError:
            check_json = {"ok": False, "suggestions": "Could not parse check response"}
        return {
            "check_ok": check_json.get("ok", False),
            "suggestions": check_json.get("suggestions", ""),
            "attempt_count": state["attempt_count"] + 1,
        }

    def check_router(state: SparqlLoopState) -> str:
        if state.get("check_ok") or state.get("attempt_count", 0) >= 3:
            return END
        return "generate_node"

    builder = StateGraph(SparqlLoopState)
    builder.add_node("generate_node", generate_node)
    builder.add_node("execute_node", execute_node)
    builder.add_node("check_node", check_node)

    builder.set_entry_point("generate_node")
    builder.add_edge("generate_node", "execute_node")
    builder.add_edge("execute_node", "check_node")
    builder.add_conditional_edges(
        "check_node", check_router, {"generate_node": "generate_node", END: END}
    )

    return builder.compile()
