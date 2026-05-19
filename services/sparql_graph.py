import json
from typing import TypedDict

from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from services.ld_utils import execute
from services.log_utils.log import log_message
from prompts.dbpedia import generation_prompt, check_result_prompt, sparql_agent_prompt


# ---------------------------------------------------------------------------
# Tool-calling agent
# ---------------------------------------------------------------------------

class ExecuteSPARQLInput(BaseModel):
    query: str = Field(description="The SPARQL query string to execute against DBpedia")


def make_sparql_agent(generation_llm, sparql_endpoint: str, lang: str = "en"):
    @tool("execute_sparql", args_schema=ExecuteSPARQLInput)
    def execute_sparql(query: str) -> str:
        """Execute a SPARQL query against DBpedia. Returns [Query] and [Result] fields; result is bindings (first 3), boolean (ASK), or error."""
        log_message(step_name="execute_sparql called", color="Cyan", messages=[query])
        try:
            raw = execute(query=query, endpoint_url=sparql_endpoint)
            if isinstance(raw, dict) and "error" not in raw:
                if "boolean" in raw:
                    bindings_repr = json.dumps({"boolean": raw["boolean"]})
                else:
                    bindings_repr = json.dumps(raw.get("results", {}).get("bindings", [])[:3])
            else:
                bindings_repr = json.dumps(raw)
        except Exception as e:
            bindings_repr = json.dumps({"error": str(e)})
        result_payload = f"[Query]: {query}\n[Result]: {bindings_repr}"
        log_message(step_name="execute_sparql result", color="Cyan", messages=[result_payload])
        return result_payload

    tools = [execute_sparql]

    prompt = ChatPromptTemplate.from_messages([
        ("system", sparql_agent_prompt[lang]),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(generation_llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=4, verbose=False)

    class _GuardedExecutor:
        """Wraps AgentExecutor to guarantee at least one execute_sparql call."""

        def invoke(self, inputs: dict) -> dict:
            result = executor.invoke(inputs)
            if not result.get("intermediate_steps"):
                # Agent skipped tool calls — force execution now so the result is verified
                query = result.get("output", "").strip()
                if query:
                    execute_sparql.invoke({"query": query})
            return result

    return _GuardedExecutor()


# ---------------------------------------------------------------------------
# StateGraph (original)
# ---------------------------------------------------------------------------

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
                if "boolean" in raw:
                    exec_result = json.dumps({"boolean": raw["boolean"]})
                else:
                    exec_result = json.dumps(raw.get("results", {}).get("bindings", [])[:3])
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
