from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from services.log_utils.LogLLMCallbackHandler import LogLLMCallbackHandler
from services.log_utils.log import log_message, set_question_log
from services.translate import translate_question

from typing import List

import os
import json
import logging
import time

from services.llm_utils import (
    dbpedia_categories_tool,
    get_expected_answer_type,
    correct_query_prefixes,
)
from services.context_graph import make_context_graph
from services.ld_utils import execute, post_process
from model.agent import PlanExecute
from prompts.dbpedia import (
    system_prompt,
    last_task,
    feedback_step_dict,
)


class LLMAgentDBpedia:
    """
    Implementation of an LLM agent that converts natural language to SPARQL over DBpedia.
    """

    def __init__(
            self,
            model_name: str = "openai/gpt-4o-mini",
            embedding_model_name: str = "intfloat/multilingual-e5-large",
            return_N: int = 5,
            tools: List = [],
            lang: str = "en"
        ):

        load_dotenv()
        self.sparql_endpoint = "https://dbpedia.org/sparql"
        self.lang = lang
        self.embedding_model_name = embedding_model_name
        self.model_name = model_name

        ### START Initialize embeddings
        model_kwargs = {'device': 'cpu', 'model_kwargs': {'use_safetensors': False}}
        encode_kwargs = {'normalize_embeddings': False}
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        ### END Initialize embeddings

        ### START Load ICL VDB
        icl_file_path = f"./data/datasets/qald_9_plus_train_dbpedia_{lang}.json"
        with open(icl_file_path, "r", encoding='utf-8') as f:
            self.icl_json_data = json.load(f)

        icl_faiss_vdb = icl_file_path.split("/")[-1].replace(".json", "")
        icl_faiss_vdb_path = os.path.join(".", "data", "experience-pool", icl_faiss_vdb)
        self.return_N = return_N
        self.icl_db = FAISS.load_local(icl_faiss_vdb_path, self.hf_embeddings, allow_dangerous_deserialization=True)
        ### END Load ICL VDB

        ### START Initialize agent
        self._base_tools = tools
        self.current_model = model_name
        self.app = None

        client = Client()
        self.agent_prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)
        self._step_times: list = []
        self._agent_call_count: int = 0
        ### END Initialize agent

    def _init_llms(self, model_name: str):
        self.llm_eat = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_EAT_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.llm_execution_original = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_original_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.entities_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Entities_LLM"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=50,
            callbacks=[self.log_handler]
        )

        self.shapes_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Shapes_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.translation_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Translation_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.shape_check_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Context_LLM", os.getenv("mKGQAgent_Execution_original_LLM")),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self._context_graph = make_context_graph(
            self.entities_llm, self.shapes_llm, self.shape_check_llm
        )

        self.tools = [dbpedia_categories_tool] + self._base_tools

        self.agent_runnable = create_tool_calling_agent(self.llm_execution_original, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent_runnable, tools=self.tools, verbose=False
        )

        self.app = None  # reset workflow on model change
        self.current_model = model_name

    def _translate_step(self, nlq: str):
        try:
            translated_question = translate_question(nlq, self.translation_llm)
        except Exception as e:
            logging.warning(f"Translation failed, using original question: {e}")
            translated_question = nlq
        log_message(step_name="Translated question", color="Yellow", messages=[translated_question])
        return translated_question

    def _get_similar_examples_step(self, chat_history: list, nlq: str):
        icl_message = self.get_similar_examples(nlq)
        chat_history.append(HumanMessage(icl_message))

    def _eat_step(self, chat_history: list, nlq: str):
        """Classify expected answer type and append it to chat_history."""
        try:
            expected_answer_type = get_expected_answer_type(nlq, self.llm_eat)
            eat = expected_answer_type["expected_answer_type"]["eat"]
            eat_message = f"Expected answer type: {eat}"
            chat_history.append(AIMessage(eat_message))
            log_message(step_name="Expected answer type", color="Yellow", messages=[eat])
        except Exception as e:
            log_message(step_name="Expected answer type failed", color="Red", messages=[str(e)])

    def _plan_step(self, state: PlanExecute):
        _t0 = time.perf_counter()
        result = {"plan": ["generate SPARQL query with the context provided in the chat history", last_task[self.lang]]}
        self._step_times.append(f"planner: {time.perf_counter() - _t0:.2f}s")
        return result

    def _context_step(self, state: PlanExecute):
        _t0 = time.perf_counter()
        result = self._context_graph.invoke({
            "nlq": state["input"],
            "retry_count": 0,
            "failed_attempts": [],
            "entities": [],
            "entity_uris": [],
            "shape": "",
            "check_valid": False,
            "check_reason": "",
            "accepted_shape": None,
            "accepted_entity_uris": None,
        })

        accepted_shape = result.get("accepted_shape") or result.get("shape") or "No shape generated."
        entity_uris = result.get("accepted_entity_uris") or result.get("entity_uris") or []

        context_msg = (
            f"Entity URIs: {json.dumps(entity_uris)}\n"
            f"Shape:\n{accepted_shape}"
        )
        log_message(step_name="Context generated", color="Cyan", messages=[context_msg])
        self._step_times.append(f"context: {time.perf_counter() - _t0:.2f}s")
        return {"chat_history": state["chat_history"] + [AIMessage(content=context_msg)]}

    def _agent_step(self, state: PlanExecute):
        self._agent_call_count += 1
        _t0 = time.perf_counter()
        task = state["feedback_task"] if state["gave_feedback"] else state["plan"].pop(0)
        log_message(step_name="Agent task", color="Cyan", messages=[str(task)])

        task_input = f"User question: '{state['input']}'\nTask: {task}"

        try:
            agent_response = self.agent_executor.invoke(
                {"input": task_input, "chat_history": state["chat_history"]}
            )
            output = agent_response["output"]
        except Exception as e:
            output = str(e)
            agent_response = {"output": output, "intermediate_steps": []}

        state["chat_history"].append(AIMessage(output))
        log_message(step_name="Agent response", color="Yellow", messages=[output])
        self._step_times.append(f"agent_{self._agent_call_count}: {time.perf_counter() - _t0:.2f}s")

        return {
            "past_steps": [task, output],
            "intermediate_steps": [task, agent_response.get("intermediate_steps", [])],
            "gave_feedback": state["gave_feedback"],
        }

    def _feedback_step(self, state: PlanExecute):
        _t0 = time.perf_counter()
        current_query = state["chat_history"][-1].content
        feedback_has_results = False
        feedback_is_timeout = False
        try:
            feedback = execute(query=current_query, endpoint_url=self.sparql_endpoint)
            if isinstance(feedback, dict) and "error" not in feedback:
                bindings = feedback.get("results", {}).get("bindings", [])[:3]
                if bindings:
                    feedback_has_results = True
                feedback = json.dumps(bindings)
            elif isinstance(feedback, dict) and "timed out" in str(feedback.get("error", "")).lower():
                feedback_is_timeout = True
                feedback = json.dumps(feedback)
        except Exception as e:
            feedback = str(e)
            if "timed out" in str(e).lower():
                feedback_is_timeout = True

        log_message(step_name="Feedback", color="Yellow", messages=[str(feedback)])

        feedback_task = str(feedback_step_dict[self.lang].format(
            question=state["input"],
            query=current_query,
            feedback=feedback,
            last_task=last_task[self.lang]
        ))
        self._step_times.append(f"feedback: {time.perf_counter() - _t0:.2f}s")
        return {
            "feedback_task": feedback_task,
            "gave_feedback": True,
            "feedback_has_results": feedback_has_results,
            "feedback_is_timeout": feedback_is_timeout,
        }

    def _feedback_router(self, state: PlanExecute):
        if len(state["plan"]) > 0:
            return "agent"
        if not state["gave_feedback"]:
            return "feedback"
        if state.get("feedback_is_timeout", False):
            return END
        return END

    def _init_workflow(self):
        workflow = StateGraph(PlanExecute)

        workflow.add_node("planner", self._plan_step)
        workflow.add_node("context", self._context_step)
        workflow.add_node("agent", self._agent_step)
        workflow.add_node("feedback", self._feedback_step)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "context")
        workflow.add_edge("context", "agent")

        workflow.add_conditional_edges(
            "agent",
            self._feedback_router,
            {"feedback": "feedback", "agent": "agent", END: END}
        )

        workflow.add_edge("feedback", "agent")

        self.app = workflow.compile()

    def get_similar_examples(self, input_question: str) -> str:
        results = self.icl_db.similarity_search_with_score(input_question, k=self.return_N)

        example = "--- Successful example for in context learning ---"
        for result in results[:self.return_N]:
            idx = result[0].metadata['seq_num'] - 1
            question = self.icl_json_data[idx]["question"]
            sparql = self.icl_json_data[idx]["sparql"]
            example += f"""\nInput: {question}\nOutput: {sparql}\n"""
            example += "--- End example ---"
        log_message(step_name="Similar examples retrieved for ICL", color="Cyan", messages=[example])
        return example

    def generate_sparql(self, input_question: str, model_name: str = "openai/gpt-4o-mini", log_calls: bool = True, shape_step: bool = True) -> dict:
        """
        Convert a natural language question to a SPARQL query.

        Args:
            input_question: The natural language question
            model_name: OpenRouter model identifier (e.g. "openai/gpt-4o-mini")
            log_calls: If True, log LLM calls
            shape_step: Kept for API compatibility (shape generation is always active via generate_context_tool)

        Returns:
            Dict with translated_question, query, prompt_tokens, completion_tokens, requests
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name)
            if self.app is None:
                self._init_workflow()

            self._step_times = []
            self._agent_call_count = 0
            self.log_handler.reset(input_question, enabled=log_calls)
            set_question_log(input_question)
            log_message(step_name="Original question", color="Yellow", messages=[input_question])
            chat_history = [SystemMessage(content=system_prompt[self.lang])]

            with get_openai_callback() as cb:
                _t0 = time.perf_counter()
                translated_question = self._translate_step(input_question)
                self._step_times.append(f"translation: {time.perf_counter() - _t0:.2f}s")

                _t0 = time.perf_counter()
                self._eat_step(chat_history, translated_question)
                self._step_times.append(f"eat: {time.perf_counter() - _t0:.2f}s")

                _t0 = time.perf_counter()
                self._get_similar_examples_step(chat_history, translated_question)
                self._step_times.append(f"icl: {time.perf_counter() - _t0:.2f}s")

                result = self.app.invoke(
                    {
                        "input": translated_question,
                        "chat_history": chat_history,
                        "gave_feedback": False,
                        "plan": [],
                        "past_steps": [],
                        "intermediate_steps": [],
                        "feedback_task": "",
                        "feedback_has_results": False,
                        "feedback_is_timeout": False,
                    },
                    config={"callbacks": [self.log_handler]}
                )

            sparql_result = result["chat_history"][-1].content
            generated_query = post_process(sparql_result)
            generated_query = correct_query_prefixes(generated_query, self.shape_check_llm)
            log_message(step_name="Generated SPARQL query", color="Green", messages=[generated_query])

            return {
                "translated_question": translated_question,
                "query": generated_query,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "requests": cb.successful_requests,
                "step_times": self._step_times,
            }

        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return {
                "translated_question": input_question,
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
                "step_times": self._step_times,
            }


if __name__ == "__main__":

    dbpedia_agent = LLMAgentDBpedia(
        model_name="openai/gpt-4o-mini",
        embedding_model_name="intfloat/multilingual-e5-large",
        return_N=5,
        tools=[],
        lang="en"
    )

    text = "Who is the author of the book 'The Great Gatsby'?"

    query = dbpedia_agent.generate_sparql(text)

    print(f"Input: {text}")
    print(f"Output: {query}")
