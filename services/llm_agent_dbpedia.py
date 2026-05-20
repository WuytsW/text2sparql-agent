from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from services.log_utils.LogLLMCallbackHandler import LogLLMCallbackHandler
from services.log_utils.log import log_message, set_question_log
from services.translate import translate_question

import os
import json
import logging
import time

from services.llm_utils import (
    get_expected_answer_type,
    correct_query_prefixes,
)
from services.context_graph_dbpedia import make_context_graph
from services.sparql_graph import make_sparql_agent, make_sparql_graph
from services.ld_utils import post_process
from model.agent import PlanExecute
from prompts.dbpedia import system_prompt


class LLMAgentDBpedia:
    """
    Implementation of an LLM agent that converts natural language to SPARQL over DBpedia.
    """

    def __init__(
            self,
            model_name: str = "openai/gpt-4o-mini",
            embedding_model_name: str = "intfloat/multilingual-e5-large",
            return_N: int = 5,
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
        self.current_model = model_name
        self.app = None

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)
        self._step_times: list = []
        ### END Initialize agent

    def _init_llms(self, model_name: str, log_calls: bool = False):
        # OpenRouter routes qwen models to Novita's /completions endpoint by default,
        # but Novita only supports /chat/completions for these models. Ignoring Novita
        # forces OpenRouter to pick a provider that handles chat completions correctly.
        _or_kwargs = {"extra_body": {"provider": {"ignore": ["Novita"], "require_parameters": True}}}

        self.llm_eat = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_EAT_LLM"),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.llm_execution_original = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_original_LLM"),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.entities_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Entities_LLM"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=50,
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.profile_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Shapes_LLM"),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.translation_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Translation_LLM"),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.profile_check_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Context_LLM", os.getenv("mKGQAgent_Execution_original_LLM")),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.categories_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Categories_LLM"),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self.check_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Check_LLM", os.getenv("mKGQAgent_Context_LLM", os.getenv("mKGQAgent_Execution_original_LLM"))),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=_or_kwargs,
            callbacks=[self.log_handler]
        )

        self._context_graph = make_context_graph(
            self.entities_llm, self.profile_llm, self.profile_check_llm,
            categories_llm=self.categories_llm, log_calls=log_calls
        )

        self._sparql_agent = make_sparql_agent(self.llm_execution_original, self.sparql_endpoint, self.lang)
        self._sparql_graph = make_sparql_graph(self.llm_execution_original, self.check_llm)

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

    def _context_step(self, state: PlanExecute):
        result = self._context_graph.invoke({
            "nlq": state["input"],
            "retry_count": 0,
            "failed_attempts": [],
            "entities": [],
            "entity_uris": [],
            "categories": [],
            "entity_profile": "",
            "check_valid": False,
            "check_reason": "",
            "accepted_entity_profile": None,
            "accepted_entity_uris": None,
            "accepted_categories": None,
            "step_times": {"extraction": [], "el": [], "dbc": [], "entity_profile": [], "check": []},
        })

        accepted_entity_profile = result.get("accepted_entity_profile") or result.get("entity_profile") or "No entity profile generated."
        entity_uris = result.get("accepted_entity_uris") or result.get("entity_uris") or []
        categories = result.get("accepted_categories") or result.get("categories") or []

        annotated_lines = []
        for line in accepted_entity_profile.splitlines():
            if line.lstrip().startswith("dbp:"):
                annotated_lines.append(line + "  [USE dbp:, NOT dbo:]")
            else:
                annotated_lines.append(line)
        annotated_entity_profile = "\n".join(annotated_lines)

        context_msg = (
            f"Entity URIs: {json.dumps(entity_uris)}\n"
            f"Entity Profile:\n{annotated_entity_profile}"
        )
        if categories:
            cats_str = "\n".join(f"  {c['uri']}  ({c['label']})" for c in categories)
            context_msg += f"\nDBpedia Categories (dbc:):\n{cats_str}"
        log_message(step_name="Context generated", color="Yellow", messages=[context_msg])
        self._step_times.append({"context": result.get("step_times", {})})
        return {"chat_history": state["chat_history"] + [AIMessage(content=context_msg)]}

    def _sparql_loop_step(self, state: PlanExecute):
        _t0 = time.perf_counter()
        # Agent invoke (make_sparql_agent)

        final_query = self._call_sparql_agent_or_graph(state["input"], state["chat_history"], agent_mode=True)

        
        log_message(step_name="SPARQL loop result", color="Yellow", messages=[final_query])
        self._step_times.append(f"sparql_loop: {time.perf_counter() - _t0:.2f}s")
        return {"chat_history": state["chat_history"] + [AIMessage(final_query)]}

    def _call_sparql_agent_or_graph(self, question: str, chat_history: list, agent_mode: bool = True):
        # Agent invoke (make_sparql_agent)
        if(agent_mode):
            result = self._sparql_agent.invoke({
                "question": question,
                "chat_history": chat_history,
            })
            final_query = result.get("output", "")
        else:
            result = self._sparql_graph.invoke({
                "question": question,
                "chat_history": chat_history,
                "sparql_endpoint": self.sparql_endpoint,
                "lang": self.lang,
                "query": "",
                "exec_result": "",
                "check_ok": False,
                "suggestions": "",
                "attempt_count": 0,
            })
            final_query = result.get("query", "")
        return final_query

    def _init_workflow(self):
        workflow = StateGraph(PlanExecute)

        workflow.add_node("context", self._context_step)
        workflow.add_node("sparql_loop", self._sparql_loop_step)

        workflow.set_entry_point("context")
        workflow.add_edge("context", "sparql_loop")
        workflow.add_edge("sparql_loop", END)

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
        log_message(step_name="Similar examples retrieved for ICL", color="Yellow", messages=[example])
        return example

    def generate_sparql(self, input_question: str, model_name: str = "openai/gpt-4o-mini", log_calls: bool = True, entity_profile_step: bool = True) -> dict:
        """
        Convert a natural language question to a SPARQL query.

        Args:
            input_question: The natural language question
            model_name: OpenRouter model identifier (e.g. "openai/gpt-4o-mini")
            log_calls: If True, log LLM calls
            entity_profile_step: Kept for API compatibility (entity profile generation is always active via generate_context_tool)

        Returns:
            Dict with translated_question, query, prompt_tokens, completion_tokens, requests
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name, log_calls=log_calls)
            if self.app is None:
                self._init_workflow()

            self._step_times = []
            self.log_handler.reset(input_question, enabled=log_calls)
            set_question_log(input_question)
            log_message(step_name="Original question", color="Green", messages=[input_question])
            chat_history = [SystemMessage(content=system_prompt[self.lang])]

            with get_openai_callback() as cb:
                _t0 = time.perf_counter()
                translated_question = self._translate_step(input_question)
                self._step_times.append(f"translation: {time.perf_counter() - _t0:.2f}s")

                #_t0 = time.perf_counter()
                #self._eat_step(chat_history, translated_question)
                #self._step_times.append(f"eat: {time.perf_counter() - _t0:.2f}s")

                _t0 = time.perf_counter()
                self._get_similar_examples_step(chat_history, translated_question)
                self._step_times.append(f"icl: {time.perf_counter() - _t0:.2f}s")

                result = self.app.invoke(
                    {
                        "input": translated_question,
                        "chat_history": chat_history,
                    },
                    config={"callbacks": [self.log_handler]}
                )

            sparql_result = result["chat_history"][-1].content
            generated_query = post_process(sparql_result)
            # generated_query = correct_query_prefixes(generated_query, self.profile_check_llm)
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
            logging.error(f"Error in generate_sparql: {e}", exc_info=True)
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
        lang="en"
    )

    text = "Who is the author of the book 'The Great Gatsby'?"

    query = dbpedia_agent.generate_sparql(text)

    print(f"Input: {text}")
    print(f"Output: {query}")
