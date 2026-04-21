from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
from services.log_utils.LogLLMCallbackHandler import LogLLMCallbackHandler
from services.log_utils.log import log_message
from services.translate import translate_question

from typing import List, Callable, Optional

import os
import json
import logging

from services.llm_utils import get_expected_answer_type
from services.entity_extraction import extract_entities
from services.shape_generation_generic import generate_shape_generic, KGConfig
from services.ld_utils import execute, post_process


class LLMAgent:
    """
    Generic LLM agent that converts natural language questions to SPARQL queries
    over any RDF Knowledge Graph.

    Compared to LLMAgentDBpedia, all KG-specific details are injected at
    construction time:
      - kg_config      : endpoint, URI prefixes, namespaces (see KGConfig)
      - prompts        : system / execute_step / feedback_step / last_task prompt dicts
      - icl_dataset_path / icl_faiss_path : paths to the in-context learning data
      - entity_linker  : optional callable(nlq, entity_labels) -> linked
      - tools          : optional list of LangChain tools (e.g. a categories tool)
    """

    def __init__(
        self,
        kg_config: KGConfig,
        prompts: dict,
        icl_dataset_path: str,
        icl_faiss_path: str,
        entity_linker: Optional[Callable] = None,
        model_name: str = "openai/gpt-4o-mini",
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        return_N: int = 5,
        tools: List = [],
        lang: str = "en",
    ):
        """
        Parameters
        ----------
        kg_config:
            KGConfig instance describing the target KG (endpoint, prefixes, …).
        prompts:
            Dict with keys ``"system"``, ``"execute_step"``, ``"feedback_step"``,
            ``"last_task"`` — each value is itself a lang-keyed dict, e.g.
            ``{"en": "…"}``.  Mirrors the structure of the dicts in
            ``prompts/dbpedia.py``.
        icl_dataset_path:
            Path to the JSON file used for in-context learning examples.
            The file must be a list of objects with at least ``"question"``
            and ``"sparql"`` keys and a ``seq_num`` metadata field.
        icl_faiss_path:
            Directory path of the FAISS vector index built from the ICL dataset.
        entity_linker:
            Optional callable with signature ``(nlq: str, entity_labels: list) -> dict``.
            Should return a dict mapping labels to KG URIs.  Pass ``None`` to
            skip entity linking (shape generation still runs via label strings).
        model_name:
            OpenRouter model identifier, e.g. ``"openai/gpt-4o-mini"``.
        embedding_model_name:
            HuggingFace model used to embed ICL examples.
        return_N:
            Number of similar ICL examples to retrieve.
        tools:
            Extra LangChain tools given to the agent executor (e.g. a KG
            categories lookup tool).
        lang:
            Language key used when indexing into the prompt dicts.
        """
        load_dotenv()

        self.kg_config = kg_config
        self.prompts = prompts
        self.lang = lang
        self.embedding_model_name = embedding_model_name
        self.model_name = model_name
        self.entity_linker = entity_linker

        # --- Embeddings ---
        model_kwargs = {'device': 'cpu', 'model_kwargs': {'use_safetensors': False}}
        encode_kwargs = {'normalize_embeddings': False}
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # --- ICL dataset & FAISS index ---
        with open(icl_dataset_path, "r", encoding="utf-8") as f:
            self.icl_json_data = json.load(f)

        self.return_N = return_N
        self.icl_db = FAISS.load_local(
            icl_faiss_path, self.hf_embeddings, allow_dangerous_deserialization=True
        )

        # --- Agent setup ---
        self._base_tools = tools
        self.current_model = model_name

        client = Client()
        self.agent_prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _init_llms(self, model_name: str):
        def _llm(api_key_env: str, **kwargs):
            return ChatOpenAI(
                model=model_name,
                api_key=os.getenv(api_key_env),
                base_url="https://openrouter.ai/api/v1",
                callbacks=[self.log_handler],
                **kwargs,
            )

        self.llm_eat = _llm("mKGQAgent_EAT_LLM")
        self.llm_execution_original = _llm("mKGQAgent_Execution_original_LLM")
        self.entities_llm = _llm("mKGQAgent_Entities_LLM", temperature=0.2, max_tokens=50)
        self.shapes_llm = _llm("mKGQAgent_Shapes_LLM")
        self.translation_llm = _llm("mKGQAgent_Translation_LLM")

        self.tools = list(self._base_tools)

        self.agent_runnable_execution_original = create_tool_calling_agent(
            self.llm_execution_original, self.tools, self.agent_prompt
        )
        self.agent_executor_original = AgentExecutor(
            agent=self.agent_runnable_execution_original,
            tools=self.tools,
            verbose=False,
        )
        self.current_model = model_name

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _translate_step(self, nlq: str) -> str:
        translated = translate_question(nlq, self.translation_llm)
        log_message(step_name="Translated question", color="Yellow", messages=[translated])
        return translated

    def _eat_step(self, chat_history: list, nlq: str):
        try:
            expected_answer_type = get_expected_answer_type(nlq, self.llm_eat)
            eat = expected_answer_type["expected_answer_type"]["eat"]
            chat_history.append(AIMessage(f"Expected answer type: {eat}"))
            log_message(step_name="Expected answer type", color="Yellow", messages=[eat])
        except Exception as e:
            log_message(step_name="Expected answer type failed", color="Red", messages=[str(e)])

    def _get_similar_examples_step(self, chat_history: list, nlq: str):
        icl_message = self.get_similar_examples(nlq)
        chat_history.append(HumanMessage(icl_message))

    def _context_step(self, chat_history: list, nlq: str, shapes_step: bool):
        """Extract entities, optionally link them, generate shape, append context."""

        # Entity extraction
        try:
            entity_labels = extract_entities(nlq, self.entities_llm)
            log_message(step_name="Entity extraction", color="Cyan", messages=[str(entity_labels)])
        except Exception as e:
            entity_labels = []
            log_message(step_name="Entity extraction failed", color="Red", messages=[str(e)])

        # Entity linking (optional)
        linked = {}
        if self.entity_linker is not None:
            try:
                linked = self.entity_linker(nlq, entity_labels)
                log_message(step_name="Entity linking", color="Cyan", messages=[str(linked)])
            except Exception as e:
                log_message(step_name="Entity linking failed", color="Red", messages=[str(e)])

        # Shape generation
        shape = ""
        if shapes_step:
            try:
                shape = generate_shape_generic(
                    nlq, entity_labels, self.shapes_llm, self.kg_config
                )
                log_message(step_name="Shape generation", color="Cyan", messages=[shape])
            except Exception as e:
                shape = ""
                log_message(step_name="Shape generation failed", color="Red", messages=[str(e)])

        # Compile context message
        context_parts = []
        if linked:
            context_parts.append(f"Entity URIs: {json.dumps(linked)}")
        if shape:
            context_parts.append(f"KG shape:\n{shape}")
        if context_parts:
            context_msg = "\n\n".join(context_parts)
            chat_history.append(HumanMessage(content=context_msg))
            log_message(step_name="Context added", color="Yellow", messages=[context_msg])

    def _execute_step(self, task: str, chat_history: list) -> str:
        log_message(step_name="Execute task", color="Cyan", messages=[task])
        try:
            agent_response = self.agent_executor_original.invoke(
                {"input": task, "chat_history": chat_history}
            )
            output = agent_response["output"]
        except Exception as e:
            output = str(e)

        chat_history.append(HumanMessage(task))
        chat_history.append(AIMessage(output))
        log_message(step_name="Execute response", color="Yellow", messages=[output])
        return output

    def _feedback_step(self, chat_history: list, nlq: str) -> tuple:
        current_query = chat_history[-1].content
        feedback_has_results = False
        try:
            feedback = execute(
                query=current_query, endpoint_url=self.kg_config.sparql_endpoint
            )
            if isinstance(feedback, dict) and "error" not in feedback:
                bindings = feedback["results"]["bindings"][:3]
                if bindings:
                    feedback_has_results = True
                feedback = json.dumps(bindings)
        except Exception as e:
            feedback = str(e)

        log_message(step_name="Feedback", color="Yellow", messages=[feedback])

        feedback_task = str(
            self.prompts["feedback_step"][self.lang].format(
                question=nlq,
                query=current_query,
                feedback=feedback,
                last_task=self.prompts["last_task"][self.lang],
            )
        )
        return feedback_task, feedback_has_results

    # ------------------------------------------------------------------
    # ICL retrieval
    # ------------------------------------------------------------------

    def get_similar_examples(self, input_question: str) -> str:
        results = self.icl_db.similarity_search_with_score(input_question, k=self.return_N)
        example = "--- Successful example for in context learning ---"
        for result in results[: self.return_N]:
            idx = result[0].metadata["seq_num"] - 1
            question = self.icl_json_data[idx]["question"]
            sparql = self.icl_json_data[idx]["sparql"]
            example += f"\nInput: {question}\nOutput: {sparql}\n"
            example += "--- End example ---"
        log_message(
            step_name="Similar examples retrieved for ICL", color="Cyan", messages=[example]
        )
        return example

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sparql(
        self,
        input_question: str,
        model_name: str = "openai/gpt-4o-mini",
        log_calls: bool = True,
        shape_step: bool = True,
    ) -> dict:
        """
        Convert a natural language question to a SPARQL query.

        Returns
        -------
        dict
            ``translated_question``, ``query``, ``prompt_tokens``,
            ``completion_tokens``, ``requests``.
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name)

            chat_history = [SystemMessage(content=self.prompts["system"][self.lang])]

            self.log_handler.reset(input_question, enabled=log_calls)
            with get_openai_callback() as cb:
                translated_question = self._translate_step(input_question)
                self._eat_step(chat_history, translated_question)
                self._get_similar_examples_step(chat_history, translated_question)
                self._context_step(chat_history, translated_question, shape_step)

                self._execute_step(
                    str(self.prompts["execute_step"][self.lang].format(nlq=translated_question)),
                    chat_history,
                )

                feedback_task, has_results = self._feedback_step(chat_history, translated_question)
                if not has_results:
                    self._execute_step(feedback_task, chat_history)

            sparql_result = chat_history[-1].content
            generated_query = post_process(sparql_result)
            log_message(
                step_name="Generated SPARQL query", color="Green", messages=[generated_query]
            )
            self.log_handler._flush_to_file(generated_query)

            return {
                "translated_question": translated_question,
                "query": generated_query,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "requests": cb.successful_requests,
            }

        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return {
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
            }


# ---------------------------------------------------------------------------
# Convenience factory: DBpedia agent (drop-in equivalent of LLMAgentDBpedia)
# ---------------------------------------------------------------------------

def make_dbpedia_agent(
    model_name: str = "openai/gpt-4o-mini",
    embedding_model_name: str = "intfloat/multilingual-e5-large",
    return_N: int = 5,
    lang: str = "en",
    tools: List = [],
) -> LLMAgent:
    """
    Build an LLMAgent pre-configured for DBpedia — equivalent to
    instantiating the old LLMAgentDBpedia class.
    """
    from services.shape_generation_generic import DBPEDIA_CONFIG
    from services.entity_linking import dbpedia_el
    from services.llm_utils import dbpedia_categories_tool
    from prompts.dbpedia import (
        system_prompt,
        last_task,
        feedback_step_dict,
        execute_step_prompt,
    )

    prompts = {
        "system": system_prompt,
        "execute_step": execute_step_prompt,
        "feedback_step": feedback_step_dict,
        "last_task": last_task,
    }

    icl_dataset_path = f"./data/datasets/qald_9_plus_train_dbpedia_{lang}.json"
    icl_faiss_vdb = icl_dataset_path.split("/")[-1].replace(".json", "")
    icl_faiss_path = os.path.join(".", "data", "experience-pool", icl_faiss_vdb)

    return LLMAgent(
        kg_config=DBPEDIA_CONFIG,
        prompts=prompts,
        icl_dataset_path=icl_dataset_path,
        icl_faiss_path=icl_faiss_path,
        entity_linker=dbpedia_el,
        model_name=model_name,
        embedding_model_name=embedding_model_name,
        return_N=return_N,
        tools=[dbpedia_categories_tool] + list(tools),
        lang=lang,
    )


if __name__ == "__main__":
    agent = make_dbpedia_agent(
        model_name="openai/gpt-4o-mini",
        embedding_model_name="intfloat/multilingual-e5-large",
        return_N=5,
        lang="en",
    )

    text = "Who is the author of the book 'The Great Gatsby'?"
    query = agent.generate_sparql(text)

    print(f"Input: {text}")
    print(f"Output: {query}")
