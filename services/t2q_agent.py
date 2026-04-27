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

from typing import List

import os
import json
import logging

from services.llm_utils import get_expected_answer_type
from services.entity_linking import dbpedia_el
from services.entity_extraction import extract_entities
from services.shape_generation import generate_shape
from services.ld_utils import execute, post_process
from services.context_tools import validate_shape_with_llm
from prompts.dbpedia import (
    system_prompt,
    last_task,
    feedback_step_dict,
    execute_step_prompt,
)


class T2QAgent:
    """
    Text-to-SPARQL agent over DBpedia with a validated context loop and
    multi-iteration SPARQL refinement loop.

    Pipeline:
      1. Translate question to English
      2. Expected Answer Type (EAT) classification
      3. In-context learning examples (FAISS retrieval)
      4. Context loop (max 3 iter): extract entities → link → shape → LLM validation
      5. SPARQL loop (max 5 iter): agent generates SPARQL → execute → feedback
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

        model_kwargs = {'device': 'cpu', 'model_kwargs': {'use_safetensors': False}}
        encode_kwargs = {'normalize_embeddings': False}
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        icl_file_path = f"./data/datasets/qald_9_plus_train_dbpedia_{lang}.json"
        with open(icl_file_path, "r", encoding='utf-8') as f:
            self.icl_json_data = json.load(f)

        icl_faiss_vdb = icl_file_path.split("/")[-1].replace(".json", "")
        icl_faiss_vdb_path = os.path.join(".", "data", "experience-pool", icl_faiss_vdb)
        self.return_N = return_N
        self.icl_db = FAISS.load_local(icl_faiss_vdb_path, self.hf_embeddings, allow_dangerous_deserialization=True)

        self._base_tools = tools
        self.current_model = model_name

        client = Client()
        self.agent_prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)

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

        # Re-uses shapes API key; split to mKGQAgent_Validation_LLM if separate rate limit needed
        self.llm_validation = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Shapes_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.tools = self._base_tools

        self.agent_runnable_execution_original = create_tool_calling_agent(
            self.llm_execution_original, self.tools, self.agent_prompt
        )
        self.agent_executor_original = AgentExecutor(
            agent=self.agent_runnable_execution_original, tools=self.tools, verbose=False
        )

        self.current_model = model_name

    # -------------------------------------------------------------------------
    # Shared helpers (identical to LLMAgentDBpedia)
    # -------------------------------------------------------------------------

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

    def _translate_step(self, nlq: str) -> str:
        translated_question = translate_question(nlq, self.translation_llm)
        log_message(step_name="Translated question", color="Yellow", messages=[translated_question])
        return translated_question

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

    # -------------------------------------------------------------------------
    # New: context loop with LLM-validated shape
    # -------------------------------------------------------------------------

    def _context_loop(self, chat_history: list, nlq: str, max_iterations: int = 3):
        """Extract entities → link → shape → validate, up to max_iterations times.

        On each failed validation the LLM suggestion is fed back as a hint to the
        entity extraction step of the next iteration. The best attempt (first valid
        or, if none, the first iteration) is appended to chat_history.
        """
        extra_hint = ""
        best_shape: str | None = None
        best_linked: list = []

        for iteration in range(1, max_iterations + 1):
            log_message(
                step_name=f"Context loop {iteration}/{max_iterations}",
                color="Cyan", messages=[]
            )

            # Step 1: Extract entities (inject hint from previous validation if any)
            extraction_input = (
                nlq if not extra_hint
                else f"{nlq}\nHint for entity extraction: {extra_hint}"
            )
            try:
                entity_labels = extract_entities(extraction_input, self.entities_llm)
                log_message(step_name="Entity extraction", color="Cyan", messages=[str(entity_labels)])
            except Exception as e:
                entity_labels = []
                log_message(step_name="Entity extraction failed", color="Red", messages=[str(e)])

            # Step 2: Link entities via Falcon (always use original nlq for Falcon)
            try:
                linked = dbpedia_el(nlq, entity_labels)
                log_message(step_name="Entity linking", color="Cyan", messages=[str(linked)])
            except Exception as e:
                linked = []
                log_message(step_name="Entity linking failed", color="Red", messages=[str(e)])

            # Step 3: Generate shape
            try:
                entity_uris = {k: v for item in linked for k, v in item.items()}
                shape = generate_shape(nlq, entity_labels, self.shapes_llm, entity_uris=entity_uris)
                log_message(step_name="Shape generation", color="Cyan", messages=[str(shape)])
            except Exception as e:
                shape = None
                log_message(step_name="Shape generation failed", color="Red", messages=[str(e)])

            # Step 4: Validate shape usefulness
            validation = validate_shape_with_llm(
                question=nlq,
                shape=shape or "",
                entity_labels=entity_labels,
                linked=linked,
                llm=self.llm_validation
            )
            log_message(
                step_name=f"Shape validation iter {iteration}",
                color="Yellow",
                messages=[
                    f"useful={validation['useful']}",
                    f"reason={validation['reason']}"
                ]
            )

            # Seed best from first iteration; override whenever we get a valid shape
            if iteration == 1 or validation["useful"]:
                best_shape = shape
                best_linked = linked

            if validation["useful"]:
                log_message(
                    step_name="Shape accepted",
                    color="Green",
                    messages=[f"Iteration {iteration}"]
                )
                break

            extra_hint = validation["suggestion"]
            if iteration == max_iterations:
                log_message(
                    step_name="Context loop exhausted, using best attempt",
                    color="Yellow", messages=[]
                )

        # Append best context to chat_history
        context_parts = []
        if best_linked:
            context_parts.append(f"Entity URIs from DBpedia: {json.dumps(best_linked)}")
        if best_shape:
            context_parts.append(f"DBpedia shape:\n{best_shape}")
        if context_parts:
            context_msg = "\n\n".join(context_parts)
            chat_history.append(HumanMessage(content=context_msg))
            log_message(
                step_name="Context added to chat history",
                color="Yellow", messages=[context_msg]
            )

    # -------------------------------------------------------------------------
    # New: SPARQL generation + verification loop
    # -------------------------------------------------------------------------

    def _sparql_loop(
        self,
        task: str,
        chat_history: list,
        nlq: str,
        max_iterations: int = 5
    ):
        """Generate SPARQL, execute, and refine up to max_iterations times.

        Uses the existing feedback_step_dict prompt to build refinement tasks.
        Breaks early when the query returns non-empty results.
        """
        for iteration in range(1, max_iterations + 1):
            log_message(
                step_name=f"SPARQL loop {iteration}/{max_iterations}",
                color="Cyan", messages=[task[:300]]
            )

            try:
                output = self.agent_executor_original.invoke(
                    {"input": task, "chat_history": chat_history}
                )["output"]
            except Exception as e:
                output = str(e)

            chat_history.append(HumanMessage(task))
            chat_history.append(AIMessage(output))
            log_message(step_name="SPARQL agent output", color="Yellow", messages=[output])

            # Execute generated SPARQL on DBpedia
            has_results = False
            try:
                result = execute(query=output, endpoint_url=self.sparql_endpoint)
                if isinstance(result, dict) and "error" not in result:
                    bindings = result["results"]["bindings"][:3]
                    has_results = bool(bindings)
                    feedback_str = json.dumps(bindings)
                else:
                    feedback_str = json.dumps(result)
            except Exception as e:
                feedback_str = str(e)

            log_message(step_name="SPARQL execution feedback", color="Yellow", messages=[feedback_str])

            if has_results:
                log_message(
                    step_name="SPARQL loop: results found",
                    color="Green", messages=[f"Iteration {iteration}"]
                )
                break

            if iteration == max_iterations:
                log_message(
                    step_name="SPARQL loop exhausted",
                    color="Yellow",
                    messages=[f"Using last output after {max_iterations} iterations"]
                )
                break

            task = str(feedback_step_dict[self.lang].format(
                question=nlq,
                query=output,
                feedback=feedback_str,
                last_task=last_task[self.lang]
            ))

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def generate_sparql(
        self,
        input_question: str,
        model_name: str = "openai/gpt-4o-mini",
        log_calls: bool = True,
        shape_step: bool = True
    ) -> dict:
        """Convert a natural language question to a SPARQL query over DBpedia.

        Args:
            input_question: The natural language question (any language).
            model_name: OpenRouter model identifier (e.g. "openai/gpt-4o-mini").
            log_calls: If True, log LLM calls to logs/llm_calls.json.

        Returns:
            Dict with translated_question, query, prompt_tokens, completion_tokens, requests.
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name)

            chat_history = [SystemMessage(content=system_prompt[self.lang])]
            self.log_handler.reset(input_question, enabled=log_calls)

            with get_openai_callback() as cb:
                translated = self._translate_step(input_question)
                self._eat_step(chat_history, translated)
                self._get_similar_examples_step(chat_history, translated)
                self._context_loop(chat_history, translated)
                initial_task = str(execute_step_prompt[self.lang].format(nlq=translated))
                self._sparql_loop(initial_task, chat_history, translated)

            generated_query = post_process(chat_history[-1].content)
            log_message(step_name="Generated SPARQL query", color="Green", messages=[generated_query])
            self.log_handler._flush_to_file(generated_query)

            return {
                "translated_question": translated,
                "query": generated_query,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "requests": cb.successful_requests
            }

        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return {
                "translated_question": input_question,
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0
            }


if __name__ == "__main__":
    agent = T2QAgent(
        model_name="openai/gpt-4o-mini",
        embedding_model_name="intfloat/multilingual-e5-large",
        return_N=5,
        tools=[],
        lang="en"
    )

    text = "Who is the author of the book 'The Great Gatsby'?"
    result = agent.generate_sparql(text)

    print(f"Input: {text}")
    print(f"Output: {result['query']}")
