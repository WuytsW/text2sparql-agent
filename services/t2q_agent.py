from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

from services.llm_utils import get_expected_answer_type, make_extract_entities_tool, make_generate_shape_tool
from services.context_tools import make_entity_linking_tool, make_execute_sparql_tool
from services.ld_utils import execute, post_process
from prompts.dbpedia import system_prompt, execute_agent_system_prompt


class T2QAgent:
    """
    Text-to-SPARQL agent over DBpedia.

    Pipeline:
      1. Translate question to English
      2. Expected Answer Type (EAT) classification
      3. In-context learning examples (FAISS retrieval)
      4. Execute step: tool-calling agent that autonomously extracts entities,
         links them, generates a shape, writes SPARQL, and verifies by execution
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

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)

    def _init_llms(self, model_name: str):
        self.llm_eat = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_EAT_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.log_handler]
        )

        self.llm_execution = ChatOpenAI(
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

        # Build the four tools for the execute agent
        execute_agent_tools = [
            make_extract_entities_tool(self.entities_llm),
            make_entity_linking_tool(),
            make_generate_shape_tool(self.shapes_llm),
            make_execute_sparql_tool(self.sparql_endpoint),
        ] + self._base_tools

        # Build prompt locally — no LangSmith dependency.
        # SystemMessage is used directly to avoid ChatPromptTemplate parsing
        # the literal { } in the prompt text as template variables.
        execute_agent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=execute_agent_system_prompt[self.lang]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent_runnable = create_tool_calling_agent(
            self.llm_execution, execute_agent_tools, execute_agent_prompt
        )
        self.agent_executor = AgentExecutor(
            agent=agent_runnable,
            tools=execute_agent_tools,
            verbose=False,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        self.current_model = model_name

    # -------------------------------------------------------------------------
    # Helpers
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

    def _execute_step(self, chat_history: list, nlq: str, max_iterations: int = 3) -> str:
        """Run the tool-calling agent in a loop until a working SPARQL query is found.

        Each iteration invokes the full agent (which can call any tool multiple times).
        If the output query returns no results, feedback is added to chat_history and
        the agent is retried, up to max_iterations times.
        """
        task = (
            f"Question: {nlq}\n\n"
            "Follow ALL mandatory steps from your instructions:\n"
            "1. extract_entities_tool\n"
            "2. dbpedia_el_tool\n"
            "3. generate_shape_tool\n"
            "4. Construct the SPARQL query\n"
            "5. execute_sparql_tool — you MUST call this; if empty/error, revise and call again\n"
            "Output only the final verified SPARQL query."
        )

        output = ""
        for iteration in range(1, max_iterations + 1):
            log_message(
                step_name=f"Execute loop {iteration}/{max_iterations}",
                color="Cyan", messages=[nlq]
            )
            try:
                output = self.agent_executor.invoke(
                    {"input": task, "chat_history": chat_history}
                )["output"]
            except Exception as e:
                log_message(step_name="Execute step failed", color="Red", messages=[str(e)])
                output = str(e)

            log_message(step_name="Agent output", color="Yellow", messages=[output])

            # Verify the query by running it
            try:
                result = execute(query=output, endpoint_url=self.sparql_endpoint)
                if isinstance(result, dict) and "error" not in result:
                    bindings = result.get("results", {}).get("bindings", [])
                    if bindings:
                        log_message(
                            step_name="Valid query found",
                            color="Green", messages=[f"Iteration {iteration}"]
                        )
                        return output
                feedback_str = json.dumps(result)
            except Exception as e:
                feedback_str = str(e)

            log_message(step_name="Query produced no results", color="Yellow", messages=[feedback_str])

            if iteration == max_iterations:
                break

            # Feed the failure back so the next iteration has context
            chat_history.append(AIMessage(content=output))
            chat_history.append(HumanMessage(
                content=(
                    f"The query above returned no results or an error:\n{feedback_str}\n\n"
                    "Please use your tools again (re-check entities, shape, or try different "
                    "properties) and produce a corrected SPARQL query."
                )
            ))
            task = (
                f"Question: {nlq}\n\n"
                "Your previous query failed (see the error/empty result in context). "
                "Use your tools to investigate and produce a corrected, verified SPARQL query."
            )

        log_message(step_name="Execute loop exhausted", color="Yellow", messages=["Using last output"])
        return output

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
            shape_step: Kept for API compatibility; the execute agent decides
                        whether to call the shape tool autonomously.

        Returns:
            Dict with translated_question, query, prompt_tokens, completion_tokens, requests.
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name)

            self.log_handler.reset(input_question, enabled=log_calls)

            with get_openai_callback() as cb:
                translated = self._translate_step(input_question)
                chat_history = [SystemMessage(content=system_prompt[self.lang])]
                self._eat_step(chat_history, translated)
                self._get_similar_examples_step(chat_history, translated)
                final_query = self._execute_step(chat_history, translated)

            generated_query = post_process(final_query)
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
