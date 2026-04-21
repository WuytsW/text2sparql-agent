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

from services.llm_utils import dbpedia_categories_tool, get_expected_answer_type
from services.entity_linking import dbpedia_el
from services.entity_extraction import extract_entities
from services.shape_generation import generate_shape
from services.shape_generation_generic import generate_shape_generic, DBPEDIA_CONFIG
from services.ld_utils import execute, post_process
from prompts.dbpedia import (
    system_prompt,
    last_task,
    feedback_step_dict,
    execute_step_prompt
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

        client = Client()
        self.agent_prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        self.log_handler = LogLLMCallbackHandler()
        self._init_llms(model_name)
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

        self.tools = [dbpedia_categories_tool] + self._base_tools

        self.agent_runnable_execution_original = create_tool_calling_agent(self.llm_execution_original, self.tools, self.agent_prompt)
        self.agent_executor_original = AgentExecutor(
            agent=self.agent_runnable_execution_original, tools=self.tools, verbose=False
        )

        self.current_model = model_name


    def _translate_step(self, nlq: str):
        translated_question = translate_question(nlq, self.translation_llm)
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


    def _context_step(self, chat_history: list, nlq: str, shapes_step: bool):
        """Extract entities, link via DBpedia EL, generate shape, append context to chat_history."""
        
        # Entity extraction
        try:
            entity_labels = extract_entities(nlq, self.entities_llm)
            log_message(step_name="Entity extraction", color="Cyan", messages=[str(entity_labels)])
        except Exception as e:
            entity_labels = []
            log_message(step_name="Entity extraction failed", color="Red", messages=[str(e)])
        
        # Entity linking
        try:
            linked = dbpedia_el(nlq, entity_labels)
            log_message(step_name="Entity linking", color="Cyan", messages=[linked])
        except Exception as e:
            linked = []
            log_message(step_name="Entity linking failed", color="Red", messages=[str(e)])

        # Shape generation
        shape = ""
        if shapes_step:
            try:
                # shape = generate_shape(nlq, entity_labels, self.shapes_llm)
                shape = generate_shape_generic(nlq, entity_labels, self.shapes_llm, DBPEDIA_CONFIG)
                log_message(step_name="Shape generation", color="Cyan", messages=[shape])
            except Exception as e:
                shape = ""
                log_message(step_name="Shape generation failed", color="Red", messages=[str(e)])


        # Compile context message    
        context_parts = []
        if linked:
            context_parts.append(f"Entity URIs from DBpedia: {json.dumps(linked)}")
        if shape:
            context_parts.append(f"DBpedia shape:\n{shape}")
        if context_parts:
            context_msg = "\n\n".join(context_parts)
            chat_history.append(HumanMessage(content=context_msg))
            log_message(step_name="Entity linking context added", color="Yellow", messages=[context_msg])

    def _execute_step(self, task: str, chat_history: list) -> str:
        """Run the agent executor and append the result to chat_history. Returns agent output."""
        log_message(step_name="Execute task", color="Cyan", messages=[task])

        try:
            agent_response = self.agent_executor_original.invoke({"input": task, "chat_history": chat_history})
            output = agent_response["output"]
        except Exception as e:
            output = str(e)

        chat_history.append(HumanMessage(task))
        chat_history.append(AIMessage(output))
        log_message(step_name="Execute response", color="Yellow", messages=[output])
        return output

    def _feedback_step(self, chat_history: list, nlq: str) -> tuple:
        """Execute the current SPARQL query and return (feedback_task, has_results)."""
        current_query = chat_history[-1].content
        feedback_has_results = False
        try:
            feedback = execute(query=current_query, endpoint_url=self.sparql_endpoint)
            if isinstance(feedback, dict) and "error" not in feedback:
                bindings = feedback['results']['bindings'][:3]
                if bindings:
                    feedback_has_results = True
                feedback = json.dumps(bindings)
        except Exception as e:
            feedback = str(e)

        log_message(step_name="Feedback", color="Yellow", messages=[feedback])

        feedback_task = str(feedback_step_dict[self.lang].format(
            question=nlq,
            query=current_query,
            feedback=feedback,
            last_task=last_task[self.lang]
        ))
        return feedback_task, feedback_has_results

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

        Returns:
            Dict with translated_question, query, prompt_tokens, completion_tokens, requests
        """
        try:
            if model_name != self.current_model:
                self._init_llms(model_name)


            chat_history = [SystemMessage(content=system_prompt[self.lang])]

            self.log_handler.reset(input_question, enabled=log_calls)
            with get_openai_callback() as cb:
                translated_question = self._translate_step(input_question)
                self._eat_step(chat_history, translated_question)
                self._get_similar_examples_step(chat_history, translated_question)
                self._context_step(chat_history, translated_question, shape_step)

                self._execute_step(str(execute_step_prompt[self.lang].format(nlq=translated_question)), chat_history)

                feedback_task, has_results = self._feedback_step(chat_history, translated_question)
                if not has_results:
                    self._execute_step(feedback_task, chat_history)




            sparql_result = chat_history[-1].content
            generated_query = post_process(sparql_result)
            log_message(step_name="Generated SPARQL query", color="Green", messages=[generated_query])
            self.log_handler._flush_to_file(generated_query)

            return {
                "translated_question": translated_question,
                "query": generated_query,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "requests": cb.successful_requests
            }

        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return {
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0
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
