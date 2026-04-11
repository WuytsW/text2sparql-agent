from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
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

from services.llm_utils import dbpedia_el, Plan, get_expected_answer_type, make_extract_entities_tool, make_generate_shape_tool
from services.ld_utils import execute, post_process
from model.agent import PlanExecute
from prompts.dbpedia import (
    system_prompt,
    last_task,
    planner_prompt_dct,
    feedback_step_dict
)


class LLMAgentDBpedia:
    """
    Implementation of an LLM agent that converts natural language to SPARQL over DBpedia.
    This will be replaced with an actual LLM implementation later.
    """
    
    def __init__(
            self,
            model_name: str = "openai/gpt-4o-mini",
            embedding_model_name: str = "intfloat/multilingual-e5-large",
            return_N: int = 5,
            tools: List = [dbpedia_el],
            lang: str = "en"
        ):

        load_dotenv()
        """Initialize the LLM agent with any required configurations"""
        self.model_name = "text-to-sparql-mock"
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
        self.icl_db = FAISS.load_local(icl_faiss_vdb_path, self.hf_embeddings, allow_dangerous_deserialization = True)
        ### END Load ICL VDB

        ### START Initialize agent
        self._base_tools = tools  # static tools without LLM dependency
        self.current_model = model_name
        self.compact_mode = False

        client = Client()
        self.agent_prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        self._init_llms(model_name)
        self.app = None
        self.log_handler = LogLLMCallbackHandler()

        ### END Initialize agent

    def _init_llms(self, model_name: str):
        self.plan_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("mKGQAgent_Run_no_changes"),
            base_url="https://openrouter.ai/api/v1",
        ).with_structured_output(Plan)

        self.llm_eat = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_EAT_LLM"),
            base_url="https://openrouter.ai/api/v1"
        )

        self.llm_execution_original = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_original_LLM"),
            base_url="https://openrouter.ai/api/v1"
        )

        self.llm_execution_compact = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_compact_LLM"),
            base_url="https://openrouter.ai/api/v1"
        )

        self.entities_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Entities_LLM"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=50,
        )

        self.shapes_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Shapes_LLM"),
            base_url="https://openrouter.ai/api/v1",
        )

        self.translation_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Translation_LLM"),
            base_url="https://openrouter.ai/api/v1",
        )

        self.tools = [
            make_extract_entities_tool(self.entities_llm),
            make_generate_shape_tool(self.shapes_llm),
        ] + self._base_tools

        self.agent_runnable_execution_original = create_tool_calling_agent(self.llm_execution_original, self.tools, self.agent_prompt)
        self.agent_runnable_execution_compact = create_tool_calling_agent(self.llm_execution_compact, self.tools, self.agent_prompt)
        self.agent_executor_original = AgentExecutor(
            agent=self.agent_runnable_execution_original, tools=self.tools, verbose=False, return_intermediate_steps=True
        )
        self.agent_executor_compact = AgentExecutor(
            agent=self.agent_runnable_execution_compact, tools=self.tools, verbose=False, return_intermediate_steps=True
        )

        self.current_model = model_name

    def _plan_step(self, state: PlanExecute):
        try:
            plan = self.plan_llm.invoke(planner_prompt_dct[self.lang].format(objective=state["input"]))
            log_message(step_name="Planning", color="Magenta", messages=[plan.steps])
            return {"plan": plan.steps + [last_task]}
        except Exception as e:
            plan = [last_task]
            return {"plan": plan}
        
    def _append_tool_trace(self, state: PlanExecute, task: str, agent_response: dict):
        """Append the full tool call trace from an AgentExecutor response to chat_history.

        Adds: HumanMessage(task) → AIMessage(tool_calls) → ToolMessage(s) → AIMessage(output)
        so that subsequent steps can see which tools were already called.
        """
        state['chat_history'].append(HumanMessage(task))

        seen_msg_ids = set()
        for action, observation in agent_response.get('intermediate_steps', []):
            if hasattr(action, 'message_log'):
                for msg in action.message_log:
                    if id(msg) not in seen_msg_ids:
                        seen_msg_ids.add(id(msg))
                        state['chat_history'].append(msg)
            if hasattr(action, 'tool_call_id'):
                state['chat_history'].append(
                    ToolMessage(content=str(observation), tool_call_id=action.tool_call_id)
                )

        state['chat_history'].append(AIMessage(agent_response['output']))

    def _execute_step_compact(self, state: PlanExecute):
        print("Compact")
        if state["gave_feedback"]:
            task = state["feedback_task"]
        else:
            all_steps = list(state["plan"])
            state["plan"].clear()
            task = "Complete all of the following steps in order:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_steps))

        log_message(step_name="Execute (Compact) task", color="Magenta", messages=[task])

        try:
            agent_response = self.agent_executor_compact.invoke({"input": task, "chat_history": state['chat_history']})
            self._append_tool_trace(state, task, agent_response)
        except Exception as e:
            state['chat_history'].append(HumanMessage(task))
            state['chat_history'].append(AIMessage(str(e)))
            agent_response = {"output": str(e), "intermediate_steps": []}

        log_message(step_name="Execute (Compact) response", color="Magenta", messages=[agent_response["output"]])

        return {
            "past_steps": [task, agent_response["output"]],
            "intermediate_steps": [task, agent_response["intermediate_steps"]],
            "gave_feedback": state["gave_feedback"]
        }

    def _execute_step_original(self, state: PlanExecute):
        print("Original")
        if state["gave_feedback"]:
            task = state["feedback_task"]
        else:
            task = f"User question: {state['input']}\n\nTask: {state['plan'].pop(0)}"

        log_message(step_name="Execute (Original) task", color="Magenta", messages=[task])

        try:
            agent_response = self.agent_executor_original.invoke({"input": task, "chat_history": state['chat_history']})
            self._append_tool_trace(state, task, agent_response)
        except Exception as e:
            state['chat_history'].append(HumanMessage(task))
            state['chat_history'].append(AIMessage(str(e)))
            agent_response = {"output": str(e), "intermediate_steps": []}

        log_message(step_name="Execute (Original) response", color="Magenta", messages=[agent_response["output"]])

        return {
            "past_steps": [task, agent_response["output"]],
            "intermediate_steps": [task, agent_response["intermediate_steps"]],
            "gave_feedback": state["gave_feedback"]
        }

    def _feedback_step(self, state: PlanExecute):
        task = feedback_step_dict[self.lang]
        try:
            feedback = execute(query=state['chat_history'][-1].content, endpoint_url=self.sparql_endpoint)
            if type(feedback) == dict and "error" not in feedback.keys():
                feedback = json.dumps(feedback['results']['bindings'][:3])
        except Exception as e:
            feedback = str(e)
        
        log_message(step_name="Feedback", color="Magenta", messages=[feedback])

        return {
            "feedback_task": str(task.format(question=state["input"], query=state['chat_history'][-1].content, feedback=feedback, last_task=last_task[self.lang])),
            "gave_feedback": True
        }
    
    def _eat_step(self, state: PlanExecute):
        try:
            expected_answer_type = get_expected_answer_type(state['input'], self.llm_eat)

            log_message(step_name="eat", color="Magenta", messages=[expected_answer_type["expected_answer_type"]["eat"]])

            state['chat_history'].append(AIMessage(expected_answer_type["expected_answer_type"]["eat"])) # update chat history
        except Exception as e:
            log_message(step_name="eat", color="Magenta", messages=[str(e)])

        return {
            "gave_feedback": state["gave_feedback"]
        }
    
    def _init_workflow(self):
        workflow = StateGraph(PlanExecute)

        # Select the execution step (compact or original based on self.compact_mode)
        execute_fn = self._execute_step_compact if self.compact_mode else self._execute_step_original

        workflow.add_node("planner", self._plan_step)
        workflow.add_node("eat", self._eat_step)
        workflow.add_node("agent", execute_fn)
        workflow.add_node("feedback", self._feedback_step)

        # Define the edges between the nodes
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "eat")
        workflow.add_edge("eat", "agent")
        workflow.add_conditional_edges(
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self._feedback_router,
            {
                "feedback": "feedback",
                "agent": "agent",
                END: END
            }
        )
        workflow.add_edge("feedback", "agent")

        self.app = workflow.compile()

        return True

    def _feedback_router(self, state: PlanExecute):
        if len(state["plan"]) > 0:
            return "agent"
        if len(state["plan"]) == 0 and state["gave_feedback"] == False:
            return "feedback"
        if len(state["plan"]) == 0 and state["gave_feedback"] == True:
            return END
  

    def get_similar_examples(self, input_question: str):
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

    def generate_sparql(self, input_question: str, model_name: str = "openai/gpt-4o-mini", compact: bool = False, log_calls: bool = False) -> dict:
        """
        Convert a natural language question to a SPARQL query.

        Args:
            input_question: The natural language question
            model_name: OpenRouter model identifier (e.g. "openai/gpt-4o-mini")
            compact: If True, execute all plan steps in a single agent call
            log_calls: If True, log LLM calls

        Returns:
            Dict with query, prompt_tokens, completion_tokens, requests
        """
        try:
            if model_name != self.current_model or compact != self.compact_mode:
                self._init_llms(model_name)
                self.compact_mode = compact
                self.app = None

            if self.app is None:
                self._init_workflow()

            log_message(step_name="Input question", color="Yellow", messages=[input_question])
            translated_question = translate_question(input_question, self.translation_llm)
            log_message(step_name="Translated question", color="Yellow", messages=[translated_question])

            self.log_handler.reset(translated_question, enabled=log_calls)
            with get_openai_callback() as cb:
                agent_result = self.app.invoke(
                    {"input": translated_question, "chat_history": [SystemMessage(content=f"""{system_prompt[self.lang]}
                    {self.get_similar_examples(translated_question)}""")],
                    "gave_feedback": False},
                    config={"callbacks": [self.log_handler]}
                )

            sparql_result = agent_result['chat_history'][-1].content

            generated_query = post_process(sparql_result)
            log_message(step_name="Generated SPARQL query", color="Yellow", messages=[generated_query])

            self.log_handler._flush_to_file(generated_query)

            return {
                "query": generated_query,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "requests": cb.successful_requests
            }
        
        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return {
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
                "prompt_tokens":0,
                "completion_tokens": 0,
                "requests": 0
            }



if __name__ == "__main__":   

    dbpedia_agent = LLMAgentDBpedia(
        model_name="openai/gpt-4o-mini",
        embedding_model_name="intfloat/multilingual-e5-large",
        return_N=5,
        tools=[dbpedia_el],
        lang="en"
    )


    text = "Who is the author of the book 'The Great Gatsby'?"         

    query = dbpedia_agent.generate_sparql(text)
        
    print(f"Input: {text}")
    print(f"Output: {query}")
