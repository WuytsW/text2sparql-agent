from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv


from typing import List

import os
import json
import logging
from datetime import datetime

from services.llm_utils import dbpedia_el, Plan, get_expected_answer_type
from services.ld_utils import execute, post_process
from model.agent import PlanExecute
from prompts.dbpedia import (
    system_prompt,
    last_task,
    planner_prompt_dct,
    feedback_step_dict
)

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

RESET   = "\033[0m"

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)



class LogLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self._log_entries = []

    def reset(self, question: str):
        self.call_count = 0
        self._log_entries = []
        self._question = question
        self._start_time = datetime.now().isoformat()

    def _flush_to_file(self, sparql: str, log_path: str = "logs/llm_calls.json"):
        record = {
            "time": self._start_time,
            "question": self._question,
            "total_llm_calls": self.call_count,
            "sparql": sparql,
            "calls": self._log_entries,
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        existing = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(record)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def on_chat_model_start(self, serialized, _messages, **kwargs):
        self.call_count += 1
        model = serialized.get("kwargs", {}).get("model_name", "unknown")
        msgs = [[{"type": m.type, "content": m.content} for m in grp] for grp in _messages]
        self._log_entries.append({"call": self.call_count, "model": model, "messages": msgs})
        logging.info(f"{BLUE}[LLM API call #{self.call_count}] model={model} \n messages={_messages}{RESET}")

    def on_llm_end(self, response, **kwargs):
        gen = response.generations[0][0]
        text = gen.text or (gen.message.content if hasattr(gen, "message") else "")
        if text:
            if self._log_entries:
                self._log_entries[-1]["response"] = text
            logging.info(f"{GREEN}[LLM response #{self.call_count}]:\n{text}{RESET}")


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
        model_kwargs = {'device': 'cpu'}
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
        self.tools = tools
        self.llm_callback = LogLLMCallbackHandler()

        self.plan_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback],
        ).with_structured_output(Plan)


        client = Client()
        prompt = client.pull_prompt("hwchase17/openai-functions-agent")

        # Choose the LLM that will drive the agent
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback],
        )

        # Construct the OpenAI Functions agent
        self.agent_runnable = create_tool_calling_agent(self.llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent_runnable, tools=tools, verbose=False, return_intermediate_steps=True
        )

        self.app = None

        ### EMD Initialize agent

    def _plan_step(self, state: PlanExecute):
        try:
            plan = self.plan_llm.invoke(planner_prompt_dct[self.lang].format(objective=state["input"]))
            logging.info(f"{MAGENTA}[plan_step] Generated plan: {plan.steps}{RESET}")
            return {"plan": plan.steps + [last_task]}

        except Exception as e:
            plan = [last_task]
            return {"plan": plan}
        
    def _execute_step(self, state: PlanExecute):
        if state["gave_feedback"]:
            task = state["feedback_task"]
        else:
            task = state["plan"].pop(0)

        try:
            agent_response = self.agent_executor.invoke({"input": task, "chat_history": state['chat_history']})
            state['chat_history'].append(AIMessage(agent_response['output'])) # update chat history
        except Exception as e:
            state['chat_history'].append(AIMessage(str(e)))
            agent_response = {"output": str(e), "intermediate_steps": "No"}

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
        
        return {
            "feedback_task": str(task.format(question=state["input"], query=state['chat_history'][-1].content, feedback=feedback, last_task=last_task)),
            "gave_feedback": True
        }
    
    def _eat_step(self, state: PlanExecute):
        try:
            expected_answer_type = get_expected_answer_type(state['input'], self.llm)
            state['chat_history'].append(AIMessage(expected_answer_type["expected_answer_type"]["eat"])) # update chat history
        except Exception as e:
            pass

        return {
            "gave_feedback": state["gave_feedback"]
        }
    
    def _init_workflow(self):
        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", self._plan_step)

        # Add the eat step
        workflow.add_node("eat", self._eat_step)

        # Add the execution step
        workflow.add_node("agent", self._execute_step)

        # Add the feedback step
        workflow.add_node("feedback", self._feedback_step)


        # Add the new step
        



        workflow.set_entry_point("planner")

        # From plan we go to agent
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

        # From feedback we go to agent
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
  
    def generate_sparql(self, input_question: str) -> str:
        """
        Convert a natural language question to a SPARQL query
        
        Args:
            question: The natural language question
            dataset: The dataset URL to query against
            
        Returns:
            A SPARQL query string
        """
        try:
            self.llm_callback.reset(input_question)

            if self.app is None:
                self._init_workflow()

            results = self.icl_db.similarity_search_with_score(input_question, k=self.return_N)

            example = "--- Successful example for in context learning ---"

            for result in results[:self.return_N]:
                idx = result[0].metadata['seq_num'] - 1
                question = self.icl_json_data[idx]["question"]
                sparql = self.icl_json_data[idx]["sparql"]

                example += f"""

                                Input: {question}
                                Output: {sparql}

                                """
                
                example += "--- End example ---"

            agent_result = self.app.invoke(
                {"input": input_question, "chat_history": [SystemMessage(content=f"""{system_prompt[self.lang]}      
                {example}""")],
                "gave_feedback": False}
            )
        
            sparql_result = agent_result['chat_history'][-1].content

            logging.info(f"{CYAN}[Total LLM calls]: {self.llm_callback.call_count}{RESET}")
            self.llm_callback._flush_to_file(sparql_result)
            logging.info(f"Generated SPARQL query: {sparql_result}")

            generated_query = post_process(sparql_result)

            return generated_query
        except Exception as e:
            logging.error(f"Error in generate_sparql: {e}")
            return """SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o
}
LIMIT 10"""



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
    
