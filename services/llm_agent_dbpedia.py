from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

from services.translate import translate_question

from typing import List

import os
import json
import logging
from services.LogLLMCallbackHandler import LogLLMCallbackHandler
from  services.shape_generation import generate_shape
from services.entity_extraction import extract_entities

from services.llm_utils import dbpedia_el, Plan, get_expected_answer_type
from services.ld_utils import execute, post_process
from model.agent import PlanExecute
from prompts.dbpedia import (
    system_prompt,
    last_task,
    planner_prompt_dct,
    feedback_step_dict
)



logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

RESET   = "\033[0m"

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
            api_key=os.getenv("mKGQAgent_Plan_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback]
        ).with_structured_output(Plan)


        client = Client()
        prompt = client.pull_prompt("hwchase17/openai-functions-agent")
        logging.info(f"{CYAN}Pulled prompt for agent construction: {prompt}{RESET}")

        # Choose the LLM that will drive the agent
        self.llm_eat = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_EAT_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback]
        )

        self.llm_execution_original = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_original_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback]
        )

        self.llm_execution_compact = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Execution_compact_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback]
        )

        self.entities_llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("mKGQAgent_Entities_LLM"),
            base_url="https://openrouter.ai/api/v1",
            callbacks=[self.llm_callback],
            temperature=0.2,
            max_tokens=50,
        )


        # Construct the OpenAI Functions agent
        self.agent_runnable_execution_original = create_tool_calling_agent(self.llm_execution_original, tools, prompt)
        self.agent_runnable_execution_compact = create_tool_calling_agent(self.llm_execution_compact, tools, prompt)


        self.agent_executor_original = AgentExecutor(
            agent=self.agent_runnable_execution_original, tools=tools, verbose=False, return_intermediate_steps=True
        )
        self.agent_executor_compact = AgentExecutor(
            agent=self.agent_runnable_execution_compact, tools=tools, verbose=False, return_intermediate_steps=True
        )
        self.app = None

        ### EMD Initialize agent

    def _plan_step(self, state: PlanExecute):
        logging.info(f"{MAGENTA}[plan_step]{RESET}")
        try:
            plan = self.plan_llm.invoke(planner_prompt_dct[self.lang].format(objective=state["input"]))
            # logging.info(f"{MAGENTA}[plan_step] Generated plan: {plan.steps}{RESET}")
            return {"plan": plan.steps + [last_task[self.lang]]}

        except Exception as e:
            plan = [last_task[self.lang]]
            return {"plan": plan}
        
    def _execute_step_compact(self, state: PlanExecute):
        logging.info(f"{MAGENTA}[execute_step]{RESET}")
        if state["gave_feedback"]:
            task = state["feedback_task"]
        else:
            all_steps = list(state["plan"])
            state["plan"].clear()
            task = "Complete all of the following steps in order:\n" + "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(all_steps)
            )

        chat_history = state['chat_history']
        if state.get("shape"):
            shape_message = SystemMessage(content=(
                "--- Entity shapes (ShEx) for the entities in this question ---\n"
                + "\n".join(state["shape"])
                + "\n--- End entity shapes ---\n"
                "\nIMPORTANT INSTRUCTIONS FOR SPARQL GENERATION:\n"
                "- Only use properties that are explicitly listed in the entity shapes above.\n"
                "- Do NOT invent properties. Use only what appears in the shape.\n"
                "- dcterms:subject ONLY links to Category: resources (e.g. dbc:Presidents_of_the_United_States). NEVER use dcterms:subject with a regular resource.\n"
                "- To find entities related to another entity, follow the property FROM that entity (e.g. dbr:Vietnam_War dbo:commander ?uri).\n"
                "- To filter by type of person, prefer dcterms:subject with a Category over rdf:type with an ontology class.\n"
            ))
            chat_history = list(state['chat_history']) + [shape_message]

        try:
            agent_response = self.agent_executor_compact.invoke({"input": task, "chat_history": chat_history})
            state['chat_history'].append(AIMessage(agent_response['output'])) # update chat history
        except Exception as e:
            state['chat_history'].append(AIMessage(str(e)))
            agent_response = {"output": str(e), "intermediate_steps": "No"}

        return {
            "past_steps": [task, agent_response["output"]],
            "intermediate_steps": [task, agent_response["intermediate_steps"]],
            "gave_feedback": state["gave_feedback"]
        }

    def _execute_step(self, state: PlanExecute):
        logging.info(f"{MAGENTA}[execute_step]{RESET}")
        logging.info(f"{MAGENTA}Current state: {state}{RESET}")
        if state["gave_feedback"]:
            task = state["feedback_task"]
        else:
            task = state["plan"].pop(0)

        try:
            agent_response = self.agent_executor_original.invoke({"input": task, "chat_history": state['chat_history']})
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
        logging.info(f"{MAGENTA}[feedback_step]{RESET}")
        task = feedback_step_dict[self.lang]
        try:
            feedback = execute(query=state['chat_history'][-1].content, endpoint_url=self.sparql_endpoint)
            logging.info(f"{MAGENTA}Execution feedback: {feedback}{RESET}")
            if type(feedback) == dict and "error" not in feedback.keys():
                feedback = json.dumps(feedback['results']['bindings'][:3])
        except Exception as e:
            feedback = str(e)
        
        return {
            "feedback_task": str(task.format(question=state["input"], query=state['chat_history'][-1].content, feedback=feedback, last_task=last_task[self.lang])),
            "gave_feedback": True
        }
    
    def _eat_step(self, state: PlanExecute):
        logging.info(f"{MAGENTA}[eat_step]{RESET}")
        try:
            expected_answer_type = get_expected_answer_type(state['input'], self.llm_eat)
            state['chat_history'].append(AIMessage(expected_answer_type["expected_answer_type"]["eat"])) # update chat history
        except Exception as e:
            pass

        return {
            "gave_feedback": state["gave_feedback"]
        }

    def _shaper_step(self, state: PlanExecute):
        logging.info(f"{MAGENTA}[shaper_step]{RESET}")

        entities = extract_entities(state['input'], self.entities_llm)
        shape = generate_shape(entities)

        return {
            "shape": [shape] if shape else [],
            "gave_feedback": state["gave_feedback"]
        }
    
    def _init_workflow(self):
        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", self._plan_step)

        # Add the eat step
        workflow.add_node("eat", self._eat_step)

        workflow.add_node("shaper", self._shaper_step)

        # Add the execution step
        workflow.add_node("agent", self._execute_step_compact)

        # Add the feedback step
        workflow.add_node("feedback", self._feedback_step)

        workflow.set_entry_point("planner")

        # From plan we go to agent
        workflow.add_edge("planner", "eat")
        workflow.add_edge("eat", "shaper")
        workflow.add_edge("shaper", "agent")

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
  
    def generate_sparql(self, input_question: str, log: bool = False) -> str:
        """
        Convert a natural language question to a SPARQL query
        
        Args:
            question: The natural language question
            dataset: The dataset URL to query against
            
        Returns:
            A SPARQL query string
        """
        try:
            # Translate the input question to English if necessary, using the LLM-based translator
            logging.info(f"{YELLOW}Received question: \n{input_question}{RESET}")
            translated_input_question = translate_question(input_question)
            logging.info(f"{YELLOW}Translated question: \n{translated_input_question}{RESET}")
            
            
            self.llm_callback.reset(translated_input_question, enabled=log)

            if self.app is None:
                self._init_workflow()

            results = self.icl_db.similarity_search_with_score(translated_input_question, k=self.return_N)

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
                {"input": translated_input_question, "chat_history": [SystemMessage(content=f"""{system_prompt[self.lang]}
                {example}""")],
                "gave_feedback": False,
                "shape": []}
            )
        
            sparql_result = agent_result['chat_history'][-1].content

            if log:
                logging.info(f"{YELLOW}Raw generated SPARQL query: \n{sparql_result}{RESET}")
                logging.info(f"{CYAN}[Total LLM calls]: {self.llm_callback.call_count}{RESET}")
                self.llm_callback._flush_to_file(sparql_result)
                logging.info(f"Generated SPARQL query: {sparql_result}")

            generated_query = post_process(sparql_result)
            logging.info(f"{YELLOW}Post-processed generated SPARQL query: \n{generated_query}{RESET}")
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
    
