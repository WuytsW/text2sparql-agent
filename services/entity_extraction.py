import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from services.log_utils.LogLLMCallbackHandler import LogLLMCallbackHandler
from prompts.dbpedia import entities_extraction_prompt
from services.log_utils.log import log_message

load_dotenv(dotenv_path=".env")

def extract_entities(question, llm):
    """
    Extracts entities from the given question using an LLM and resolves them against a SPARQL endpoint.
    """

    """
    Uses an LLM to extract the most relevant entities from a natural language query.
    Cleans and parses the extracted entity names into a clean Python list of strings.
    """

    user_prompt = entities_extraction_prompt["en"].format(nlq=question)

    response = llm.invoke([
        SystemMessage(content="You are an expert in extracting named entities from questions."),
        HumanMessage(content=user_prompt)
    ])

    raw_response = response.content.strip()

    # Strip any outer quotes wrapping the entire response, then split on commas
    cleaned = raw_response.strip('"').strip("'")
    entities = [e.strip().strip('"').strip("'") for e in cleaned.split(",") if e.strip()]

    entities = [e for e in entities if len(e) > 0 and not e.isspace()]

    if not isinstance(entities, list):
        raise ValueError("❌ Extraction failed, result is not a list.")

    return entities
