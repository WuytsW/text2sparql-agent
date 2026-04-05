import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from services.LogLLMCallbackHandler import LogLLMCallbackHandler
from prompts.dbpedia import entities_extraction_prompt

load_dotenv(dotenv_path=".env")


def extract_entities_with_llm(nlq, llm):
    """
    Uses an LLM to extract the most relevant entities from a natural language query.
    Cleans and parses the extracted entity names into a clean Python list of strings.
    """

    user_prompt = entities_extraction_prompt["en"].format(nlq=nlq)

    

    response = llm.invoke([
        SystemMessage(content="You are an expert in extracting named entities from questions."),
        HumanMessage(content=user_prompt)
    ])

    raw_response = response.content.strip()
    # print(f"✅ Question: {nlq}")
    # print(f"✅ LLM response:\n{raw_response}")

    entities = re.findall(r'"([^"]+)"', raw_response)

    if entities:
        flat = []
        for e in entities:
            flat.extend([part.strip() for part in e.split(",") if part.strip()])
        entities = flat
    else:
        entities = [e.strip() for e in raw_response.split(",") if e.strip()]

    entities = [e for e in entities if len(e) > 0 and not e.isspace()]

    if not isinstance(entities, list):
        raise ValueError("❌ Extraction failed, result is not a list.")

    # print(f"✅ Extracted entities: {entities}")
    return entities


def extract_entities(question, llm):
    """
    Extracts entities from the given question using an LLM and resolves them against a SPARQL endpoint.
    """

    # print(f"[INFO] Extracting entities from question: {question}")
    dbpedia_sparql_url = os.getenv("DBPEDIA_SPARQL_URL")


    # print(f"[DEBUG] dbpedia_sparql_url: {dbpedia_sparql_url}")
    return extract_entities_with_llm(question, llm)
