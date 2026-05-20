import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from services.log_utils.LogLLMCallbackHandler import LogLLMCallbackHandler
from prompts.dbpedia import entities_extraction_prompt
from services.log_utils.log import log_message

load_dotenv(dotenv_path=".env")

def extract_entities(question, llm, failed_attempts: list = None):
    """
    Extracts entities from the given question using an LLM.
    failed_attempts: list of dicts [{entities, entity_profile, reason}] from prior entity_profile-check failures.
    """
    user_prompt = entities_extraction_prompt["en"].format(nlq=question)

    if failed_attempts:
        retry_lines = "\n".join(
            f"- Entities tried: {', '.join(a.get('entities', []))} — Reason failed: {a.get('reason', '')}"
            for a in failed_attempts
        )
        user_prompt += (
            f"\n\nPreviously tried entity labels that produced unhelpful entity profiles — "
            f"avoid these and try alternative labels:\n{retry_lines}"
        )

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
