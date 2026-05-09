import os
import logging
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from services.log_utils.log import log_message
from prompts.dbpedia import category_selection_prompt

load_dotenv(dotenv_path=".env")

_DBPEDIA_URL = os.getenv("DBPEDIA_SPARQL_URL", "https://dbpedia.org/sparql")
_CATEGORY_PREFIX = "http://dbpedia.org/resource/Category:"


def _fetch_categories_for_topic(topic: str, limit: int = 20) -> list[dict]:
    """Query DBpedia for dbc: category URIs matching a topic string."""
    search_term = topic.lower().replace(" ", "_")
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?cat ?label WHERE {{
  ?cat a skos:Concept ;
       rdfs:label ?label .
  FILTER(STRSTARTS(STR(?cat), "{_CATEGORY_PREFIX}"))
  FILTER(CONTAINS(LCASE(STR(?cat)), "{search_term}"))
}} LIMIT {limit}
"""
    try:
        sparql = SPARQLWrapper(_DBPEDIA_URL)
        sparql.setTimeout(30)
        sparql.setQuery(query)
        sparql.setReturnFormat(SPARQL_JSON)
        result = sparql.query().convert()
        return [
            {"uri": b["cat"]["value"], "label": b.get("label", {}).get("value", "")}
            for b in result.get("results", {}).get("bindings", [])
            if b.get("cat", {}).get("value")
        ]
    except Exception as e:
        logging.warning(f"[category_linking] Query failed for '{topic}': {e}")
        return []


def select_relevant_categories(nlq: str, categories: list[dict], llm) -> list[dict]:
    """Filter categories to those relevant to the NLQ using an LLM."""
    if not categories:
        return []
    cats_str = "\n".join(f"  {c['uri']}  ({c['label']})" for c in categories)
    prompt = category_selection_prompt["en"].format(nlq=nlq, categories=cats_str)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        selected_uris = {u.strip() for u in response.content.strip().split(",") if u.strip()}
        return [c for c in categories if c["uri"] in selected_uris]
    except Exception as e:
        logging.warning(f"[category_linking] LLM selection failed: {e}")
        return categories


def fetch_categories(nlq: str, entities: list[str], llm=None) -> list[dict]:
    """
    Fetch DBpedia categories for a question's extracted entities.
    Deduplicates by URI. Filters to relevant ones via LLM if llm is provided.
    Returns list of {"uri", "label"} dicts.
    """
    seen_uris: set[str] = set()
    results: list[dict] = []

    for entity in entities[:20]:
        for cat in _fetch_categories_for_topic(entity, limit=20):
            if cat["uri"] not in seen_uris:
                seen_uris.add(cat["uri"])
                results.append(cat)

    if llm and results:
        results = select_relevant_categories(nlq, results, llm)

    log_message(
        step_name="[context] DBpedia categories",
        color="Cyan",
        messages=[f"entities={entities[:20]}", f"found={len(results)} categories", results],
    )
    return results
