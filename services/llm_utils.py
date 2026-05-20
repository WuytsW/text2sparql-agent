import os
import json
import logging
import requests
import ast
import random

from typing import List

from pydantic import BaseModel, Field
from langchain.tools import tool
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
from services.log_utils.log import log_message

from services.context_utils.entity_profile_generation_dbpedia import generate_entity_profile
from services.context_utils.category_linking import _fetch_categories_for_topic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent



logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

KNOWN_EAT_MAPPINGS = [
    ["http://www.w3.org/2001/XMLSchema#integer", "integer"],
    ["http://www.w3.org/2001/XMLSchema#boolean", "boolean"],
    ["http://www.w3.org/2001/XMLSchema#date", "date"],
    ["http://www.w3.org/2001/XMLSchema#dateTime", "dateTime"],
    ["http://www.w3.org/2001/XMLSchema#time", "time"],
    ["http://www.w3.org/2001/XMLSchema#string", "string", "literal"],
    ["http://www.w3.org/2001/XMLSchema#anyURI", "uri", "resource", "http://www.w3.org/2000/01/rdf-schema#Resource",
        "http://www.w3.org/2000/01/rdf-schema#List", "http://www.w3.org/2000/01/rdf-schema#Container", "http://www.w3.org/2000/01/rdf-schema#Collection", "http://www.w3.org/2000/01/rdf-schema#Bag", "http://www.w3.org/2000/01/rdf-schema#Set"],
    ["http://www.w3.org/2001/XMLSchema#double", "http://www.w3.org/2001/XMLSchema#decimal",
        "http://www.w3.org/2001/XMLSchema#float", "double", "decimal", "float"],
]

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
class NELInput(BaseModel):
    ne_list: list = Field(description="should be a list of named entities (strings) to be linked to the Wikidata URIs")

class DBpediaELInput(BaseModel):
    nlq: str = Field(description="The user's natural language question to link entities from")
    ne_list: list = Field(description="List of named entity strings extracted from the question")

class RELInput(BaseModel):
    rel_list: list = Field(description="should be a list of relations (strings) to be linked to the Knowledge Graph  URIs")

class EntityProfileInput(BaseModel):
    nlq: str = Field(description="The user's natural language question")
    entity_labels: list[str] = Field(
        description="List of DBpedia class or entity labels to generate entity profiles for, e.g. ['Germany'] or ['Scientist']"
    )

class EntityExtractionInput(BaseModel):
    nlq: str = Field(description="The user's natural language question to extract entities from")
    
@tool("wikidata_el", args_schema=NELInput)
def wikidata_el(ne_list: list) -> list:
    """Performs entity linking to Wikidata based on the provided list of named entity strings. Returns list of dict with linking candidates: [{"label": "URI"}]"""
    nel_list = []
    N = 5
    for ne in ne_list[:N]:
        entities, relations = [[], []] # search_entity(query=ne)
        falcon_relations, falcon_entities = [[], []] # falcon_rel(query=ne)
        relations += falcon_relations
        
        nel_list += entities
        nel_list += falcon_entities
        nel_list += relations

    log_message(step_name="Entity linking candidates from Wikidata", color="Yellow", messages=[str(nel_list)])
    return nel_list

def make_extract_entities_tool(llm):
    """Factory that returns an extract_entities_tool bound to the given LLM."""
    from services.context_utils.entity_extraction import extract_entities

    @tool("extract_entities_tool", args_schema=EntityExtractionInput)
    def extract_entities_tool(nlq: str) -> list[str]:
        """
        Extract named entities and classes from a natural language question.
        Returns a list of entity/class label strings (e.g. ['Germany', 'Scientist']).
        Call this before generate_entity_profile_tool to determine which entities to generate profiles for.
        """
        return extract_entities(nlq, llm)

    return extract_entities_tool


class DBpediaCategoriesInput(BaseModel):
    topic: str = Field(description="A topic or entity label to search for matching DBpedia category URIs (e.g. 'James Bond films', 'Countries in Africa')")


@tool("dbpedia_categories_tool", args_schema=DBpediaCategoriesInput)
def dbpedia_categories_tool(topic: str) -> list:
    """
    Searches DBpedia for Wikipedia category URIs (dbc:) that match a given topic.
    Use this when a question is about group membership and structured ontology triples are insufficient.
    Returns a list of matching category URIs that can be used as: ?uri dct:subject <category_uri>
    """
    return _fetch_categories_for_topic(topic)


def make_generate_entity_profile_tool(llm):
    """Factory that returns a generate_entity_profile_tool bound to the given LLM."""

    @tool("generate_entity_profile_tool", args_schema=EntityProfileInput)
    def generate_entity_profile_tool(nlq: str, entity_labels: list[str]) -> str:
        """
        Generate a DBpedia-oriented entity profile for the given entity/class labels.
        Returns a text block with relevant properties and, when possible, controlled values.
        """
        result = generate_entity_profile(
            nlq=nlq,
            entity_labels=entity_labels,
            profile_llm=llm,
        )

        return result or "No entity profile could be generated."

    return generate_entity_profile_tool


class EntityLinkingInput(BaseModel):
    nlq: str = Field(description="The natural language question")
    ne_list: list = Field(description="List of named entity strings to link to DBpedia URIs")


def make_entity_linking_tool():
    """Factory that returns an entity_linking_tool wrapping dbpedia_el."""
    from services.context_utils.entity_linking_dbpedia import dbpedia_el

    @tool("entity_linking_tool", args_schema=EntityLinkingInput)
    def entity_linking_tool(nlq: str, ne_list: list) -> str:
        """
        Links named entities to DBpedia URIs using the Falcon entity linking service.
        Call this after extract_entities_tool to get DBpedia resource URIs.
        Returns a JSON string of linking candidates: [{"label": "...", "uri": "..."}, ...]
        """
        result = dbpedia_el(nlq, ne_list)
        log_message(step_name="Entity linking (tool)", color="Cyan", messages=[str(result)])
        return json.dumps(result)

    return entity_linking_tool


class ExecuteSPARQLInput(BaseModel):
    query: str = Field(description="The SPARQL query string to execute against DBpedia")


def make_execute_sparql_tool(endpoint: str):
    """Factory that returns an execute_sparql_tool bound to the given SPARQL endpoint."""
    from services.ld_utils import execute

    @tool("execute_sparql_tool", args_schema=ExecuteSPARQLInput)
    def execute_sparql_tool(query: str) -> str:
        """
        Executes a SPARQL query against the DBpedia endpoint.
        Returns the first 5 result bindings as a JSON string, or an error message.
        Use this to verify that your generated SPARQL query returns results.
        """
        try:
            result = execute(query=query, endpoint_url=endpoint)
            if isinstance(result, dict) and "error" not in result:
                bindings = result.get("results", {}).get("bindings", [])[:5]
                return json.dumps(bindings)
            return json.dumps(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return execute_sparql_tool


class ContextCheckInput(BaseModel):
    nlq: str = Field(description="The natural language question")
    entities: list = Field(description="Extracted entity/class labels")
    entity_uris: list = Field(description="Linked DBpedia URIs for the entities")
    categories: list = Field(description="DBpedia category URIs and labels")
    entity_profile: str = Field(description="The DBpedia entity profile text (available properties)")


def make_context_check_tool(llm, context_prompt=None):
    """Factory that returns a context_check_tool bound to the given LLM."""
    if context_prompt is None:
        from prompts.dbpedia import context_check_prompt
        context_prompt = context_check_prompt

    @tool("context_check_tool", args_schema=ContextCheckInput)
    def context_check_tool(nlq: str, entities: list, entity_uris: list, categories: list, entity_profile: str) -> dict:
        """
        Evaluates whether the full extracted context (entities, URIs, categories, entity profile) is useful for answering the question.
        Returns a dict with 'valid' (bool) and 'reason' (str).
        """
        import json as _json
        prompt = context_prompt["en"].format(
            nlq=nlq,
            entities=entities or [],
            entity_uris=entity_uris or [],
            categories=categories or [],
            entity_profile=entity_profile or "(empty)",
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        raw = response.content.strip()
        try:
            result = _json.loads(raw)
            return {"valid": bool(result.get("valid", False)), "reason": result.get("reason", "")}
        except Exception:
            valid = "true" in raw.lower() and "false" not in raw.lower()
            return {"valid": valid, "reason": raw}

    return context_check_tool


_PREFIX_CORRECTIONS = {
    "dbp:homepage":  "foaf:homepage",
    "dbp:nick":      "foaf:nick",
    "dbp:name":      "foaf:name",
    "dbp:depiction": "foaf:depiction",
    "dbp:mbox":      "foaf:mbox",
    "dbp:abstract":  "dbo:abstract",
    "dbp:thumbnail": "dbo:thumbnail",
}

def correct_query_prefixes(query: str, llm=None) -> str:
    """Correct dbp: properties to semantic equivalents (foaf:, dbo:) using a fixed rule table."""
    for src, dst in _PREFIX_CORRECTIONS.items():
        query = query.replace(src, dst)
    return query


def get_corporate_entities(query: str, is_relation: bool) -> list:
    """
    Make a GET request to the Corporate entity service and return the parsed response
    """
    try:
        base_url = os.environ.get("CORPORATE_SERVICE_BASE_URL", "http://141.57.8.18:9199")
        
        if is_relation:

            url = f"{base_url}/corporate/relations/?query={query}"
        else:
            url = f"{base_url}/corporate/entities/?query={query}"

        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()[:3]
        else:
            logging.error(f"Error fetching entities: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Exception in get_corporate_entities: {str(e)}")
        return []

@tool("corporate_el", args_schema=NELInput)
def el_corporate(ne_list: list) -> list:
    """Performs entity linking to Corporate based on the provided list of named entity strings. Returns list of dict with linking candidates: [{"label": "URI"}]"""
    nel_list = []
    N = 5
    for ne in ne_list[:N]:
        entities = get_corporate_entities(ne, False)
        for entity in entities:
            nel_list.append({
                "label": entity.get("label", ""),
                "uri": entity.get("uri", ""),
                "score": entity.get("score", 0),
                "extra_score": entity.get("extra_score", 0)
            })

    return nel_list

@tool("corporate_rel", args_schema=RELInput)
def rel_corporate(rel_list: list) -> list:
    """Performs relation linking to Corporate KG based on the provided list of relations strings. Returns list of dict with linking candidates: [{"label": "URI"}]"""
    nel_list = []
    N = 5
    for rel in rel_list[:N]:
        relations = get_corporate_entities(rel, True)
        for relation in relations:
            nel_list.append({
                "label": relation.get("label", ""),
                "uri": relation.get("uri", ""),
                "score": relation.get("score", 0),
                "extra_score": relation.get("extra_score", 0)
            })

    return nel_list

def nel(ne_list: str) -> list:
    """Performs entity linking to Wikidata based on the provided list of named entity strings. Returns list of dict with linking candidates: [{"label": "URI"}]"""
    nel_list = []
    N = 5
    for ne in ne_list[:N]:
        entities, relations = [[], []] # search_entity(query=ne)
        falcon_relations, falcon_entities = [[], []] # falcon_rel(query=ne)
        relations += falcon_relations
        
        nel_list += entities
        nel_list += falcon_entities
        nel_list += relations

    return nel_list

def find_first_correct_item(results, json_data):
    for r in results:
        idx = r[0].metadata['seq_num']
        
        if json_data[idx-1]['precision'] == 1 and json_data[idx-1]['recall'] == 1:
            return idx - 1
        # TODO: check score
        
    return None

def find_random_item(results, json_data):    
    return random.randint(0, len(json_data) - 1)

def find_first_incorrect_item(results, json_data):
    for r in results:
        idx = r[0].metadata['seq_num']
        
        if json_data[idx-1]['precision'] == 0 and json_data[idx-1]['recall'] == 0:
            return idx - 1
        # TODO: check score
        
    return None

def find_random_correct_item(results, json_data):
    idx_list = []
    for r in results:
        idx = r[0].metadata['seq_num']
        
        if json_data[idx-1]['precision'] == 1 and json_data[idx-1]['recall'] == 1:
            idx_list.append(idx - 1)
        # TODO: check score

    if len(idx_list) > 0:
        return random.choice(idx_list)
        
    return None

def find_random_incorrect_item(results, json_data):
    idx_list = []
    for r in results:
        idx = r[0].metadata['seq_num']
        
        if json_data[idx-1]['precision'] == 0 and json_data[idx-1]['recall'] == 0:
            idx_list.append(idx - 1)
        # TODO: check score

    if len(idx_list) > 0:
        return random.choice(idx_list)
        
    return None

def construct_shot(idx, json_data):
    shot = "" # f"Input question: {json_data[idx]['past_steps'][0]}"
    step_num = 1
    for step in json_data[idx]['past_steps']:
        if type(step) == str:
            step = step.replace("\n", "")
            shot += f"Step {step_num}: {step}\n"
            step_num += 1
        elif type(step) == dict:
            log = step['log'].replace("\n", "")
            shot += f"Action: {log}\n"
        elif type(step) == list and len(step) == 0:
            shot += f"Action: Call plain LLM\n"
        else:
            pass
        
    return shot

def eat_json_answer_template(question, eat, confidence):
    return {
        "question": question,
        "expected_answer_type": {
            "eat": eat,
            "confidence": confidence
        }
    }

expected_answer_type_questions_and_expected_answer_types = [
    {"question": "Show me the birthday of Friedrich Schiller", "expected_answer_type": eat_json_answer_template(
        "Show me the birthday of Friedrich Schiller", "http://www.w3.org/2001/XMLSchema#date", 0.95)},
    {"question": "What is the capital of Germany?", "expected_answer_type": eat_json_answer_template(
        "What is the capital of Germany?", "http://www.w3.org/2000/01/rdf-schema#Resource", 0.95)},
    {"question": "What is the population of Berlin?", "expected_answer_type": eat_json_answer_template(
        "What is the population of Berlin?", "http://www.w3.org/2001/XMLSchema#integer", 0.95)},
    {"question": "What is the speed of light?", "expected_answer_type": eat_json_answer_template(
        "What is the speed of light?", "http://www.w3.org/2001/XMLSchema#decimal", 0.95)},
    {"question": "Is the capital of Germany Berlin?", "expected_answer_type": eat_json_answer_template(
        "Is the capital of Germany Berlin?", "http://www.w3.org/2001/XMLSchema#boolean", 0.95)},
]

def get_expected_answer_type(text, llm):
    """
    Perform Expected Answer Type (EAT) Analysis on the given text using a language model.
    Args:
        text (str): The input text to analyze for expected answer type.
    Returns:
        resource:datatype. If the response cannot be parsed as JSON, an empty list is returned.
    Example:
        >>> get_expected_answer_type("Show me the birthday of Friedrich Schiller")
        ["xsd:date"]
    Note:
        This function uses a language model to perform a EAT analysis and expects the model to return the recognized 
        expected answer type in a structured JSON format. If the response is not valid JSON, an error is logged and an empty list is returned.
    """

    # example_string = "Show me the birthday of Friedrich Schiller"
    # assistant_docstring = """["xsd:date"]"""

    logging.info(
        "get_expected_answer_type: Calling OpenAI API for question: '%s'", text)

    if text is None or text == "":
        logging.error("get_expected_answer_type: Text is None or empty")
        raise ValueError("Text is None or empty")

    messages = [
        (
            "system",
            """You are a Expected Answer Type Tool.
    Recognize named the expected answer type of the given question and output as RDF datatype and your confidence score.
    **Output ONLY the structured data.**
    Below is a text for you to analyze."""
        ),
        (
            "human", 
            expected_answer_type_questions_and_expected_answer_types[0]["question"]
        ),
        (
            "assistant",
            f"{expected_answer_type_questions_and_expected_answer_types[0]['expected_answer_type']}"
        ),
        (
            "human",
            expected_answer_type_questions_and_expected_answer_types[1]["question"]
        ),
        (
            "assistant",
            f"{expected_answer_type_questions_and_expected_answer_types[1]['expected_answer_type']}"
        ),
        (
            "human", 
            expected_answer_type_questions_and_expected_answer_types[2]["question"]
        ),
        (
            "assistant",
            f"{expected_answer_type_questions_and_expected_answer_types[2]['expected_answer_type']}"
        ),
        (
            "human", 
            text
        )
    ]

    result_text = llm.invoke(messages).content

    # parse the result
    try:
        # load JSON data from result_text using ' as the quote character
        result = ast.literal_eval(result_text)
        logging.info("LLM EAT result for question '%s': %s", text, result)
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: %s", result_text)
        return ["None"]

    eat = result.get("expected_answer_type", {}).get("eat", None)
    confidence = result.get("expected_answer_type", {}).get("confidence", None)

    if eat is None:
        logging.error("LLM EAT result contains invalid eat: %s", result)
        raise ValueError("LLM EAT result contains invalid eat: %s", result)
    if confidence is None:
        logging.error("LLM EAT result contains invalid confidence: %s", result)
        raise ValueError(
            "LLM EAT result contains invalid confidence: %s", result)

    return result
