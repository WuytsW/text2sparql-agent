import json
import logging
import ast

from pydantic import BaseModel, Field
from langchain.tools import tool
from services.log_utils.log import log_message

from services.context_utils.entity_profile_generation_dbpedia import generate_entity_profile
from services.context_utils.category_linking import _fetch_categories_for_topic


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


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


class EntityProfileInput(BaseModel):
    nlq: str = Field(description="The user's natural language question")
    entity_labels: list[str] = Field(
        description="List of DBpedia class or entity labels to generate entity profiles for, e.g. ['Germany'] or ['Scientist']"
    )

class EntityExtractionInput(BaseModel):
    nlq: str = Field(description="The user's natural language question to extract entities from")


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
