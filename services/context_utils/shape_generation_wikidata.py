import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts.wikidata import shape_selection_prompt_per_entity, class_instances_prompt
from services.log_utils import log_message, log_warning
from SPARQLWrapper import SPARQLWrapper, JSON

load_dotenv(dotenv_path=".env")

_URI_PREFIXES_WDT = [
    ("http://www.wikidata.org/prop/direct/", "wdt:"),
    ("http://www.wikidata.org/prop/statement/", "ps:"),
    ("http://www.wikidata.org/prop/qualifier/", "pq:"),
    ("http://www.wikidata.org/prop/", "p:"),
    ("http://www.wikidata.org/entity/", "wd:"),
    ("http://www.w3.org/2001/XMLSchema#", "xsd:"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
    ("http://schema.org/", "schema:"),
]

_MAX_PROPS_WITHOUT_FILTER = 30


def _shorten_wdt_uri(uri: str) -> str:
    for full, prefix in _URI_PREFIXES_WDT:
        if uri.startswith(full):
            return prefix + uri[len(full):]
    return f"<{uri}>"


def _run_sparql_wikidata(query: str, endpoint: str, timeout: int = 30) -> list:
    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.timeout = timeout
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        return result.get("results", {}).get("bindings", [])
    except Exception as e:
        log_warning("shape_generation_wikidata", str(e))
        return []


def _run_sparql_for_entity_wikidata(qid: str, endpoint: str, log_calls: bool = False) -> list:
    """Return wdt: properties with inferred range for a specific Wikidata entity (QID)."""
    query = f"""
SELECT DISTINCT ?prop (SAMPLE(?val) AS ?sampleVal) WHERE {{
  wd:{qid} ?prop ?val .
  FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
}}
GROUP BY ?prop
LIMIT 150
"""
    bindings = _run_sparql_wikidata(query, endpoint)
    items = []
    for b in bindings:
        prop_uri = b.get("prop", {}).get("value", "")
        if not prop_uri:
            continue
        prop_short = _shorten_wdt_uri(prop_uri)
        sample = b.get("sampleVal", {})
        if sample.get("type") == "uri":
            range_short = "IRI"
        else:
            dt = sample.get("datatype", "")
            range_short = _shorten_wdt_uri(dt) if dt else "xsd:string"
        items.append(f"{prop_short} -> {range_short}")
    if log_calls:
        log_message("shape_gen_entity_wikidata", "Cyan", [qid, str(len(items))])
    return items


def _run_sparql_for_class_wikidata(qid: str, endpoint: str, log_calls: bool = False) -> list:
    """Return wdt: properties found on instances of a Wikidata class (via P31/P279*)."""
    query = f"""
SELECT DISTINCT ?prop (SAMPLE(?val) AS ?sampleVal) WHERE {{
  ?inst wdt:P31/wdt:P279* wd:{qid} .
  ?inst ?prop ?val .
  FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
}}
GROUP BY ?prop
LIMIT 100
"""
    bindings = _run_sparql_wikidata(query, endpoint, timeout=45)
    items = []
    for b in bindings:
        prop_uri = b.get("prop", {}).get("value", "")
        if not prop_uri:
            continue
        prop_short = _shorten_wdt_uri(prop_uri)
        sample = b.get("sampleVal", {})
        if sample.get("type") == "uri":
            range_short = "IRI"
        else:
            dt = sample.get("datatype", "")
            range_short = _shorten_wdt_uri(dt) if dt else "xsd:string"
        items.append(f"{prop_short} -> {range_short}")
    if log_calls:
        log_message("shape_gen_class_wikidata", "Cyan", [qid, str(len(items))])
    return items


def _llm_classify(label: str, llm) -> bool:
    """Returns True if label is a CLASS/type, False if it is a named ENTITY."""
    prompt = class_instances_prompt["en"].format(label=label)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().startswith("CLASS")


def _select_relevant_shape_parts(nlq: str, shape: str, llm, label: str) -> list:
    prompt = shape_selection_prompt_per_entity["en"].format(nlq=nlq, label=label, shape=shape)
    response = llm.invoke([HumanMessage(content=prompt)])
    return [i.strip() for i in response.content.strip().split(",") if i.strip()]


def _process_entity_wikidata(label: str, qid: str, nlq: str, shapes_llm, endpoint: str, log_calls: bool = False) -> str:
    is_class = _llm_classify(label, shapes_llm)
    if is_class:
        items = _run_sparql_for_class_wikidata(qid, endpoint, log_calls=log_calls)
    else:
        items = _run_sparql_for_entity_wikidata(qid, endpoint, log_calls=log_calls)

    if not items:
        return ""
    if shapes_llm and len(items) > _MAX_PROPS_WITHOUT_FILTER:
        items = _select_relevant_shape_parts(nlq, "\n".join(items), shapes_llm, label=label)
    if not items:
        return ""
    indented = "\n".join(f"  {line}" for line in items)
    return f"{label}:\n{indented}"


def generate_shape_wikidata(nlq: str, entity_uris: list, shapes_llm, log_calls: bool = False) -> str:
    """Generate a Wikidata property shape for the given linked entity URIs.

    entity_uris: [{label: "http://www.wikidata.org/entity/QID"}, ...]
    Returns a multi-section string with wdt: properties per entity.
    """
    load_dotenv(dotenv_path=".env")
    endpoint = os.getenv("WIKIDATA_SPARQL_URL", "https://query.wikidata.org/sparql")

    tasks = []
    for item in entity_uris:
        if not item:
            continue
        label = list(item.keys())[0]
        uri = list(item.values())[0]
        if not uri or "wikidata.org/entity/" not in uri:
            continue
        qid = uri.rstrip("/").split("/")[-1]
        if not qid.startswith("Q"):
            continue
        tasks.append((label, qid))

    if not tasks:
        return None

    try:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda t: _process_entity_wikidata(t[0], t[1], nlq, shapes_llm, endpoint, log_calls=log_calls),
                tasks
            ))
        sections = [s for s in results if s]
    except Exception as e:
        log_warning("generate_shape_wikidata", str(e))
        return None

    return "\n".join(sections) if sections else None
