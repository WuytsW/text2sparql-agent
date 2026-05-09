import os
import re as _re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts.dbpedia import (
    shape_selection_prompt,
    shape_selection_prompt_per_entity,
    class_instances_prompt,
)
from services.log_utils import log_message, log_warning
from SPARQLWrapper import SPARQLWrapper, JSON

load_dotenv(dotenv_path=".env")

# ---------------------------------------------------------------------------
# URI prefix table (used for shortening and expanding URIs)
# ---------------------------------------------------------------------------
_URI_PREFIXES = [
    ("http://dbpedia.org/ontology/", "dbo:"),
    ("http://dbpedia.org/property/", "dbp:"),
    ("http://dbpedia.org/resource/", "dbr:"),
    ("http://www.w3.org/2001/XMLSchema#", "xsd:"),
    ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
    ("http://xmlns.com/foaf/0.1/", "foaf:"),
    ("http://schema.org/", "schema:"),
]

# Max distinct values to enumerate; beyond this the property is not a controlled vocabulary
_MAX_ENUM_VALUES = 30
_MAX_PROPS_WITHOUT_FILTER = 30

# Matches "prop -> range" lines (with optional leading whitespace)
_PROP_RANGE_RE = _re.compile(r"^\s*(\S+)\s*->\s*(\S+)\s*$")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _shorten_uri(uri: str) -> str:
    for full, prefix in _URI_PREFIXES:
        if uri.startswith(full):
            return prefix + uri[len(full):]
    return f"<{uri}>"


def _expand_prefixed(prefixed: str):
    """Expand a prefixed name like dbo:conservationStatus to its full URI."""
    for full, prefix in _URI_PREFIXES:
        if prefixed.startswith(prefix):
            return full + prefixed[len(prefix):]
    return None


def _normalize_range(r: str) -> str:
    """Normalize a range string to a consistent prefixed form."""
    r = r.strip().strip("@[]")
    if not r:
        return "IRI"
    if r.startswith("<") and r.endswith(">"):
        return _shorten_uri(r[1:-1])
    if r.startswith("http://") or r.startswith("https://"):
        return _shorten_uri(r)
    return r


def _tbox_to_prop_range_items(properties: list) -> list:
    """Convert get_tbox_properties() dicts into plain prop -> range strings."""
    return [
        f"{_shorten_uri(p['prop'])} -> {_normalize_range(p.get('range', ''))}"
        for p in properties
    ]


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def get_tbox_properties(class_uri: str, sparql_endpoint: str, log_calls: bool = False) -> list:
    """
    Returns all properties applicable to a class (including inherited via rdfs:subClassOf*).
    Each result is a shortened 'prop -> range' string.
    """
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?prop ?domain ?range WHERE {{
  <{class_uri}> rdfs:subClassOf* ?domain .
  ?prop rdfs:domain ?domain .
  OPTIONAL {{ ?prop rdfs:range ?range }}
}}
"""
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.timeout = 30
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        if log_calls:
            log_warning("get_tbox_properties", str(e))
        return []

    properties = []
    for binding in result.get("results", {}).get("bindings", []):
        prop = binding.get("prop", {}).get("value", "")
        domain = binding.get("domain", {}).get("value", "")
        range_ = binding.get("range", {}).get("value", "")
        if prop:
            properties.append({"prop": prop, "domain": domain, "range": range_})

    items = _tbox_to_prop_range_items(properties)
    if log_calls:
        log_message("get_tbox_properties", "Cyan", [
            _shorten_uri(class_uri),
            str(len(items)),
            str(items),
        ])
    return items


def get_abox_dbp_properties(class_uri: str, sparql_endpoint: str, sample_size: int = 3, log_calls: bool = False) -> list:
    """
    Returns dbp: (raw Wikipedia infobox) properties found on sample instances of a class.
    These are NOT in the T-Box but frequently hold the actual data values in DBpedia.
    Each result is a shortened 'prop -> IRI' string (range unknown from A-box).
    """
    dbp_ns = "http://dbpedia.org/property/"
    query = f"""
SELECT DISTINCT ?prop WHERE {{
  ?instance a <{class_uri}> .
  ?instance ?prop ?val .
  FILTER(STRSTARTS(STR(?prop), "{dbp_ns}"))
}} LIMIT 60
"""
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.timeout = 20
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        if log_calls:
            log_warning("get_abox_dbp_properties", str(e))
        return []

    properties = []
    seen = set()
    for binding in result.get("results", {}).get("bindings", []):
        prop = binding.get("prop", {}).get("value", "")
        if prop and prop not in seen:
            seen.add(prop)
            properties.append({"prop": prop, "domain": "", "range": ""})

    items = _tbox_to_prop_range_items(properties)
    if log_calls:
        log_message("get_abox_dbp_properties", "Cyan", [
            _shorten_uri(class_uri),
            str(len(items)),
            str(items),
        ])
    return items


def _query_property_values(prop_prefixed: str, sparql_endpoint: str, log_calls: bool = False) -> list:
    """
    Returns distinct values for prop_prefixed if the property has at most
    _MAX_ENUM_VALUES distinct values (controlled vocabulary check via LIMIT trick).
    Handles both IRI objects (dbr:Extinct) and string literals ("EX").
    """
    prop_uri = _expand_prefixed(prop_prefixed)
    if not prop_uri:
        return []
    query = f"SELECT DISTINCT ?val WHERE {{ ?s <{prop_uri}> ?val . }} LIMIT {_MAX_ENUM_VALUES + 1}"
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.timeout = 15
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        if log_calls:
            log_warning("_query_property_values", str(e))
        return []
    bindings = result.get("results", {}).get("bindings", [])
    if len(bindings) > _MAX_ENUM_VALUES:
        return []
    values = []
    for b in bindings:
        val_data = b.get("val", {})
        val = val_data.get("value", "")
        if not val:
            continue
        if val_data.get("type") == "uri":
            values.append(_shorten_uri(val))
        else:
            values.append('"' + val + '"')
    if log_calls:
        log_message("_query_property_values", "Cyan", [
            prop_prefixed,
            str(len(values)),
            str(values),
        ])
    return values


def add_possible_values_to_shape(relevant_items: list, sparql_endpoint: str, log_calls: bool = False) -> str:
    """
    For each prop -> range item, queries the A-Box for distinct values using
    cardinality filtering. Properties with few distinct values (controlled
    vocabulary) get their values listed; others (e.g. scientificName) are left as-is.
    """
    result_lines = []
    for item in relevant_items:
        item = item.strip()
        if not item:
            continue
        m = _PROP_RANGE_RE.match(item)
        if not m:
            result_lines.append(item)
            continue
        prop, range_ = m.group(1), _normalize_range(m.group(2))
        values = _query_property_values(prop, sparql_endpoint, log_calls=log_calls)
        if values:
            vals_str = ", ".join(values)
            result_lines.append(f"{prop} -> {range_} [values: {vals_str}]")
        else:
            result_lines.append(f"{prop} -> {range_}")
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_classify(label, llm):
    """Returns True if label is a CLASS/type, False if it is a named ENTITY."""
    prompt = class_instances_prompt["en"].format(label=label)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().startswith("CLASS")


def select_relevant_shape_parts(nlq: str, shape: str, llm, label: str = None) -> list:
    """
    Uses an LLM to filter a newline-joined list of prop -> range items down to
    the ones most relevant for answering nlq. When label is provided, uses the
    per-entity prompt variant for better accuracy.
    Returns a list of stripped prop -> range strings.
    """
    if label:
        prompt = shape_selection_prompt_per_entity["en"].format(
            nlq=nlq, label=label, shape=shape
        )
    else:
        prompt = shape_selection_prompt["en"].format(nlq=nlq, shape=shape)
    response = llm.invoke([HumanMessage(content=prompt)])
    return [i.strip() for i in response.content.strip().split(",") if i.strip()]


# ---------------------------------------------------------------------------
# Per-entity pipeline
# ---------------------------------------------------------------------------

def _run_sparql_for_entity(label_clean: str, endpoint: str, log_calls: bool = False) -> list:
    """
    Query DBpedia A-Box for all properties of a named entity and return
    prop -> range items directly.
    """
    entity_uri = f"http://dbpedia.org/resource/{label_clean}"
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?p (SAMPLE(?declaredRange) AS ?range) (SAMPLE(?o) AS ?sampleO)
WHERE {{
  <{entity_uri}> ?p ?o .
  OPTIONAL {{ ?p rdfs:range ?declaredRange }}
  FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
}}
GROUP BY ?p
LIMIT 200
"""
    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.timeout = 30
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        if log_calls:
            log_warning("_run_sparql_for_entity", str(e))
        return []

    items = []
    for binding in result.get("results", {}).get("bindings", []):
        prop_uri = binding.get("p", {}).get("value", "")
        if not prop_uri:
            continue
        prop_short = _shorten_uri(prop_uri)

        range_data = binding.get("range", {})
        sample_o = binding.get("sampleO", {})

        if range_data.get("value"):
            range_short = _shorten_uri(range_data["value"])
        elif sample_o.get("type") == "uri":
            range_short = "IRI"
        else:
            dt = sample_o.get("datatype", "")
            range_short = _shorten_uri(dt) if dt else "xsd:string"

        items.append(f"{prop_short} -> {range_short}")

    if log_calls:
        log_message("_run_sparql_for_entity", "Cyan", [
            label_clean,
            str(len(items)),
            str(items),
        ])
    return items


def _process_entity_section(
    label_clean: str,
    items: list,
    nlq: str,
    llm,
    endpoint: str,
    log_calls: bool = False,
) -> str:
    """
    Runs the filter -> values pipeline for one entity and returns a labeled
    multi-line section string, or empty string if no items survive.
    """
    if not items:
        return ""
    if llm and len(items) > _MAX_PROPS_WITHOUT_FILTER:
        items = select_relevant_shape_parts(nlq, "\n".join(items), llm, label=label_clean)
    if not items:
        return ""
    enriched = add_possible_values_to_shape(items, endpoint, log_calls=log_calls)
    if not enriched.strip():
        return ""
    indented = "\n".join(f"  {line}" for line in enriched.splitlines())
    return f"{label_clean}:\n{indented}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _process_label(label: str, nlq: str, shapes_llm, endpoint: str, log_calls: bool = False) -> str:
    label_clean = label.replace(" ", "_")
    label_clean = label_clean[0].upper() + label_clean[1:]

    if _llm_classify(label_clean, shapes_llm):
        class_uri = f"http://dbpedia.org/ontology/{label_clean}"
        items = get_tbox_properties(class_uri, endpoint, log_calls=log_calls)
        # Also fetch raw Wikipedia infobox (dbp:) properties from A-box sample instances.
        # These are absent from the T-Box but often hold the actual data values.
        dbp_items = get_abox_dbp_properties(class_uri, endpoint, log_calls=log_calls)
        # Merge, avoiding duplicates (dbo: items take precedence).
        existing_props = {item.split(" ->")[0].strip() for item in items}
        for dbp_item in dbp_items:
            dbp_key = dbp_item.split(" ->")[0].strip()
            if dbp_key not in existing_props:
                items.append(dbp_item)
                existing_props.add(dbp_key)
    else:
        items = _run_sparql_for_entity(label_clean, endpoint, log_calls=log_calls)

    return _process_entity_section(label_clean, items, nlq, shapes_llm, endpoint, log_calls=log_calls)


def generate_shape(nlq: str, entity_labels: list, shapes_llm, log_calls: bool = False) -> str:
    load_dotenv(dotenv_path=".env")
    endpoint = os.getenv("DBPEDIA_SPARQL_URL")

    try:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda label: _process_label(label, nlq, shapes_llm, endpoint, log_calls=log_calls),
                entity_labels
            ))
        sections = [s for s in results if s]

    except Exception as e:
        return None

    if not sections:
        return None

    return "\n".join(sections)
