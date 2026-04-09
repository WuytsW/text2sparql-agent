import os
import re as _re
import logging
from shexer.shaper import Shaper
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts.dbpedia import (
    shape_selection_prompt,
    shape_selection_prompt_per_entity,
    class_instances_prompt,
)
from SPARQLWrapper import SPARQLWrapper, JSON
from services.log_utils.log import log_message

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

# Parses a single shexer ShEx statement line: "   dbo:capital  IRI  ;"
_SHEX_STMT_RE = _re.compile(r"^\s{1,6}(\^?[\w:<>]+)\s+(@?[\w:<>\[\]]+)")

# Properties to skip when parsing ShEx output
_SKIP_PROPS = {"rdf:type"}


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


def _tbox_to_prop_range_items(properties: list) -> list:
    """Convert get_tbox_properties() dicts into plain prop -> range strings."""
    return [
        f"{_shorten_uri(p['prop'])} -> {_shorten_uri(p['range']) if p.get('range') else 'IRI'}"
        for p in properties
    ]


def _parse_shex_to_prop_range_items(shex_string: str) -> list:
    """
    Parse a shexer ShEx block into prop -> range strings compatible with
    select_relevant_shape_parts and add_possible_values_to_shape.
    Skips rdf:type lines and structural tokens (PREFIX, shape name, braces).
    """
    items = []
    for line in shex_string.splitlines():
        s = line.strip()
        if not s or s in ("{", "}") or s.startswith(("PREFIX", "shapes:", "<http")):
            continue
        m = _SHEX_STMT_RE.match(line)
        if not m:
            continue
        prop = m.group(1)
        range_ = m.group(2).strip("@[]")
        if prop in _SKIP_PROPS:
            continue
        items.append(f"{prop} -> {range_}")
    return items


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def get_tbox_properties(class_uri: str, sparql_endpoint: str) -> list:
    """
    Returns all properties applicable to a class (including inherited via rdfs:subClassOf*).
    Each result is a dict with 'prop', 'domain', and optionally 'range'.
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
        logging.warning(f"[get_tbox_properties] T-Box query failed for <{class_uri}>: {e}")
        return []

    properties = []
    for binding in result.get("results", {}).get("bindings", []):
        prop = binding.get("prop", {}).get("value", "")
        domain = binding.get("domain", {}).get("value", "")
        range_ = binding.get("range", {}).get("value", "")
        if prop:
            properties.append({"prop": prop, "domain": domain, "range": range_})

    logging.info(f"[get_tbox_properties] Found {len(properties)} properties for <{class_uri}>")
    return properties


def _query_property_values(prop_prefixed: str, sparql_endpoint: str) -> list:
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
        logging.warning(f"[_query_property_values] Failed for {prop_prefixed}: {e}")
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
    return values


def add_possible_values_to_shape(relevant_items: list, sparql_endpoint: str) -> str:
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
        prop, range_ = m.group(1), m.group(2)
        values = _query_property_values(prop, sparql_endpoint)
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

def _run_shexer_for_entity(label_clean: str, endpoint: str, namespaces_dict: dict) -> str:
    """
    Run shexer for a single named entity and return the raw ShEx string.
    Returns empty string on failure.
    """
    entity_id = f"http://dbpedia.org/resource/{label_clean}"
    shape_label = f"http://shapes.dbpedia.org/{label_clean}"
    shape_map_raw = f"<{entity_id}>@<{shape_label}>"
    try:
        shaper = Shaper(
            shape_map_raw=shape_map_raw,
            url_endpoint=endpoint,
            namespaces_dict=namespaces_dict,
            disable_comments=True,
        )
        return shaper.shex_graph(string_output=True) or ""
    except Exception as e:
        logging.warning(f"[_run_shexer_for_entity] shexer failed for {label_clean}: {e}")
        return ""


def _process_entity_section(
    label_clean: str,
    items: list,
    nlq: str,
    llm,
    endpoint: str,
    use_llm: bool,
) -> str:
    """
    Runs the filter -> values pipeline for one entity and returns a labeled
    multi-line section string, or empty string if no items survive.
    """
    if not items:
        return ""
    if llm and (use_llm or len(items) > _MAX_PROPS_WITHOUT_FILTER):
        items = select_relevant_shape_parts(nlq, "\n".join(items), llm, label=label_clean)
    if not items:
        return ""
    enriched = add_possible_values_to_shape(items, endpoint)
    if not enriched.strip():
        return ""
    indented = "\n".join(f"  {line}" for line in enriched.splitlines())
    return f"{label_clean}:\n{indented}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_NAMESPACES_DICT = {
    "http://example.org/": "ex",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://dbpedia.org/resource/": "dbr",
    "http://dbpedia.org/ontology/": "dbo",
    "http://dbpedia.org/property/": "dbp",
    "http://dbpedia.org/class/yago/": "yago",
    "http://purl.org/dc/terms/": "dcterms",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/2007/05/powder-s#": "powders",
    "http://www.w3.org/ns/prov#": "prov",
    "http://umbel.org/umbel/rc/": "umbel",
    "http://schema.org/": "schema",
    "http://shapes.dbpedia.org/": "shapes",
}


def generate_shape(nlq: str, entity_labels: list, shapes_llm, use_llm: bool = False):
    load_dotenv(dotenv_path=".env")
    endpoint = os.getenv("DBPEDIA_SPARQL_URL")
    logging.info(f"[generate_shape] Entity labels: {entity_labels}")

    sections = []
    try:
        for label in entity_labels:
            label_clean = label.replace(" ", "_")
            label_clean = label_clean[0].upper() + label_clean[1:]
            logging.info(f"[generate_shape] Processing '{label_clean}'")

            if _llm_classify(label_clean, shapes_llm):
                logging.info(f"[generate_shape] '{label_clean}' -> CLASS (T-Box path)")
                class_uri = f"http://dbpedia.org/ontology/{label_clean}"
                props = get_tbox_properties(class_uri, endpoint)
                items = _tbox_to_prop_range_items(props)
            else:
                logging.info(f"[generate_shape] '{label_clean}' -> ENTITY (shexer path)")
                shex_str = _run_shexer_for_entity(label_clean, endpoint, _NAMESPACES_DICT)
                items = _parse_shex_to_prop_range_items(shex_str)

            section = _process_entity_section(
                label_clean, items, nlq, shapes_llm, endpoint, use_llm
            )
            if section:
                sections.append(section)

    except Exception as e:
        logging.error(f"[generate_shape] Failed: {e}", exc_info=True)
        return None

    if not sections:
        logging.warning(f"[generate_shape] No sections produced for labels: {entity_labels}")
        return None

    result = "\n\n".join(sections)

    log_message("generate_shape", "Cyan", [f"NLQ: {nlq}", f"Entity labels: {entity_labels}", f"Generated shape:\n{result}"])

    return result
