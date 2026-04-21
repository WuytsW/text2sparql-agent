import os
import re as _re
import logging
from dataclasses import dataclass, field
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from shexer.shaper import Shaper
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts.dbpedia import (
    shape_selection_prompt,
    shape_selection_prompt_per_entity,
    class_instances_prompt,
)
from SPARQLWrapper import SPARQLWrapper, JSON

load_dotenv(dotenv_path=".env")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MAX_ENUM_VALUES = 30
_MAX_PROPS_WITHOUT_FILTER = 30

# Matches "prop -> range" lines (with optional leading whitespace)
_PROP_RANGE_RE = _re.compile(r"^\s*(\S+)\s*->\s*(\S+)\s*$")

# Parses a single shexer ShEx statement line: "   dbo:capital  IRI  ;"
_SHEX_STMT_RE = _re.compile(r"^\s{1,6}(\^?[\w:<>]+)\s+(@?[\w:<>\[\]]+)")

# Properties to skip when parsing ShEx output
_SKIP_PROPS = {"rdf:type"}


# ---------------------------------------------------------------------------
# KGConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KGConfig:
    """
    Configuration bundle for a SPARQL Knowledge Graph.

    Parameters
    ----------
    sparql_endpoint:
        The SPARQL endpoint URL for the KG.
    uri_prefixes:
        Ordered list of (full_uri, "prefix:") pairs used for URI shortening
        and expansion. Place longer/more-specific namespaces first so the
        longest-match wins (e.g. dbp: before db:).
    namespaces_dict:
        Namespace dict passed to shexer. Keys are full URIs, values are the
        short prefix *without* a colon (e.g. {"http://dbpedia.org/ontology/": "dbo"}).
    class_namespace:
        Base URI prepended to a label string when constructing a class URI
        (e.g. "http://dbpedia.org/ontology/" → "http://dbpedia.org/ontology/City").
    resource_namespace:
        Base URI prepended to a label string when constructing an entity URI
        (e.g. "http://dbpedia.org/resource/" → "http://dbpedia.org/resource/Paris").
    shape_namespace:
        Base URI prepended to a label string when constructing the shexer shape
        URI (e.g. "http://shapes.dbpedia.org/Paris").
    abox_extra_namespaces:
        Optional list of property namespace URIs to scan on A-Box instances.
        For each class, any property whose URI starts with one of these strings
        is included (analogous to dbp: properties in DBpedia). Defaults to [].
    label_transform:
        Optional callable applied to a raw label string to produce the
        KG-canonical form before URI construction. Use this for conventions
        like DBpedia's "capitalize + replace spaces with underscores". When
        None, the label is used as-is. Not applied to labels that are already
        full URIs (starting with "http").
    """
    sparql_endpoint: str
    uri_prefixes: list
    namespaces_dict: dict
    class_namespace: str
    resource_namespace: str
    shape_namespace: str
    abox_extra_namespaces: list = field(default_factory=list)
    label_transform: Callable = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _shorten_uri(uri: str, kg_config: KGConfig) -> str:
    for full, prefix in kg_config.uri_prefixes:
        if uri.startswith(full):
            return prefix + uri[len(full):]
    return f"<{uri}>"


def _expand_prefixed(prefixed: str, kg_config: KGConfig):
    """Expand a prefixed name like dbo:conservationStatus to its full URI."""
    for full, prefix in kg_config.uri_prefixes:
        if prefixed.startswith(prefix):
            return full + prefixed[len(prefix):]
    return None


def _tbox_to_prop_range_items(properties: list, kg_config: KGConfig) -> list:
    """Convert get_tbox_properties() dicts into plain prop -> range strings."""
    return [
        f"{_shorten_uri(p['prop'], kg_config)} -> {_shorten_uri(p['range'], kg_config) if p.get('range') else 'IRI'}"
        for p in properties
    ]


def _parse_shex_to_prop_range_items(shex_string: str) -> list:
    """
    Parse a shexer ShEx block into prop -> range strings compatible with
    select_relevant_shape_parts and add_possible_values_to_shape.
    Skips rdf:type lines and structural tokens (PREFIX declarations, shape
    name lines, braces).
    """
    items = []
    for line in shex_string.splitlines():
        s = line.strip()
        if not s or s in ("{", "}") or s.startswith(("PREFIX", "<http")):
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
    Works on any RDFS-compliant KG.
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

    return properties


def get_abox_extra_properties(
    class_uri: str,
    sparql_endpoint: str,
    abox_extra_namespaces: list,
) -> list:
    """
    Returns properties found on A-Box instances of a class whose URIs start
    with any of the given namespace strings.  Returns [] immediately when
    abox_extra_namespaces is empty (no SPARQL call is made).

    This generalises get_abox_dbp_properties from the DBpedia-specific module:
    pass ["http://dbpedia.org/property/"] to replicate the original behaviour.
    """
    if not abox_extra_namespaces:
        return []

    ns_filters = " || ".join(
        f'STRSTARTS(STR(?prop), "{ns}")'
        for ns in abox_extra_namespaces
    )
    query = f"""
SELECT DISTINCT ?prop WHERE {{
  ?instance a <{class_uri}> .
  ?instance ?prop ?val .
  FILTER({ns_filters})
}} LIMIT 60
"""
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.timeout = 20
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        logging.warning(f"[get_abox_extra_properties] A-Box query failed for <{class_uri}>: {e}")
        return []

    seen = set()
    properties = []
    for binding in result.get("results", {}).get("bindings", []):
        prop = binding.get("prop", {}).get("value", "")
        if prop and prop not in seen:
            seen.add(prop)
            properties.append({"prop": prop, "domain": "", "range": ""})

    return properties


def _query_property_values(
    prop_prefixed: str,
    sparql_endpoint: str,
    kg_config: KGConfig,
) -> list:
    """
    Returns distinct values for prop_prefixed if the property has at most
    _MAX_ENUM_VALUES distinct values (controlled vocabulary check via LIMIT trick).
    Handles both IRI objects and string literals.
    """
    prop_uri = _expand_prefixed(prop_prefixed, kg_config)
    if not prop_uri:
        return []
    query = f"SELECT DISTINCT ?val WHERE {{ ?s <{prop_uri}> ?val . }} LIMIT {_MAX_ENUM_VALUES + 1}"
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.timeout = 5
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        logging.warning(f"[_query_property_values] Value query failed for <{prop_prefixed}>: {e}")
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
            values.append(_shorten_uri(val, kg_config))
        else:
            values.append('"' + val + '"')
    return values


def add_possible_values_to_shape(
    relevant_items: list,
    sparql_endpoint: str,
    kg_config: KGConfig,
) -> list:
    """
    For each prop -> range item, queries the A-Box for distinct values using
    cardinality filtering. Properties with few distinct values (controlled
    vocabulary) get their values listed; others are left as-is.

    Returns a list of annotated prop -> range strings.
    """
    # Separate items that need a value query from pass-through items,
    # preserving original order via an index map.
    indexed_results = {}
    futures_map = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, item in enumerate(relevant_items):
            item = item.strip()
            if not item:
                indexed_results[idx] = item
                continue
            m = _PROP_RANGE_RE.match(item)
            if not m:
                indexed_results[idx] = item
                continue
            prop, range_ = m.group(1), m.group(2)
            future = executor.submit(_query_property_values, prop, sparql_endpoint, kg_config)
            futures_map[future] = (idx, prop, range_)

        for future in as_completed(futures_map):
            idx, prop, range_ = futures_map[future]
            values = future.result()
            if values:
                indexed_results[idx] = f"{prop} -> {range_} [values: {', '.join(values)}]"
            else:
                indexed_results[idx] = f"{prop} -> {range_}"

    return [indexed_results[i] for i in sorted(indexed_results)]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_classify(label, llm) -> bool:
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
# Private pipeline helpers
# ---------------------------------------------------------------------------

def _resolve_label(label: str, kg_config: KGConfig) -> tuple:
    """
    Returns (label_clean, class_uri, resource_uri).
    If label is already a full URI it is used as-is; otherwise
    kg_config.label_transform is applied and namespace prefixes are prepended.
    """
    if label.startswith("http"):
        local = label.rsplit("/", 1)[-1]
        return local, label, label
    clean = kg_config.label_transform(label) if kg_config.label_transform else label
    return clean, kg_config.class_namespace + clean, kg_config.resource_namespace + clean


def _get_class_items(class_uri: str, endpoint: str, kg_config: KGConfig) -> list:
    """Fetch T-Box properties and merge in A-Box extra properties (deduped)."""
    items = _tbox_to_prop_range_items(get_tbox_properties(class_uri, endpoint), kg_config)
    extra = _tbox_to_prop_range_items(
        get_abox_extra_properties(class_uri, endpoint, kg_config.abox_extra_namespaces),
        kg_config,
    )
    seen = {item.split(" ->")[0].strip() for item in items}
    items += [e for e in extra if e.split(" ->")[0].strip() not in seen]
    return items


def _get_entity_items(
    resource_uri: str, label_clean: str, endpoint: str, kg_config: KGConfig
) -> list:
    """Run shexer for a named entity and parse the result into prop -> range items."""
    shape_uri = kg_config.shape_namespace + label_clean
    shex_str = _run_shexer_for_entity(resource_uri, shape_uri, endpoint, kg_config.namespaces_dict)
    return _parse_shex_to_prop_range_items(shex_str)


def _run_shexer_for_entity(
    entity_uri: str,
    shape_uri: str,
    endpoint: str,
    namespaces_dict: dict,
) -> str:
    """
    Run shexer for a single named entity and return the raw ShEx string.
    Returns empty string on failure.
    """
    shape_map_raw = f"<{entity_uri}>@<{shape_uri}>"
    try:
        shaper = Shaper(
            shape_map_raw=shape_map_raw,
            url_endpoint=endpoint,
            namespaces_dict=namespaces_dict,
            disable_comments=True,
        )
        return shaper.shex_graph(string_output=True) or ""
    except Exception as e:
        logging.warning(f"[_run_shexer_for_entity] shexer failed for <{entity_uri}>: {e}")
        return ""


def _process_entity_section(
    label_clean: str,
    items: list,
    nlq: str,
    llm,
    endpoint: str,
    kg_config: KGConfig,
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
    enriched = add_possible_values_to_shape(items, endpoint, kg_config)
    if not enriched:
        return ""
    indented = "\n".join(f"  {line}" for line in enriched)
    return f"{label_clean}:\n{indented}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_shape_generic(
    nlq: str,
    entity_labels: list,
    shapes_llm,
    kg_config: KGConfig,
) -> str:
    """
    Generate a multi-section shape document for the given entity labels,
    using the provided KGConfig to determine all KG-specific behaviour.

    Parameters
    ----------
    nlq:
        The natural-language question being answered.
    entity_labels:
        List of label strings or full URIs. If a label starts with "http" it
        is treated as a pre-built URI and used as-is; the local name
        (``label.rsplit("/", 1)[-1]``) is used as the section header.
        Otherwise ``kg_config.label_transform`` (if set) is applied first,
        then class and resource URIs are constructed by prepending
        ``kg_config.class_namespace`` / ``kg_config.resource_namespace``.
    shapes_llm:
        LLM instance used for CLASS/ENTITY classification and property
        relevance filtering.
    kg_config:
        KGConfig instance describing the target KG.

    Returns
    -------
    str or None
        Multi-section string (one section per entity) or None if no sections
        could be produced.
    """
    endpoint = kg_config.sparql_endpoint
    sections = []

    try:
        for label in entity_labels:
            label_clean, class_uri, resource_uri = _resolve_label(label, kg_config)

            if _llm_classify(label_clean, shapes_llm):
                items = _get_class_items(class_uri, endpoint, kg_config)
            else:
                items = _get_entity_items(resource_uri, label_clean, endpoint, kg_config)

            section = _process_entity_section(label_clean, items, nlq, shapes_llm, endpoint, kg_config)
            if section:
                sections.append(section)

    except Exception as e:
        logging.error(f"[generate_shape_generic] Failed: {e}", exc_info=True)
        return None

    return "\n\n".join(sections) if sections else None


# ---------------------------------------------------------------------------
# Pre-built DBpedia config
# ---------------------------------------------------------------------------

def _dbpedia_label_transform(label: str) -> str:
    """DBpedia convention: capitalize first char, replace spaces with underscores."""
    label = label.replace(" ", "_")
    return label[0].upper() + label[1:] if label else label


DBPEDIA_CONFIG = KGConfig(
    sparql_endpoint=os.getenv("DBPEDIA_SPARQL_URL"),
    uri_prefixes=[
        ("http://dbpedia.org/ontology/", "dbo:"),
        ("http://dbpedia.org/property/", "dbp:"),
        ("http://dbpedia.org/resource/", "dbr:"),
        ("http://www.w3.org/2001/XMLSchema#", "xsd:"),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
        ("http://xmlns.com/foaf/0.1/", "foaf:"),
        ("http://schema.org/", "schema:"),
    ],
    namespaces_dict={
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
    },
    class_namespace="http://dbpedia.org/ontology/",
    resource_namespace="http://dbpedia.org/resource/",
    shape_namespace="http://shapes.dbpedia.org/",
    abox_extra_namespaces=["http://dbpedia.org/property/"],
    label_transform=_dbpedia_label_transform,
)
