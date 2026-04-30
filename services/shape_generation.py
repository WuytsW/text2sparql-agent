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
    iri_expansion_prompt,
)
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

# Matches enriched lines with IRI range and concrete values: "prop -> IRI [values: dbr:X, ...]"
_IRI_VALUES_RE = _re.compile(r"^\s*(\S+)\s*->\s*IRI\s*\[values:\s*(.+)\]\s*$")

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

    #logging.info(f"[get_tbox_properties] Found {len(properties)} properties for <{class_uri}>")
    return properties


def get_abox_dbp_properties(class_uri: str, sparql_endpoint: str, sample_size: int = 3) -> list:
    """
    Returns dbp: (raw Wikipedia infobox) properties found on sample instances of a class.
    These are NOT in the T-Box but frequently hold the actual data values in DBpedia.
    Each result is a dict with 'prop' (full URI) and 'range' (empty string, unknown from A-box).
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
        #logging.warning(f"[get_abox_dbp_properties] A-Box dbp: query failed for <{class_uri}>: {e}")
        return []

    properties = []
    seen = set()
    for binding in result.get("results", {}).get("bindings", []):
        prop = binding.get("prop", {}).get("value", "")
        if prop and prop not in seen:
            seen.add(prop)
            properties.append({"prop": prop, "domain": "", "range": ""})

    #logging.info(f"[get_abox_dbp_properties] Found {len(properties)} dbp: properties for <{class_uri}>")
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
        #logging.warning(f"[_query_property_values] Failed for {prop_prefixed}: {e}")
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


def _query_entity_property_values(entity_uri: str, prop_prefixed: str, endpoint: str) -> list:
    """Query all values of a property for one specific entity (no cardinality cap)."""
    prop_uri = _expand_prefixed(prop_prefixed)
    if not prop_uri:
        return []
    query = f"SELECT DISTINCT ?val WHERE {{ <{entity_uri}> <{prop_uri}> ?val . }}"
    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.timeout = 15
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception:
        return []
    values = []
    for b in result.get("results", {}).get("bindings", []):
        val_data = b.get("val", {})
        val = val_data.get("value", "")
        if not val:
            continue
        if val_data.get("type") == "uri":
            values.append(_shorten_uri(val))
        else:
            values.append('"' + val + '"')
    return values


def add_possible_values_to_shape(relevant_items: list, sparql_endpoint: str, entity_uri: str = None) -> str:
    """
    For each prop -> range item, queries the A-Box for distinct values.
    When entity_uri is given (named entity), queries values for that specific
    entity instead of globally — avoids the >30-value cutoff for common props.
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
        if entity_uri:
            values = _query_entity_property_values(entity_uri, prop, sparql_endpoint)
        else:
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
    suggested = [i.strip() for i in response.content.strip().split(",") if i.strip()]

    # Build lookup from prop name -> original full item (shexer's authoritative version).
    # This guards against the LLM hallucinating properties that shexer never found.
    original_by_prop = {}
    for line in shape.splitlines():
        line = line.strip()
        m = _PROP_RANGE_RE.match(line)
        if m:
            original_by_prop[m.group(1)] = line

    validated = []
    for item in suggested:
        m = _PROP_RANGE_RE.match(item)
        if m and m.group(1) in original_by_prop:
            validated.append(original_by_prop[m.group(1)])
    return validated


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
        #logging.warning(f"[_run_shexer_for_entity] shexer failed for {label_clean}: {e}")
        return ""


def _resolve_entity_uri(label_clean: str, entity_uris: dict) -> str | None:
    """Find a DBpedia URI for label_clean from Falcon entity_uris dict.

    Tries exact match, underscore-to-space variant, and suffix match against URI values.
    """
    if not entity_uris:
        return None
    label_spaced = label_clean.replace("_", " ")
    for key, uri in entity_uris.items():
        if key in (label_clean, label_spaced):
            return uri
    label_lower = label_spaced.lower()
    for uri in entity_uris.values():
        suffix = uri.rstrip("/").rsplit("/", 1)[-1].replace("_", " ").lower()
        if suffix == label_lower:
            return uri
    return None


def _process_entity_section(
    label_clean: str,
    items: list,
    nlq: str,
    llm,
    endpoint: str,
    entity_uri: str = None,
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
    enriched = add_possible_values_to_shape(items, endpoint, entity_uri=entity_uri)
    if not enriched.strip():
        return ""
    indented = "\n".join(f"  {line}" for line in enriched.splitlines())
    return f"{label_clean}:\n{indented}"


def _expand_entity_links(
    label_clean: str,
    section: str,
    nlq: str,
    llm,
    endpoint: str,
    entity_uri: str = None,
) -> list:
    """
    Given an already-generated entity section string, find all IRI-range properties
    (with or without annotated values), ask the LLM which to follow, resolve values
    on-the-fly for bare IRI props, run shexer on each linked entity, and return
    additional section strings.
    """
    # Collect all IRI-range props; values list may be empty for unannotated ones
    iri_props = {}
    for line in section.splitlines():
        line = line.strip()
        m = _IRI_VALUES_RE.match(line)
        if m:
            prop, vals_str = m.group(1), m.group(2)
            iri_vals = [v.strip() for v in vals_str.split(",") if v.strip().startswith("dbr:")]
            iri_props[prop] = iri_vals
            continue
        m2 = _PROP_RANGE_RE.match(line)
        if m2 and m2.group(2) == "IRI" and m2.group(1) not in iri_props:
            iri_props[m2.group(1)] = []

    if not iri_props or not llm:
        return []

    iri_props_text = "\n".join(
        f"{prop} -> IRI" + (f" [values: {', '.join(vals)}]" if vals else "")
        for prop, vals in iri_props.items()
    )
    prompt = iri_expansion_prompt["en"].format(
        nlq=nlq, label=label_clean, iri_props=iri_props_text
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    selected_props = {p.strip() for p in response.content.strip().split(",") if p.strip()}

    if not selected_props:
        return []

    eff_entity_uri = entity_uri or f"http://dbpedia.org/resource/{label_clean}"
    sub_sections = []
    seen_labels = set()
    for prop in selected_props:
        if prop not in iri_props:
            continue
        vals = iri_props[prop]
        if not vals:
            all_vals = _query_entity_property_values(eff_entity_uri, prop, endpoint)
            vals = [v for v in all_vals if v.startswith("dbr:")]
        for iri_prefixed in vals[:2]:
            sub_label = iri_prefixed[4:]  # strip "dbr:"
            if sub_label in seen_labels:
                continue
            seen_labels.add(sub_label)
            shex_str = _run_shexer_for_entity(sub_label, endpoint, _NAMESPACES_DICT)
            items = _parse_shex_to_prop_range_items(shex_str)
            # Always filter sub-entity shapes — they won't hit the >30 threshold
            # inside _process_entity_section, so force selection here.
            if items and llm:
                items = select_relevant_shape_parts(nlq, "\n".join(items), llm, label=sub_label)
            sub_entity_uri = f"http://dbpedia.org/resource/{sub_label}"
            sub_section = _process_entity_section(
                sub_label, items, nlq, llm, endpoint, entity_uri=sub_entity_uri
            )
            if sub_section:
                sub_sections.append(sub_section)

    return sub_sections


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


def generate_shape(nlq: str, entity_labels: list, shapes_llm, entity_uris: dict = None):
    load_dotenv(dotenv_path=".env")
    endpoint = os.getenv("DBPEDIA_SPARQL_URL")
    #logging.info(f"[generate_shape] Entity labels: {entity_labels}")

    sections = []
    try:
        for label in entity_labels:
            label_clean = label.replace(" ", "_")
            label_clean = label_clean[0].upper() + label_clean[1:]
            #logging.info(f"[generate_shape] Processing '{label_clean}'")

            if _llm_classify(label_clean, shapes_llm):
                #logging.info(f"[generate_shape] '{label_clean}' -> CLASS (T-Box path)")
                class_uri = f"http://dbpedia.org/ontology/{label_clean}"
                props = get_tbox_properties(class_uri, endpoint)
                items = _tbox_to_prop_range_items(props)
                # Also fetch raw Wikipedia infobox (dbp:) properties from A-box sample instances.
                # These are absent from the T-Box but often hold the actual data values.
                dbp_props = get_abox_dbp_properties(class_uri, endpoint)
                dbp_items = _tbox_to_prop_range_items(dbp_props)
                # Merge, avoiding duplicates (dbo: items take precedence).
                existing_props = {item.split(" ->")[0].strip() for item in items}
                for dbp_item in dbp_items:
                    dbp_key = dbp_item.split(" ->")[0].strip()
                    if dbp_key not in existing_props:
                        items.append(dbp_item)
                        existing_props.add(dbp_key)
                section = _process_entity_section(
                    label_clean, items, nlq, shapes_llm, endpoint
                )
            else:
                #logging.info(f"[generate_shape] '{label_clean}' -> ENTITY (shexer path)")
                shex_str = _run_shexer_for_entity(label_clean, endpoint, _NAMESPACES_DICT)
                items = _parse_shex_to_prop_range_items(shex_str)
                entity_uri = _resolve_entity_uri(label_clean, entity_uris or {})
                section = _process_entity_section(
                    label_clean, items, nlq, shapes_llm, endpoint, entity_uri=entity_uri
                )
                if section:
                    sections.append(section)
                    if shapes_llm:
                        sub_sections = _expand_entity_links(
                            label_clean, section, nlq, shapes_llm, endpoint,
                            entity_uri=entity_uri,
                        )
                        sections.extend(sub_sections)
                section = None  # prevent double-append below
            if section:
                sections.append(section)

    except Exception as e:
        #logging.error(f"[generate_shape] Failed: {e}", exc_info=True)
        return None

    if not sections:
        #logging.warning(f"[generate_shape] No sections produced for labels: {entity_labels}")
        return None

    result = "\n\n".join(sections)

    return result
