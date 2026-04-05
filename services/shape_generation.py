import os
import requests
from shexer.shaper import Shaper
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def resolve_redirect(entity_uri, sparql_url):
    """Follow dbo:wikiPageRedirects to the canonical DBpedia resource."""
    query = f"""
    SELECT ?target WHERE {{
        <{entity_uri}> <http://dbpedia.org/ontology/wikiPageRedirects> ?target .
    }} LIMIT 1
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["target"]["value"]
    except Exception:
        pass
    return entity_uri


def get_ontology_class_uri(label_clean, sparql_url):
    """Check if dbo:{label} is an OWL class. Returns the class URI if so, else None."""
    class_uri = f"http://dbpedia.org/ontology/{label_clean}"
    query = f"""
    ASK {{ <{class_uri}> a <http://www.w3.org/2002/07/owl#Class> }}
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        if resp.json().get("boolean", False):
            return class_uri
    except Exception:
        pass
    return None


def fetch_class_instances(class_uri, sparql_url, limit=10):
    """Fetch sample instance URIs of a DBpedia ontology class, excluding persons."""
    query = f"""
    SELECT DISTINCT ?entity WHERE {{
        ?entity a <{class_uri}> .
        FILTER NOT EXISTS {{ ?entity a <http://dbpedia.org/ontology/Person> }}
        FILTER(STRSTARTS(STR(?entity), "http://dbpedia.org/resource/"))
    }} LIMIT {limit}
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [b["entity"]["value"] for b in bindings]
    except Exception:
        return []


def fetch_ancestor_classes(class_uri, sparql_url):
    """Fetch all ancestor classes via rdfs:subClassOf+ traversal, within dbo: namespace."""
    query = f"""
    SELECT DISTINCT ?ancestor WHERE {{
        <{class_uri}> <http://www.w3.org/2000/01/rdf-schema#subClassOf>+ ?ancestor .
        FILTER(STRSTARTS(STR(?ancestor), "http://dbpedia.org/ontology/"))
    }}
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [b["ancestor"]["value"] for b in bindings]
    except Exception:
        return []


def fetch_properties_of_class(class_uri, sparql_url):
    """Fetch dbo: properties with their rdfs:range for this class (ontology-level lookup)."""
    query = f"""
    SELECT DISTINCT ?property ?range WHERE {{
        ?property <http://www.w3.org/2000/01/rdf-schema#domain> <{class_uri}> .
        OPTIONAL {{ ?property <http://www.w3.org/2000/01/rdf-schema#range> ?range }}
        FILTER(STRSTARTS(STR(?property), "http://dbpedia.org/ontology/"))
    }}
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [(b["property"]["value"], b.get("range", {}).get("value", "")) for b in bindings]
    except Exception:
        return []


LITERAL_RANGES = {"string", "integer", "float", "double", "boolean", "date", "dateTime", "gYear", "nonNegativeInteger"}


def fetch_property_values(property_uri, sparql_url, limit=10):
    """Fetch distinct literal values for a property across all DBpedia instances."""
    query = f"""
    SELECT DISTINCT ?value WHERE {{
        ?instance <{property_uri}> ?value .
        FILTER(isLiteral(?value))
    }} LIMIT {limit}
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [b["value"]["value"] for b in bindings]
    except Exception:
        return []


def get_raw_ancestor_props(class_uri, sparql_url):
    """Return all ancestor properties without fetching values.
    Returns list of (ancestor_name, prop_name, prop_uri, range_name)."""
    ancestors = fetch_ancestor_classes(class_uri, sparql_url)
    result = []
    for ancestor in ancestors:
        ancestor_name = ancestor.split("/")[-1]
        for prop_uri, range_uri in fetch_properties_of_class(ancestor, sparql_url):
            prop_name = prop_uri.split("/")[-1]
            range_name = range_uri.split("/")[-1].split("#")[-1] if range_uri else ""
            result.append((ancestor_name, prop_name, prop_uri, range_name))
    return result


def append_ancestor_props_with_values(shape, selected_props, class_name, sparql_url):
    """Append selected ancestor properties with values to the shape.
    selected_props: list of (ancestor_name, prop_name, prop_uri, range_name)."""
    if not selected_props:
        return shape

    lines = [f"\n# Properties from ancestor classes (usable on dbo:{class_name} — do NOT change rdf:type):"]
    by_ancestor = {}
    for ancestor_name, prop_name, prop_uri, range_name in selected_props:
        by_ancestor.setdefault(ancestor_name, []).append((prop_name, prop_uri, range_name))

    for ancestor_name, props in by_ancestor.items():
        lines.append(f"# {ancestor_name}:")
        for prop_name, prop_uri, range_name in props:
            if range_name in LITERAL_RANGES:
                values = fetch_property_values(prop_uri, sparql_url)
                values_str = ", ".join(f'"{v}"' for v in values)
                lines.append(
                    f"#   dbo:{prop_name} (range: {range_name}) — possible values: {values_str}"
                    if values_str else
                    f"#   dbo:{prop_name} (range: {range_name})"
                )
            elif range_name:
                lines.append(f"#   dbo:{prop_name} (range: {range_name})")
            else:
                lines.append(f"#   dbo:{prop_name}")

    return shape + "\n" + "\n".join(lines)


NAMESPACES_DICT = {
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
    "http://shapes.dbpedia.org/": "shapes"
}


def generate_shex_only(dbpedia_sparql_url, entity_labels):
    """Run shexer and return (shape_str, ontology_classes).
    ontology_classes: {label: class_uri} for entities that matched a dbo: OWL class."""
    print(f"Entity labels: {entity_labels}")
    shape_lines = []
    ontology_classes = {}

    try:
        for label in entity_labels:
            label_clean = "_".join(word.capitalize() for word in label.split())
            original_uri = f"http://dbpedia.org/resource/{label_clean}"
            entity_id = resolve_redirect(original_uri, dbpedia_sparql_url)
            resource_name = entity_id.split("/")[-1]
            shape_label = f"http://shapes.dbpedia.org/{resource_name}"

            ontology_class = get_ontology_class_uri(label_clean, dbpedia_sparql_url)
            if ontology_class:
                instances = fetch_class_instances(ontology_class, dbpedia_sparql_url)
                for inst in instances:
                    shape_lines.append(f"<{inst}>@<{shape_label}>")
                ontology_classes[label_clean] = ontology_class
            else:
                shape_lines.append(f"<{entity_id}>@<{shape_label}>")

        shape_map_raw = "\n".join(shape_lines)
        print(f"Generated shape map:\n{shape_map_raw}")

        shaper = Shaper(
            shape_map_raw=shape_map_raw,
            url_endpoint=dbpedia_sparql_url,
            namespaces_dict=NAMESPACES_DICT,
            disable_comments=True,
        )
        shape = shaper.shex_graph(string_output=True)
        return shape, ontology_classes
    except Exception as e:
        print(f"Error generating ShEx graph: {e}")
        return None, {}


def generate_shape(entities):
    load_dotenv(dotenv_path=".env")
    dbpedia_sparql_url = os.getenv("DBPEDIA_SPARQL_URL")
    print(f"✅ Generating shape using sparql endpoint {dbpedia_sparql_url}.")

    shape, ontology_classes = generate_shex_only(dbpedia_sparql_url, entities)
    if shape and ontology_classes:
        for label, class_uri in ontology_classes.items():
            raw_props = get_raw_ancestor_props(class_uri, dbpedia_sparql_url)
            shape = append_ancestor_props_with_values(shape, raw_props, label, dbpedia_sparql_url)
    print(f"✅ Shape generation successful: {shape}")
    return shape
