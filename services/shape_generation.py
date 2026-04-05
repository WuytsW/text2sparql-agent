import os
import requests
from shexer.shaper import Shaper
from services.utility import Utils
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

def add_categories(shape, entity_uris, dbpedia_sparql_url):
    """Append dcterms:subject categories (own + two-hop related) to the shape string.
    Comment out the call in generate_combined_shape to disable category fetching."""
    category_lines = []
    for uri in entity_uris:
        resource_name = uri.split("/")[-1]
        own_cats = fetch_categories(uri, dbpedia_sparql_url)
        related_cats = fetch_related_categories(uri, resource_name, dbpedia_sparql_url)
        all_cats = list(dict.fromkeys(own_cats + related_cats))
        if all_cats:
            category_lines.append(f"\n# dcterms:subject categories for {resource_name}:")
            for cat in all_cats:
                category_lines.append(f"#   <{cat}>")
    if category_lines:
        shape += "\n" + "\n".join(category_lines)
    return shape



def fetch_categories(entity_uri, sparql_url):
    """Fetch dcterms:subject category URIs for an entity."""
    query = f"""
    SELECT ?category WHERE {{
        <{entity_uri}> <http://purl.org/dc/terms/subject> ?category .
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
        return [b["category"]["value"] for b in bindings]
    except Exception:
        return []


def fetch_related_categories(entity_uri, resource_name, sparql_url):
    """Two-hop lookup: entity → wikiPageWikiLink → dcterms:subject, filtered for categories
    containing the entity name. Finds categories like 'Assassins_of_Julius_Caesar' which
    are attached to related entities, not to the entity itself."""
    query = f"""
    SELECT DISTINCT ?category WHERE {{
        <{entity_uri}> <http://dbpedia.org/ontology/wikiPageWikiLink> ?related .
        ?related <http://purl.org/dc/terms/subject> ?category .
        FILTER(CONTAINS(STR(?category), "{resource_name}"))
    }} LIMIT 30
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=15
        )
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [b["category"]["value"] for b in bindings]
    except Exception:
        return []


def generate_combined_shape(dbpedia_sparql_url, entity_labels):
    print(f"Entity labels: {entity_labels}")
    shape_lines = []
    entity_uris = []

    try:
        for label in entity_labels:
            label_clean = label.replace(' ', '_')
            entity_id = f"http://dbpedia.org/resource/{label_clean}"
            entity_id = resolve_redirect(entity_id, dbpedia_sparql_url)
            resource_name = entity_id.split("/")[-1]
            shape_label = f"http://shapes.dbpedia.org/{resource_name}"
            shape_lines.append(f"<{entity_id}>@<{shape_label}>")
            entity_uris.append(entity_id)

        shape_map_raw = "\n".join(shape_lines)
        print(f"Generated shape map:\n{shape_map_raw}")

        namespaces_dict = {
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

        shaper = Shaper(
            shape_map_raw=shape_map_raw,
            url_endpoint=dbpedia_sparql_url,
            namespaces_dict=namespaces_dict,
            disable_comments=True,
        )
        shape = shaper.shex_graph(string_output=True)
        # shape = add_categories(shape, entity_uris, dbpedia_sparql_url)

        print(f"✅ Shape generation successful: {shape}")
        return shape
    except Exception as e:
        # print(f"Error generating ShEx graph: {e}")
        return None


def generate_shape(entities):

    load_dotenv(dotenv_path=".env")
    dbpedia_sparql_url = os.getenv("DBPEDIA_SPARQL_URL")
    
    
    print(f"✅ Generating shape using sparql endpoint {dbpedia_sparql_url} and generated shapes.")
    return generate_combined_shape(dbpedia_sparql_url, entities)
