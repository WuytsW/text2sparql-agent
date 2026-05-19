import os
import requests
from concurrent.futures import ThreadPoolExecutor
from SPARQLWrapper import SPARQLWrapper, JSON


def falcon_external(text: str):
    url = 'https://labs.tib.eu/falcon/falcon2/api'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}
    params = {'mode': 'long', 'db': '1'}
    response = requests.post(url, headers=headers, json=data, params=params, timeout=15)
    return response.json()


def spotlight_external(text: str, confidence: float = 0.35) -> dict:
    url = 'https://api.dbpedia-spotlight.org/en/annotate'
    headers = {'Accept': 'application/json'}
    data = {'text': text, 'confidence': confidence}
    response = requests.post(url, headers=headers, data=data, timeout=15)
    response.raise_for_status()
    return response.json()


def sparql_entity_lookup(entity_name: str) -> list[dict]:
    """Finds DBpedia resources matching the given label via rdfs:label lookup."""
    endpoint = os.getenv("DBPEDIA_SPARQL_URL", "https://dbpedia.org/sparql")
    safe = entity_name.replace("\\", "\\\\").replace('"', '\\"')
    query = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity WHERE {{
  ?entity rdfs:label "{safe}"@en .
  FILTER(STRSTARTS(STR(?entity), "http://dbpedia.org/resource/"))
  FILTER(!CONTAINS(STR(?entity), "Category:"))
}} LIMIT 5"""
    sparql = SPARQLWrapper(endpoint)
    sparql.timeout = 15
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    bindings = result.get("results", {}).get("bindings", [])
    return [{entity_name: b["entity"]["value"]} for b in bindings]


def dbpedia_el(nlq: str, ne_list: list) -> list:
    """Performs entity linking to DBpedia via SPARQL label lookup.
    Returns list of dict with linking candidates: [{"label": "URI"}]"""
    seen = set()
    nel_list = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(sparql_entity_lookup, ne_list))
    for items in results:
        for item in items:
            uri = list(item.values())[0]
            if uri and uri not in seen:
                seen.add(uri)
                nel_list.append(item)
    return nel_list


def dbpedia_el_falcon(nlq: str, ne_list: list) -> list:
    """Performs entity linking to DBpedia using both the full question and individual named entities.
    Returns list of dict with linking candidates: [{"label": "URI"}]"""
    seen = set()
    nel_list = []

    texts = [nlq] + ne_list
    with ThreadPoolExecutor() as executor:
        falcon_results = list(executor.map(falcon_external, texts))

    for falcon_result in falcon_results:
        for item in falcon_result.get("entities_dbpedia", []) + falcon_result.get("relations_dbpedia", []):
            uri = list(item.values())[0] if item else None
            if uri and uri not in seen:
                seen.add(uri)
                nel_list.append(item)

    return nel_list
