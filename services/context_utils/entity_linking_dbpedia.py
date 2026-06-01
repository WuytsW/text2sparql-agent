import logging
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from SPARQLWrapper import SPARQLWrapper, JSON


def falcon_external(text: str):
    url = 'https://labs.tib.eu/falcon/falcon2/api'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}
    params = {'mode': 'long', 'db': '1'}
    response = requests.post(url, headers=headers, json=data, params=params, timeout=30)
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


def dbpedia_el_sparql(nlq: str, ne_list: list) -> list:
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


def dbpedia_el(nlq: str, ne_list: list) -> list:
    """Performs entity linking to DBpedia using both the full question and individual named entities.
    Returns list of dict with linking candidates: [{"label": "URI"}]"""
    seen = set()
    nel_list = []

    texts = [nlq] + ne_list
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(falcon_external, t) for t in texts]
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                logging.debug(f"[dbpedia_el] falcon failed for one text: {e}")
                continue
            for item in result.get("entities_dbpedia", []) + result.get("relations_dbpedia", []):
                uri = list(item.values())[0] if item else None
                if uri and uri not in seen:
                    seen.add(uri)
                    nel_list.append(item)

    if not nel_list and ne_list:
        logging.debug("[dbpedia_el] falcon returned nothing, falling back to SPARQL")
        nel_list = dbpedia_el_sparql(nlq, ne_list)

    logging.debug(f"[dbpedia_el] found {len(nel_list)} entities: {nel_list}")
    return nel_list
