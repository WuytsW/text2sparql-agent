import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz


def falcon_external_wikidata(text: str) -> list:
    """Call Falcon 2.0 in Wikidata mode. Returns [{label: uri}, ...] list."""
    url = "https://labs.tib.eu/falcon/falcon2/api"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    params = {"mode": "short", "wikidata": "1"}
    try:
        response = requests.post(url, headers=headers, json=data, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        items = []
        for entry in result.get("entities_wikidata", []) + result.get("relations_wikidata", []):
            label = entry.get("surface form") or entry.get("label", "")
            uri = entry.get("URI", "")
            if uri:
                items.append({label: uri})
        return items
    except Exception as e:
        logging.warning(f"[falcon_wikidata] failed for '{text}': {e}")
        return []


def wikidata_api_search(entity_name: str, lang: str = "en", similarity: int = 85) -> list:
    """Search Wikidata API for entity_name. Returns [{label: uri}, ...] list."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "format": "json",
        "language": lang,
        "uselang": lang,
        "type": "item",
        "limit": 3,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        results = []
        for entity in data.get("search", []):
            label = entity.get("label", "")
            qid = entity.get("id", "")
            if qid and fuzz.partial_ratio(entity_name.lower(), label.lower()) >= similarity:
                results.append({label: f"http://www.wikidata.org/entity/{qid}"})
        return results
    except Exception as e:
        logging.warning(f"[wikidata_api_search] failed for '{entity_name}': {e}")
        return []


def wikidata_el(nlq: str, ne_list: list) -> list:
    """Entity linking for Wikidata using Falcon 2.0 with API search fallback.

    Runs Falcon on the full NLQ and each entity label in parallel. For any
    entity that Falcon does not resolve, falls back to the Wikidata search API.
    Returns a deduplicated [{label: uri}, ...] list.
    """
    texts = [nlq] + list(ne_list)
    with ThreadPoolExecutor() as executor:
        falcon_results = list(executor.map(falcon_external_wikidata, texts))

    seen_uris = set()
    nel_list = []
    falcon_found_labels = set()

    for result in falcon_results:
        for item in result:
            uri = list(item.values())[0]
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                nel_list.append(item)
                falcon_found_labels.add(list(item.keys())[0].lower())

    # For entities not found by Falcon, fall back to Wikidata API
    missing = [ne for ne in ne_list if ne.lower() not in falcon_found_labels]
    if missing:
        with ThreadPoolExecutor() as executor:
            api_results = list(executor.map(wikidata_api_search, missing))
        for result in api_results:
            for item in result:
                uri = list(item.values())[0]
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    nel_list.append(item)

    return nel_list
