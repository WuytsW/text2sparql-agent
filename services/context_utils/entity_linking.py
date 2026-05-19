import requests
from concurrent.futures import ThreadPoolExecutor


def falcon_external(text: str):
    url = 'https://labs.tib.eu/falcon/falcon2/api'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}
    params = {'mode': 'long', 'db': '1'}
    response = requests.post(url, headers=headers, json=data, params=params, timeout=10)
    return response.json()


def spotlight_external(text: str, confidence: float = 0.35) -> dict:
    url = 'https://api.dbpedia-spotlight.org/en/annotate'
    headers = {'Accept': 'application/json'}
    data = {'text': text, 'confidence': confidence}
    response = requests.post(url, headers=headers, data=data, timeout=15)
    response.raise_for_status()
    return response.json()


def dbpedia_el(nlq: str, _ne_list: list) -> list:
    """Performs entity linking to DBpedia via DBpedia Spotlight.
    Returns list of dict with linking candidates: [{"surfaceForm": "URI"}]"""
    result = spotlight_external(nlq)
    seen = set()
    nel_list = []
    for resource in result.get("Resources", []):
        uri = resource.get("@URI")
        label = resource.get("@surfaceForm", uri)
        if uri and uri not in seen:
            seen.add(uri)
            nel_list.append({label: uri})
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
