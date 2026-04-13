import requests


def falcon_external(text: str):
    url = 'https://labs.tib.eu/falcon/falcon2/api'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}
    params = {'mode': 'long', 'db': '1'}
    response = requests.post(url, headers=headers, json=data, params=params, timeout=5)
    return response.json()


def dbpedia_el(nlq: str, ne_list: list) -> list:
    """Performs entity linking to DBpedia using both the full question and individual named entities.
    Returns list of dict with linking candidates: [{"label": "URI"}]"""
    seen = set()
    nel_list = []

    for text in [nlq] + ne_list:
        falcon_result = falcon_external(text=text)
        for item in falcon_result.get("entities_dbpedia", []) + falcon_result.get("relations_dbpedia", []):
            uri = list(item.values())[0] if item else None
            if uri and uri not in seen:
                seen.add(uri)
                nel_list.append(item)

    return nel_list
