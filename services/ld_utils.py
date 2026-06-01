import os
import re
import time
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.plugins.sparql.parser import parseQuery


prefixes_list = [
    {"rdf": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"},
    {"rdfs": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"},
    {"foaf": "PREFIX foaf: <http://xmlns.com/foaf/0.1/>"},
    {"schema": "PREFIX schema: <http://schema.org/>"},
    {"skos": "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"},
    {"xsd": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"},
    {"pv": "PREFIX pv: <http://ld.company.org/prod-vocab/>"},
    {"ecc": "PREFIX ecc: <https://ns.eccenca.com/>"},
    {"void": "PREFIX void: <http://rdfs.org/ns/void#>"},
    {"vann": "PREFIX vann: <http://purl.org/vocab/vann/>"},
    {"dbp": "PREFIX dbp: <http://dbpedia.org/property/>"},
    {"dbo": "PREFIX dbo: <http://dbpedia.org/ontology/>"},
    {"dbr": "PREFIX dbr: <http://dbpedia.org/resource/>"},
    {"res": "PREFIX res: <http://dbpedia.org/resource/>"},
    {"dct": "PREFIX dct: <http://purl.org/dc/terms/>"},
    {"dbc": "PREFIX dbc: <http://dbpedia.org/resource/Category:>"},
]

wikidata_prefixes_list = [
    {"wd": "PREFIX wd: <http://www.wikidata.org/entity/>"},
    {"wdt": "PREFIX wdt: <http://www.wikidata.org/prop/direct/>"},
    {"p": "PREFIX p: <http://www.wikidata.org/prop/>"},
    {"ps": "PREFIX ps: <http://www.wikidata.org/prop/statement/>"},
    {"pq": "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>"},
    {"wikibase": "PREFIX wikibase: <http://wikiba.se/ontology#>"},
    {"bd": "PREFIX bd: <http://www.bigdata.com/rdf#>"},
    {"rdfs": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"},
    {"xsd": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"},
    {"schema": "PREFIX schema: <http://schema.org/>"},
]

_WIKIDATA_USER_AGENT = "text2sparql-agent/1.0 (https://github.com/WuytsW; contact: wuytswillem@gmail.com)"


def post_process_wikidata(query: str) -> str:
    """Extract SPARQL from code fences, inject missing Wikidata prefixes."""
    blocks = extract_code_blocks(query)
    query = blocks[0].strip() if blocks else query.strip()
    query = fix_union_syntax(query)
    try:
        parse_object = parseQuery(query)
        existing = [p.prefix for p in parse_object[0]] if parse_object[0] else []
        missing = "\n".join(
            list(p.values())[0]
            for p in wikidata_prefixes_list
            if list(p.keys())[0] not in existing
        )
        return (missing + "\n" + query).strip() if missing else query
    except Exception:
        return query


def execute_wikidata(query: str, max_retries: int = 5) -> dict:
    """Execute SPARQL against the Wikidata endpoint with retry on HTTP 429."""
    processed = post_process_wikidata(query)
    endpoint = os.getenv("WIKIDATA_SPARQL_URL", "https://query.wikidata.org/sparql")
    headers = {
        "User-Agent": _WIKIDATA_USER_AGENT,
        "Accept": "application/sparql-results+json",
        "Accept-Encoding": "gzip,deflate",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                endpoint,
                params={"query": processed, "format": "json"},
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt)
    return {"error": "max retries exceeded"}


def extract_code_blocks(text):
    import re
    pattern = r'```sparql(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    if len(code_blocks) > 0:
        return code_blocks
    else:
        pattern = r'```(.*?)```'
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return code_blocks

def fix_union_syntax(query: str) -> str:
    """Rewrite top-level UNION to valid nested UNION inside WHERE clause.

    Converts:
      SELECT ... WHERE { body1 } UNION { SELECT ... WHERE { body2 } } UNION ...
    To:
      SELECT ... WHERE { { body1 } UNION { body2 } UNION ... }
    """
    where_match = re.search(r'\bWHERE\s*\{', query, re.IGNORECASE)
    if not where_match:
        return query

    where_open = where_match.end() - 1

    depth = 0
    first_where_close = -1
    for i in range(where_open, len(query)):
        if query[i] == '{':
            depth += 1
        elif query[i] == '}':
            depth -= 1
            if depth == 0:
                first_where_close = i
                break

    if first_where_close == -1:
        return query

    after = query[first_where_close + 1:].lstrip()
    if not after.upper().startswith('UNION'):
        return query

    select_prefix = query[:where_open + 1]
    first_body = query[where_open + 1:first_where_close]

    def extract_block_body(text):
        """Return (content_inside_braces, rest_of_text) for the leading { } block."""
        if not text.startswith('{'):
            return None, text
        depth = 0
        for i in range(len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[1:i], text[i + 1:].lstrip()
        return None, text

    def unwrap_sub_select(body):
        """Strip SELECT ... WHERE { } wrapper, keeping only the inner body."""
        body = body.strip()
        m = re.match(r'SELECT\b.*?\bWHERE\s*\{', body, re.IGNORECASE | re.DOTALL)
        if not m:
            return body
        inner_open = m.end() - 1
        depth = 0
        for i in range(inner_open, len(body)):
            if body[i] == '{':
                depth += 1
            elif body[i] == '}':
                depth -= 1
                if depth == 0:
                    return body[inner_open + 1:i]
        return body

    union_bodies = []
    remaining = after
    while remaining.upper().startswith('UNION'):
        after_kw = remaining[5:].lstrip()
        block_content, remaining = extract_block_body(after_kw)
        if block_content is None:
            break
        union_bodies.append(unwrap_sub_select(block_content))

    all_bodies = [first_body] + union_bodies
    union_parts = ' UNION '.join(f'{{ {b.strip()} }}' for b in all_bodies)
    return f'{select_prefix} {union_parts} }}'


def post_process(result):
    blocks = extract_code_blocks(result)
    if len(blocks) > 0:
        query = blocks[0].strip()
    else:
        query = result.strip()

    query = fix_union_syntax(query)

    parse_object = parseQuery(query)
    query_prefixes = [prefix.prefix for prefix in parse_object[0]] if len(parse_object[0]) > 0 else []
    query = '\n'.join([list(pref.values())[0] for pref in prefixes_list if list(pref.keys())[0] not in query_prefixes]) + '\n' + query

    return query

def execute(query: str, endpoint_url: str = 'https://dbpedia.org/sparql'):
    try:
        query = post_process(query)
        parse_object = parseQuery(query)

        query_prefixes = [prefix.prefix for prefix in parse_object[0]] if len(parse_object[0]) > 0 else []

        query = '\n'.join([list(pref.values())[0] for pref in prefixes_list if list(pref.keys())[0] not in query_prefixes]) + '\n' + query
        
        sparql = SPARQLWrapper(endpoint_url)
        sparql.timeout = 20
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        response = sparql.query().convert()
        return response
    except Exception as e:
        e = str(e)
        print(e)
        if 'MalformedQueryException' in e or 'bad formed' in e:
            return {'error': str(e)}
        return  {'error': str(e)}

