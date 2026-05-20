system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system that generates SPARQL queries over Wikidata.
    """
}


entity_profile_selection_prompt = {
    "en": """Given the following question: "{nlq}", and the following entity profile: "{entity_profile}"
    select the most relevant properties and classes from the entity profile that are likely to be useful for answering the question.
    Return a comma-separated list of properties and classes from the entity profile that are relevant to the question. Only select properties and classes that are likely to be useful for answering the question. Do not select all properties, only the most relevant ones.
    Keep the formatting of the properties as they are in the entity profile (Example: wdt:P17 -> IRI).
    If the entity profile is empty, return an empty string."""
}

entity_profile_selection_prompt_per_entity = {
    "en": """Given the question: "{nlq}", and the following properties for "{label}":

{entity_profile}

Select only the most relevant properties needed to answer the question. (If you are not sure about selecting a property, it's better to include it than to miss it. Although, try to avoid including too many irrelevant properties as it may lead to slow query execution or empty results.)
Return a comma-separated list in the exact format shown (e.g. wdt:P17 -> IRI).
If none are relevant, return an empty string."""
}

class_instances_prompt = {
    "en": """Determine if the term "{label}" refers to a specific named entity or a general class/type of things.

A NAMED ENTITY is a unique, specific thing: a particular person, place, organization, creative work, etc.
Examples: "Germany", "Eiffel Tower", "Apple Inc.", "Douglas Adams"

A CLASS/TYPE is a general category that many things can belong to.
Examples: "Country", "Musical Artist", "Film", "Weapon", "City", "Human"

If "{label}" is a NAMED ENTITY, respond with exactly:
ENTITY

If "{label}" is a CLASS/TYPE, respond with exactly:
CLASS

Examples:
"Germany" → ENTITY
"Country" → CLASS"""
}


context_check_prompt = {
    "en": """You are evaluating whether the context extracted from Wikidata contains enough relevant information to answer the question with a SPARQL query.

Question: "{nlq}"

Extracted entities: {entities}

Entity URIs (linked Wikidata resources):
{entity_uris}

Entity Profile (available wdt: properties per entity/class):
{entity_profile}

Evaluate the context AS A WHOLE. It is USEFUL if any part of it — the entity profile or the entity URIs — provides information that could plausibly help construct a SPARQL query to answer the question. Individual components may be empty or sparse; that is fine as long as the overall context is helpful.

It is NOT USEFUL only if the context as a whole is clearly off-topic, completely empty, or the extracted entities are so wrong that no part of the context relates to the question.

Respond with ONLY a JSON object (no markdown):
{{"valid": true, "reason": "brief explanation"}}
or
{{"valid": false, "reason": "what is wrong and what entity labels might work better"}}"""
}

entities_extraction_prompt = {
"en": """Extract the Wikidata entity and class labels needed to answer the following question with a SPARQL query.

Question: "{nlq}"

Rules:
- Return ONLY a comma-separated list of labels, no explanations.
- Use singular form and capitalise as a Wikidata item label would be (e.g. "city" or "City").
- Descriptive adjectives like "largest", "extinct", "female" are filters, NOT entities — do not include them.
- Titles of creative works (books, films, games, albums, TV series, etc.) are single named entities regardless of how many words they contain. Treat the full title as one item (e.g. "The Hitchhiker's Guide to the Galaxy", NOT "hitchhiker", "galaxy").
- Full person names must be kept together (e.g. "Douglas Adams" NOT "Douglas" or "Adams").
- If a named entity is referred to only by a partial name, expand it to the most complete, commonly recognized form (e.g. "Napoleon" → "Napoleon Bonaparte").
- Only include a class label if it appears as an explicit noun category in the question (e.g. "novelist", "weapon", "country").
- Never extract "person" or "human" — they are too generic.

Example: "Who developed Skype?"
Result: Skype

Example: "Which countries border Germany?"
Result: Germany, country

Example: "Where was Marie Curie born?"
Result: Marie Curie

Example: "Show me all museums in Paris."
Result: museum, Paris
"""
}

sparql_planner_prompt = {
    "en": """You are a SPARQL query generation orchestrator for Wikidata.
The context (entity URIs, Wikidata entity profile) has already been provided in the conversation history.

Follow this exact workflow:
1. Call generate_sparql with no suggestions on the first attempt.
2. Call execute_sparql with the returned query.
3. Call check_result with the query and execution results.
4. If check_result returns ok=true, output the final SPARQL query and stop.
5. If check_result returns ok=false and you have made fewer than 3 attempts, call generate_sparql again passing the suggestions, then repeat steps 2-3.
6. After 3 attempts, output the best query you have, even if imperfect.

Never skip execute_sparql after generating a query. Output only the final plain SPARQL query string, no markdown, no explanation."""
}


check_result_prompt = {
    "en": """Evaluate whether this SPARQL query correctly answers the question.

Question: {question}
Query:
{query}

Execution result:
{execution_result}

If results are non-empty, or the result is a boolean ({{"boolean": true}} or {{"boolean": false}}) which is a valid ASK answer, and the result answers the question (do not be strict — a non-empty result is usually acceptable), respond with exactly (JSON only, no markdown):
{{"ok": true}}

If results are empty, an error, or do not answer the question respond with (JSON only, no markdown):
{{"ok": false, "suggestions": "<concrete fix suggestions based on the entity profile and context in the conversation>"}}"""
}

_sparql_rules = """SPARQL generation rules for Wikidata:
- Output only a plain SPARQL query. No markdown, no explanation, no code fences.
- Use Wikidata prefixes: wd: for items (wd:Q...), wdt: for direct properties (wdt:P...).
- For qualified statements (e.g. a value with a point-in-time qualifier), use the full statement pattern:
    ?item p:P... ?stmt . ?stmt ps:P... ?value . ?stmt pq:P... ?qualifier .
- To retrieve human-readable labels, add at the end of the WHERE clause:
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
  and use ?itemLabel, ?valueLabel etc. for display.
- CRITICAL: Use the EXACT property shown in the entity profile — do not substitute wdt: for p: or vice versa.
- If the answer is a date, it is already typed as xsd:dateTime in Wikidata; cast if needed: BIND(xsd:date(STR(?raw)) AS ?date)
- For a list of things, use a single ?item or ?itemLabel column.
- UNION must be wrapped inside the WHERE clause: SELECT ?item WHERE {{ {{ ... }} UNION {{ ... }} }}
- Do not use dbo:, dbr:, or any DBpedia prefixes."""

sparql_agent_prompt = {
    "en": f"""You are a SPARQL query generation agent for Wikidata.
The conversation history contains the entity URIs and Wikidata entity profile needed to answer the question.

IMPORTANT: You MUST call execute_sparql before producing any output. Never output a query without first executing it. Your very first action must be a call to execute_sparql.

The tool returns two fields:
  [Query]: the SPARQL query that was executed
  [Result]: the execution result — a list of bindings, a boolean (for ASK queries), or an error

Workflow:
1. Call execute_sparql with your best SPARQL query for the question.
2. Check [Result]:
   - Non-empty list or a boolean value → the query worked. Output ONLY the plain SPARQL query from [Query] and stop.
   - Empty list or error → call execute_sparql again with an improved query.
3. Repeat up to 3 times total. After 3 attempts, output the query from [Query] of your best attempt.

{_sparql_rules}

Output only the final plain SPARQL query string. No markdown, no code fences, no explanation."""
}


generation_prompt = {
    "en": f"""Using the context provided above (entity URIs and entity profile), generate a SPARQL query to answer the following question.

Question: {{question}}
{{suggestions_block}}
{_sparql_rules}"""
}
