# ── 1. Agent initialization ──────────────────────────────────────────────────

system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system that generates SPARQL queries over DBpedia.
    """
}

# ── 2. Entity extraction ──────────────────────────────────────────────────────

entities_extraction_prompt = {
"en": """Extract the DBpedia entity and class labels needed to answer the following question with a SPARQL query.

Question: "{nlq}"

Rules:
- Return ONLY a comma-separated list of labels, no explanations.
- Use singular form and capitalise as a DBpedia class would be.
- Descriptive adjectives are filters, NOT entities — do not include them.
- Titles of creative works are single named entities regardless of how many words they contain. Treat the full title as one item.
- Full person names must be kept together.
- If a named entity is referred to only by a partial name, expand it to the most complete, commonly recognized form.
- Only include a class label if it appears as an explicit noun category in the question.
- Never extract "Person" — it is too generic.
"""
}

# ── 3. Entity classification (named entity vs. class/type) ───────────────────

class_instances_prompt = {
    "en": """Determine if the term "{label}" refers to a specific named entity or a general class/type of things.

A NAMED ENTITY is a unique, specific thing: a particular person, place, organization, creative work, etc.

A CLASS/TYPE is a general category that many things can belong to.

If "{label}" is a NAMED ENTITY, respond with exactly:
ENTITY

If "{label}" is a CLASS/TYPE, respond with exactly:
CLASS"""
}

# ── 4. Entity profile selection ───────────────────────────────────────────────

entity_profile_selection_prompt = {
    "en": """Given the folowing question: "{nlq}", and the following entity profile: "{entity_profile}"
    select the most relevant properties and classes from the entity profile that are likely to be useful for answering the question.
    Return a comma-separated list of properties and classes from the entity profile that are relevant to the question. Only select properties and classes that are likely to be useful for answering the question. Do not select all properties, only the most relevant ones.
    Keep the formatting of the properties and classes as they are in the entity profile.
    If the entity profile is empty, return an empty string."""
}

entity_profile_selection_prompt_per_entity = {
    "en": """Given the question: "{nlq}", and the following properties for "{label}":

{entity_profile}

Select only the most relevant properties needed to answer the question. (If you are not sure about selecting a property, it's better to include it than to miss it. Altough, try to avoid including too many irrelevant properties as it may lead to slow query execution or empty results.)
Return a comma-separated list in the exact format shown.
If none are relevant, return an empty string."""
}

# ── 5. Category selection ─────────────────────────────────────────────────────

category_selection_prompt = {
    "en": """Given the question: "{nlq}", and the following DBpedia category URIs found:

{categories}

Select only the category URIs that are most likely relevant to answering the question.
Only return categories that ar elikely to return the answer to the question with a query of the form: ?uri dct:subject dbc:CategoryName.
Return a comma-separated list of the full URIs.
If none are relevant, return an empty string."""
}

# ── 6. Context validation ─────────────────────────────────────────────────────

context_check_prompt = {
    "en": """You are evaluating whether the context extracted from DBpedia contains enough relevant information to answer the question with a SPARQL query.

Question: "{nlq}"

Extracted entities: {entities}

Entity URIs (linked DBpedia resources):
{entity_uris}

DBpedia categories:
{categories}

Entity Profile (available properties per entity/class):
{entity_profile}

Evaluate the context AS A WHOLE. It is USEFUL if any part of it — the entity profile, the entity URIs, or the categories — provides information that could plausibly help construct a SPARQL query to answer the question. Individual components may be empty or sparse; that is fine as long as the overall context is helpful.

It is NOT USEFUL only if the context as a whole is clearly off-topic, completely empty, or the extracted entities are so wrong that no part of the context relates to the question.

Respond with ONLY a JSON object (no markdown):
{{"valid": true, "reason": "brief explanation"}}
or
{{"valid": false, "reason": "what is wrong and what entity labels might work better"}}"""
}

# ── 7. SPARQL generation ──────────────────────────────────────────────────────

_sparql_rules = """SPARQL generation rules:
- Output only a plain SPARQL query. No markdown, no explanation, no code fences.
- CRITICAL: Use the EXACT prefix shown in the entity profile for each property — do not substitute dbo: for dbp: or vice versa. The in-context examples above may use different prefixes for the same property; ignore those choices. The entity profile in this conversation is authoritative.
- If the answer is a date, cast it: BIND(xsd:date(STR(?raw)) AS ?date)
- For a list of things, use a single ?uri column. Merge related answers with UNION, not multiple SELECT variables.
- DBpedia Categories (dbc:) in the context can be used via: ?uri dct:subject dbc:CategoryName
  Use categories as an alternative or fallback when structured dbo:/dbp: properties don't return results.
- UNION must be wrapped inside the WHERE clause: SELECT ?uri WHERE {{ {{ ... }} UNION {{ ... }} }}"""

sparql_agent_prompt = {
    "en": f"""You are a SPARQL query generation agent for DBpedia.
The conversation history contains the entity URIs, DBpedia entity profile, and categories needed to answer the question.

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
    "en": f"""Using the context provided above (entity URIs, entity profile, categories), generate a SPARQL query to answer the following question.

Question: {{question}}
{{suggestions_block}}
{_sparql_rules}"""
}

# ── 8. Result validation ──────────────────────────────────────────────────────

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
