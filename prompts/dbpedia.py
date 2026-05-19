system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system that generates SPARQL queries over DBpedia.
    """
}


shape_selection_prompt = {
    "en": """Given the folowing question: "{nlq}", and the following shape: "{shape}"
    select the most relevant properties and classes from the shape that are likely to be useful for answering the question.
    Return a comma-separated list of properties and classes from the shape that are relevant to the question. Only select properties and classes that are likely to be useful for answering the question. Do not select all properties, only the most relevant ones.
    Keep the formatting of the properties and classes as they are in the shape (Example:  dbo:deathPlace -> dbo:Place). 
    If the shape is empty, return an empty string."""
}

shape_selection_prompt_per_entity = {
    "en": """Given the question: "{nlq}", and the following properties for "{label}":

{shape}

Select only the most relevant properties needed to answer the question. (If you are not sure about selecting a property, it's better to include it than to miss it. Altough, try to avoid including too many irrelevant properties as it may lead to slow query execution or empty results.)
Return a comma-separated list in the exact format shown (e.g. dbo:capital -> dbo:City).
If none are relevant, return an empty string."""
}

category_selection_prompt = {
    "en": """Given the question: "{nlq}", and the following DBpedia category URIs found:

{categories}

Select only the category URIs that are most likely relevant to answering the question.
Return a comma-separated list of the full URIs (e.g. http://dbpedia.org/resource/Category:Foo, http://dbpedia.org/resource/Category:Bar).
If none are relevant, return an empty string."""
}

class_instances_prompt = {
    "en": """Determine if the term "{label}" refers to a specific named entity or a general class/type of things.

A NAMED ENTITY is a unique, specific thing: a particular person, place, organization, creative work, etc.
Examples: "Michael Jackson", "Eiffel Tower", "Apple Inc.", "Uzi"

A CLASS/TYPE is a general category that many things can belong to.
Examples: "Animal", "Country", "Musical Artist", "Film", "Weapon", "City"

If "{label}" is a NAMED ENTITY, respond with exactly:
ENTITY

If "{label}" is a CLASS/TYPE, respond with exactly:
CLASS

Examples:
"Michael Jackson" → ENTITY
"Animal" → CLASS"""
}


context_check_prompt = {
    "en": """You are evaluating whether the context extracted from DBpedia contains enough relevant information to answer the question with a SPARQL query.

Question: "{nlq}"

Extracted entities: {entities}

Entity URIs (linked DBpedia resources):
{entity_uris}

DBpedia categories:
{categories}

Shape (available properties per entity/class):
{shape}

Evaluate the context AS A WHOLE. It is USEFUL if any part of it — the shape, the entity URIs, or the categories — provides information that could plausibly help construct a SPARQL query to answer the question. Individual components may be empty or sparse; that is fine as long as the overall context is helpful.

It is NOT USEFUL only if the context as a whole is clearly off-topic, completely empty, or the extracted entities are so wrong that no part of the context relates to the question.

Respond with ONLY a JSON object (no markdown):
{{"valid": true, "reason": "brief explanation"}}
or
{{"valid": false, "reason": "what is wrong and what entity labels might work better"}}"""
}

entities_extraction_prompt = {
"en": """Extract the DBpedia entity and class labels needed to answer the following question with a SPARQL query.

Question: "{nlq}"

Rules:
- Return ONLY a comma-separated list of labels, no explanations.
- Use singular form and capitalise as a DBpedia class would be (e.g. "Novelist" not "novelists").
- Descriptive adjectives like "largest", "extinct", "female" are filters, NOT entities — do not include them.
- Titles of creative works (books, films, games, albums, TV series, etc.) are single named entities regardless of how many words they contain. Treat the full title as one item (e.g. "The Pillars of the Earth", NOT "pillar", "earth").
- Full person names must be kept together (e.g. "Abraham Lincoln" NOT "Abraham" or "Lincoln").
- If a named entity is referred to only by a partial name, expand it to the most complete, commonly recognized form (e.g. "Napoleon" → "Napoleon Bonaparte").
- Only include a class label if it appears as an explicit noun category in the question (e.g. "novelist", "weapon", "state"). Never extract "Person" — it is too generic.

Example: "Who developed Skype?"
Result: Skype

Example: "Which other weapons did the designer of the Uzi develop?"
Result: Uzi, Weapon

Example: "Which city in France has the most museums?"
Result: City, France

Example: "Who wrote the book The Pillars of the Earth?"
Result: The Pillars of the Earth

Example: "Where was Nikola Tesla born?"
Result: Nikola Tesla

Example: "Show me all museums in London."
Result: Museum, London
"""
}

sparql_planner_prompt = {
    "en": """You are a SPARQL query generation orchestrator for DBpedia.
The context (entity URIs, DBpedia shape, categories) has already been provided in the conversation history.

Follow this exact workflow:
1. Call generate_sparql with no suggestions on the first attempt.
2. Call execute_sparql with the returned query.
3. Call check_result with the query and execution results.
4. If check_result returns ok=true, output the final SPARQL query and stop.
5. If check_result returns ok=false and you have made fewer than 3 attempts, call generate_sparql again passing the suggestions, then repeat steps 2-3.
6. After 3 attempts, output the best query you have, even if imperfect.

Never skip execute_sparql after generating a query. Output only the final plain SPARQL query string, no markdown, no explanation."""
}

generation_prompt = {
    "en": """Using the context provided above (entity URIs, shape, categories), generate a SPARQL query to answer the following question.

Question: {question}
{suggestions_block}
Rules:
- Output only a plain SPARQL query. No markdown, no explanation, no code fences.
- Use ASK for yes/no questions ("Are there any...", "Is X the Y of Z?"). Use SELECT for all others.
- If the answer is a date, cast it: BIND(xsd:date(STR(?raw)) AS ?date)
- For a list of things, use a single ?uri column. Merge related answers with UNION, not multiple SELECT variables.
- Use dbo: properties from the shape. Only use dbp: if the shape explicitly listed that property with a dbp: prefix.
- DBpedia Categories (dbc:) in the context can be used via: ?uri dct:subject dbc:CategoryName
  Use categories as an alternative or fallback when structured dbo:/dbp: properties don't return results, or when the question is about membership in a category.
- UNION must be wrapped inside the WHERE clause: SELECT ?uri WHERE {{ {{ ... }} UNION {{ ... }} }}
- Do NOT use SERVICE wikibase:label"""
}

check_result_prompt = {
    "en": """Evaluate whether this SPARQL query correctly answers the question.

Question: {question}
Query:
{query}

Execution result:
{execution_result}

If results are non-empty (and answer the question (Do not be strict here, most often if the result is non-empty it is acceptable)) respond with exactly (JSON only, no markdown):
{{"ok": true}}

If results are empty, an error, or do not answer the question respond with (JSON only, no markdown):
{{"ok": false, "suggestions": "<concrete fix suggestions based on the shape and context in the conversation>"}}"""
}

sparql_agent_prompt = {
    "en": """You are a SPARQL query generation agent for DBpedia.
The conversation history contains the entity URIs, DBpedia shape, and categories needed to answer the question.

Workflow:
1. Generate a SPARQL query for the question using the context provided.
2. Call execute_sparql with that query.
3. If results are non-empty and answer the question, output the final query and stop.
4. If results are empty, an error, or clearly wrong, refine the query and call execute_sparql again.
5. Repeat up to 3 times. After 3 attempts, output the best query you have.

Generation rules:
- Output only a plain SPARQL query. No markdown, no explanation, no code fences.
- Use ASK for yes/no questions ("Are there any...", "Is X the Y of Z?"). Use SELECT for all others.
- If the answer is a date, cast it: BIND(xsd:date(STR(?raw)) AS ?date)
- For a list of things, use a single ?uri column. Merge related answers with UNION, not multiple SELECT variables.
- Use dbo: properties from the shape. Only use dbp: if the shape explicitly listed that property with a dbp: prefix.
- DBpedia Categories (dbc:) in the context can be used via: ?uri dct:subject dbc:CategoryName
  Use categories as an alternative or fallback when structured dbo:/dbp: properties don't return results.
- UNION must be wrapped inside the WHERE clause: SELECT ?uri WHERE {{ {{ ... }} UNION {{ ... }} }}
- Do NOT use SERVICE wikibase:label

Evaluation: Accept a query if results are non-empty and answer the question. Do not be strict — a non-empty result is usually acceptable. If results are empty or an error, improve the query based on the shape and context.

Output only the final plain SPARQL query string. No markdown, no code fences, no explanation."""
}
