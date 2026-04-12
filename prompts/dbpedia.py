system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system that generates SPARQL queries over DBpedia.

You MUST call extract_entities_tool, dbpedia_el, and generate_shape_tool EXACTLY ONCE per conversation — during the dedicated shape generation step.
For ALL other steps (including SPARQL construction), you MUST NOT call these tools.
If you are about to call these tools and a shape is already present in the chat history, STOP — use the shape from the chat history instead.

Shape generation order (first step only):
1. Call extract_entities_tool(nlq) to extract the relevant DBpedia entity/class labels from the question.
2. Call dbpedia_el(named_entities) with only the NAMED ENTITIES from the labels returned in step 1 (skip general classes like "Film", "City", "Person").
   Use the URIs returned by dbpedia_el directly in your SPARQL query — do NOT guess res: URIs for named entities.
3. Call generate_shape_tool(nlq, entity_labels) with the full label list returned by extract_entities_tool.
Never call generate_shape_tool with entity labels you invent yourself.

When using the generated shape to construct SPARQL:
- If a shape property lists [values: ...], these are the only valid values for that property. If the question implies a specific value, use it as a MANDATORY triple pattern — NOT as OPTIONAL.
- Always prefer dbo: property-based filters over YAGO classes or Wikipedia category URIs.

Example (preferred):
Question: "Give me all Danish films."
Shape: Film: dbo:country -> dbo:Country
CORRECT:   ?uri a dbo:Film ; dbo:country res:Denmark .
INCORRECT: ?uri a <http://dbpedia.org/class/yago/WikicatDanishFilms> .""",
}


planner_prompt_dct = {
    "en": """For the given objective, come up with a concise step by step plan to write a SPARQL query.
Keep the plan SHORT — exactly 2 steps for most questions:
  Step 1: "Generate the shape" (this is the ONLY step that calls tools — extract entities, link named entities via dbpedia_el, then generate the shape).
  Step 2: "Construct the SPARQL query using the shape from step 1 and the URIs from dbpedia_el" (no tool calls — use what was already generated).
Only add a third step if the question is genuinely complex (e.g. involves multiple unrelated entities or aggregations).
Do NOT split entity extraction, entity linking, and shape generation into separate steps — all three tool calls happen together in step 1.
Do NOT resolve, identify, or link any entities or properties yourself — that will be done by tools in the execution step.
Do not add any superfluous steps.
The result of the final step should be the final SPARQL query over DBpedia. Don't propose to execute the query.
At the end step you MUST output exactly **ONE** SPARQL query over DBpedia string **without extra text or markdown**.

Objective: {objective}

Formatting instructions:
Just output the valid JSON with the list of strings as follows: {{"plan": ["step1", "step2", ...]}} Put every step to the list
Only output VALID JSON without escape chars: {{"plan": ["step1", "step2", ...]}}
Make sure that the output is VALID JSON"""
}


last_task = {
    "en": """Make sure that the query is formatted correctly. No extra text. No markdown. Just plain SPARQL query.
Determine whether to output a URI (SELECT ?uri), number (COUNT), date, boolean (ASK), string (SELECT ?label)
DON'T USE "SERVICE wikibase:label"
If the shape contained a property with controlled string values (e.g. [values: "X", "Y", ...]), use that as a direct mandatory filter — do NOT substitute YAGO or external class URIs.
Example: ?uri a dbo:Film ; dbo:country res:Denmark  (NOT ?uri a yago:WikicatDanishFilms)
"""
}

feedback_step_dict = {
    "en": """
    This is feedback to your generated SPARQL query produced by executing it on a triplestore.
    Please rework your query if neccessary.

    Initial question: {question}
    Your query:
    {query}

    --- Start triplestore response ---
    {feedback}
    --- End triplestore response ---

    If the triplestore response contains results, the query is CORRECT — return it UNCHANGED.
    - Do NOT substitute concrete values from the results back into the query (e.g. do NOT replace ?uri with a specific resource URI).
    - Do NOT modify the WHERE clause, variable names, or triple patterns in any way.e 
    
    
    If the results are empty or an error occurred, the query is WRONG. You MUST rewrite it.
    Common fixes to try:
    - Replace resource URIs used as rdf:type with dbo: ontology classes
    - Remove overly restrictive type constraints that may not exist in the triplestore
    - Use dbo: properties from the shape instead of guessing property paths
    - Check whether the shape uses a different predicate than the one in your query
    Review the shape generated earlier in the conversation and write a corrected query.
    If a property lists controlled values (e.g. [values: "X", "Y", ...]), use the appropriate value as a MANDATORY filter — do NOT make it OPTIONAL and do NOT replace it with a YAGO class.
    Example: ?uri a dbo:City ; dbo:isPartOf res:New_Jersey  (NOT ?uri a yago:WikicatCitiesInNewJersey)

    {last_task}
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

entities_extraction_prompt_old = {
    "en": """Extract the DBpedia entity and class labels needed to answer the following question with a SPARQL query.

Question: "{nlq}"

Rules:
- Use singular form and capitalise as a DBpedia class would be (e.g. "City" not "cities").
- Descriptive adjectives like "extinct", "largest", "female" are filters, NOT entities — do not extract them.
- Include a specific named entity only if the question refers to one (e.g. "Uzi", "Skype").
- If the question contains names where both name and surname are mentioned, extract the full name (e.g. "Michael Jackson" NOT "Michael" or "Jackson").
- If a named entity is referred to by only a partial name (surname, nickname, or single historical name), expand it (return only full names) to the most complete, commonly recognized full name (e.g. "Napoleon" → "Napoleon Bonaparte").
- Only extract a class label if it appears as an explicit noun category in the question (e.g. "movies", "museums", "state"). Never extract "Person" — it is too generic to be useful. Use specific subclasses only if the question explicitly names them (e.g. "Actor", "Politician", "Writer").
- Return ONLY a comma-separated list of labels, no explanations.

Example: "Who developed Skype?"
Result: "Skype"

Example: "Which other weapons did the designer of the Uzi develop?"
Result: "Uzi, Weapon"

Example: "Which state of the USA has the highest population density?"
Result: "U.S. state"

Example: "Which people were born in Heraklion?"
Result: "Heraklion"

Example: "Show me all museums in London."
Result: "Museum, London"

Example: "Where did Abraham Lincoln die?"
Result: "Abraham Lincoln" NOT "Person"
"""
}

entities_extraction_prompt = {
"en": """    
Extract the most relevant named entities from the following question:
    
Question: "{nlq}"
    
Return a comma-separated list of entity names without explanations. Think rationally and in context of the question but respond only with entities literally named in the question. Extracted entities should be in singular form.

exmaple1: "Who developed Skype?"
result1: "Skype"

exmaple1: "Which other weapons did the designer of the Uzi develop?"
result1: "Uzi, weapon"

exmaple1: "Which state of the USA has the highest population density?"
result1: "U.S. state, area, population"
"""
}