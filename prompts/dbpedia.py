system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system that generates SPARQL queries over DBpedia.
    """
}


planner_prompt_dct = {
    "en": """For the given objective, create a step-by-step plan to write a SPARQL query over DBpedia.
Available tools: generate_context_tool, dbpedia_categories_tool.

Always output exactly 2 steps:
  Step 1: "Call generate_context_tool to gather entity URIs and the DBpedia shape. Do NOT write a SPARQL query yet — only return the context."
  Step 2: "Using the entity URIs and DBpedia shape from the previous step, construct and output the SPARQL query."

Only replace Step 1 with a dbpedia_categories_tool call if the question involves Wikipedia category membership.

Rules:
- generate_context_tool handles entity extraction, linking, and shape generation internally — call it once in Step 1.
- Do NOT resolve or link entities yourself — tools do that.
- Do NOT propose executing the query — a separate feedback step handles that automatically.
- Step 2 MUST output exactly ONE SPARQL query string with no extra text or markdown.

Objective: {objective}

Output format — valid JSON only:
{{"steps": ["step 1 description", "step 2 description"]}}"""
}

execute_step_prompt = {
    "en": """User question: {nlq}
Task: Construct the SPARQL query using the pre-computed entity URIs and DBpedia shape provided in the context
    """
}


last_task = {
    "en": """Make sure that the query is formatted correctly. No extra text. No markdown. Just plain SPARQL query.
Determine whether to output a URI (SELECT ?uri), number (COUNT), date, boolean (ASK), string (SELECT ?label).
- If the question is a yes/no question ("Are there any...", "Does X...", "Is there..."), use ASK WHERE { ... } instead of SELECT.
- If the expected answer is a date (e.g. founding year, birth date), cast to xsd:date using: BIND(xsd:date(STR(?raw)) AS ?date)
- If the question asks for a single list of things (people, places, etc.), use a single ?uri SELECT column. When two related entities (e.g. both parents, both father and mother) are the answer, merge them into one column with UNION rather than using multiple SELECT variables:
    CORRECT:   SELECT ?uri WHERE { { res:X dbo:parent ?uri } }
    INCORRECT: SELECT ?father ?mother WHERE { res:X dbo:father ?father ; dbo:mother ?mother }
- If the shape contained a property with controlled string values (e.g. [values: "X", "Y", ...]), use that as a direct mandatory filter — do NOT substitute YAGO or external class URIs.
    Example: ?uri a dbo:Film ; dbo:country res:Denmark  (NOT ?uri a yago:WikicatDanishFilms)
- If the query contains UNION patterns, make sure it is correctly formatted
    CORRECT:   SELECT ?uri WHERE { { ?s dbo:a ?uri } UNION { ?s dbo:b ?uri } }
    INCORRECT: SELECT ?uri WHERE { ?s dbo:a ?uri } UNION { ?s dbo:b ?uri }
    INCORRECT: SELECT ?uri WHERE { ?s dbo:a ?uri . UNION { ?s dbo:b ?uri } }

DON'T USE "SERVICE wikibase:label"
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
    - Try dbp: prefix instead of dbo: for the main property — many facts only exist in raw Wikipedia infobox properties (e.g. dbp:date, dbp:established, dbp:satellites, dbp:crewMembers)
    - Try reversing the subject and object of the main triple — some properties (e.g. dbo:goldMedalist, dbo:museum, dbo:commander, dbo:spokenIn) have the named entity as the object, not the subject
    - Add UNION patterns for alternative access paths: location (dbo:location / dbo:city / dbo:city+dbo:isPartOf), birthplace (dbo:birthPlace direct / via dbo:country), country (dbo:country / dbp:country)
    - If structured properties fail entirely, try: ?uri dct:subject dbc:RelevantCategoryName
    - For nickname/alias questions, try foaf:nick instead of dbp:nickname: res:X foaf:nick ?name
    - If the expected answer is a string literal (not a URI), try the dbp: property directly
    Review the shape generated earlier in the conversation and write a corrected query.
    If a property lists controlled values (e.g. [values: "X", "Y", ...]), use the appropriate value as a MANDATORY filter — do NOT make it OPTIONAL and do NOT replace it with a YAGO class.
    Example: ?uri a dbo:City ; dbo:isPartOf res:New_Jersey  (NOT ?uri a yago:WikicatCitiesInNewJersey)

    {last_task}
    """
  }

feedback_step_dict_short = {
    "en": """
    This is feedback to your generated SPARQL query produced by executing it on a triplestore.
    Please rework your query if neccessary.

    Initial question: {question}
    Your query: 
    {query}

    --- Start triplestore response ---
    {feedback}
    --- End triplestore response ---

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

Example: "Which city in France has the most museums?"
Result: "City, France"

Example: "Which people were born in Heraklion?"
Result: "Heraklion"

Example: "Show me all museums in London."
Result: "Museum, London"

Example: "Where was Nikola Tesla born?"
Result: "Nikola Tesla" NOT "Person"
"""
}

shape_check_prompt = {
    "en": """You are evaluating whether a DBpedia knowledge graph shape is useful for answering a question.

Question: "{nlq}"

Shape:
{shape}

A shape is USEFUL if:
- It is non-empty (has at least one property line)
- At least one property is plausibly relevant to answering the question
- The entity labels correspond to real DBpedia resources or classes (not empty placeholders)

A shape is NOT USEFUL if:
- It is empty or contains no property lines
- None of the extracted properties relate to what the question is asking
- The entity labels are clearly wrong or too generic for the question

Respond with ONLY a JSON object in this exact format (no markdown, no explanation outside the JSON):
{{"valid": true, "reason": "brief explanation of why the shape is useful"}}
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