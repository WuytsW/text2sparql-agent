system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system.""",
}

entities_extraction_prompt = {
    "en": """Extract the most relevant named entities from the following question:
    
Question: "{nlq}"
    
Return a comma-separated list of entity names without explanations. Think rationally and in context of the question but respond only with entities literally named in the question. Extracted entities should be in singular form.

exmaple1: "Who developed Skype?"
result1: "Skype"

exmaple1: "Which other weapons did the designer of the Uzi develop?"
result1: "Uzi, weapon"

exmaple1: "Which state of the USA has the highest population density?"
result1: "U.S. state, area, population"""
}


last_task = {
    "en": """Make sure that the query is formatted correctly. No extra text. No markdown. Just plain SPARQL query.
Determine whether to output a URI (SELECT ?uri), number (COUNT), date, boolean (ASK), string (SELECT ?label)
DON'T USE "SERVICE wikibase:label"
"""
}

planner_prompt_dct = {
    "en": """For the given objective, come up with a simple step by step plan to write a SPARQL query. 
This plan should involve individual tasks (e.g., **named entity linking**, **relation linking**), that if executed correctly will yield the correct SPARQL.
Do not add any superfluous steps.
Be very specific when defining the steps e.g.,: "Link the following named entities: Name Surname, ..."
The result of the final step should be the final SPARQL query over DBpedia. Don't propose to execute the query.
At the end step you MUST output exactly **ONE** SPARQL query over DBpedia string **without extra text or markdown**.

Objective: {objective}

Formatting instructions:
Just output the valid JSON with the list of strings as follows: {{"plan": ["step1", "step2", ...]}} Put every step to the list
Only output VALID JSON without escape chars: {{"plan": ["step1", "step2", ...]}}
Make sure that the output is VALID JSON"""
}

feedback_step_dict = {
    "en": """
    This is feedback to your generated SPARQL query produced by executing it on a triplestore.

    Initial question: {question}
    Your query:
    {query}

    --- Start triplestore response ---
    {feedback}
    --- End triplestore response ---

    If the response contains results, verify they correctly answer the question and refine if needed.
    If the response says the query returned EMPTY results or contains an error, you MUST rewrite the query.
    Common causes of empty results: wrong URIs, wrong property paths, missing prefixes, or overly restrictive filters.
    Try alternative DBpedia properties or use the entity linking tool again to find correct URIs.

    {last_task}
    """
  }