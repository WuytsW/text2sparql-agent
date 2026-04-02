system_prompt = {
    "en": """You are an intelligent Knowledge Graph-based Question Answering system.""",
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
    Your SPARQL query was executed against a triplestore but the results do not correctly answer the question.
    Review ALL previous attempts below and generate an improved query that avoids the same mistakes.

    Initial question: {question}

    --- Attempt history ---
{history}
    --- End attempt history ---

    Latest attempt:
    Query: {query}
    Triplestore results: {feedback}
    Validation: INVALID

    {last_task}
    """
  }

feedback_validation_prompt = {
    "en": """You are a SPARQL query validator. A SPARQL query was executed against a triplestore and produced results.
Determine whether the triplestore results actually answer the original question.

Original question: {question}
SPARQL query: {query}
Triplestore results: {feedback}

Reply with ONLY "YES" if the results answer the question, or "NO" if they do not (e.g. empty results, wrong data type, unrelated results, or an error occurred).
Answer:"""
}