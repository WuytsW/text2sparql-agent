from fastapi import FastAPI, HTTPException
from typing import List
from services.llm_agent_dbpedia import LLMAgentDBpedia
from services.llm_agent_wikidata import LLMAgentWikidata


__version__ = "0.1.0"

app = FastAPI(
    title="KGQAgent Text2SPARQL API",
    description="API for converting natural language questions to SPARQL queries using LLMs.",
    version=__version__
)

KNOWN_DATASETS: List[str] = [
    "https://dbpedia.org/sparql",
    "https://query.wikidata.org/sparql",
]

dbpedia_agent = LLMAgentDBpedia()
wikidata_agent = LLMAgentWikidata()

@app.get("/api")
async def get_answer(
    question: str,
    dataset: str,
    model_name: str = "openai/gpt-4o-mini",
    log_calls: bool = True,
    temperature: float = 0,
    use_icl: bool = True,
    use_eat: bool = True,
    use_context: bool = True,
):
    """
    Process a natural language question and convert it to SPARQL query for the specified dataset.

    Args:
        question: The natural language question to process
        dataset: The dataset URL to query against
        model_name: OpenRouter model identifier (default: openai/gpt-4o-mini)
        log_calls: If True, log LLM calls
        temperature: The temperature for LLM sampling
        use_icl: If True, run the in-context learning step (default: True)
        use_eat: If True, run the expected answer type step (default: True)
        use_context: If True, run the entity context step (default: True)

    Returns:
        JSON with the dataset, original question, generated SPARQL query, and LLM usage stats
    """
    if dataset not in KNOWN_DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset. Please use one of the known datasets.")

    if "dbpedia" in dataset:
        result = dbpedia_agent.generate_sparql(question, model_name=model_name, log_calls=log_calls, temperature=temperature, use_icl=use_icl, use_eat=use_eat, use_context=use_context)
    elif "wikidata" in dataset:
        result = wikidata_agent.generate_sparql(question, model_name=model_name, log_calls=log_calls, use_icl=use_icl, use_eat=use_eat, use_context=use_context)
    else:
        raise HTTPException(status_code=404, detail="Unknown dataset. Please use one of the known datasets.")
          
    return {
        "dataset": dataset,
        "question": question,
        "model_name": model_name,
        "translated_question": result["translated_question"],
        "query": result["query"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "requests": result["requests"],
        "step_times": result.get("step_times", []),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
