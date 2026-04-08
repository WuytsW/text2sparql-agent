from fastapi import FastAPI, HTTPException
from typing import List
# from services.llm_agent import LLMAgent
from services.llm_agent_dbpedia import LLMAgentDBpedia
from services.llm_agent_corporate import LLMAgentCorporate


__version__ = "0.1.0"

app = FastAPI(
    title="KGQAgent Text2SPARQL API",
    description="API for converting natural language questions to SPARQL queries using LLMs.",
    version=__version__
)

KNOWN_DATASETS: List[str] = [
    "https://dbpedia.org/sparql",
    "https://text2sparql.aksw.org/2025/corporate/"
]

dbpedia_agent = LLMAgentDBpedia()
#corporate_agent = LLMAgentCorporate()

@app.get("/api")
async def get_answer(
    question: str,
    dataset: str,
    model_name: str = "openai/gpt-4o-mini",
    compact: bool = False,
):
    """
    Process a natural language question and convert it to SPARQL query for the specified dataset.

    Args:
        question: The natural language question to process
        dataset: The dataset URL to query against
        model_name: OpenRouter model identifier (default: openai/gpt-4o-mini)
        compact: If True, execute all plan steps in a single agent call (default: False)

    Returns:
        JSON with the dataset, original question, generated SPARQL query, and LLM usage stats
    """
    if dataset not in KNOWN_DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset. Please use one of the known datasets.")

    if "dbpedia" in dataset:
        result = dbpedia_agent.generate_sparql(question, model_name=model_name, compact=compact)
    #elif "corporate" in dataset:
       # result = corporate_agent.generate_sparql(question, model_name=model_name, compact=compact)
    else:
        raise HTTPException(status_code=404, detail="Unknown dataset. Please use one of the known datasets.")

    return {
        "dataset": dataset,
        "question": question,
        "model_name": model_name,
        "compact": compact,
        "query": result["query"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "requests": result["requests"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
