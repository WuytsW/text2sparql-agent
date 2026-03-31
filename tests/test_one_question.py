import requests

API_URL = "http://localhost:8000/api"
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


def ask_question(question: str):
    # Step 1: get generated SPARQL query
    response = requests.get(
        API_URL,
        params={"question": question, "dataset": SPARQL_ENDPOINT},
        timeout=60,
    )

    if response.status_code != 200:
        print("API error:", response.text)
        return

    data = response.json()
    query = data.get("query")

    if not query:
        print("No query returned")
        return

    print("\nGenerated SPARQL query:\n")
    print(query)

    # Step 2: execute SPARQL
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=60,
    )

    if response.status_code != 200:
        print("SPARQL error:", response.text)
        return

    result = response.json()

    print("\nSPARQL result:\n")
    print(result)


if __name__ == "__main__":
    question = "Which people were born in Brussels?"
    ask_question(question)