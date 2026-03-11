import requests
import json
import time
import logging
import http.client

http.client.HTTPConnection.debuglevel = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

api_url = "http://localhost:8000/api"
sparql_endpoint = "https://dbpedia.org/sparql"

questions = [
    "List all boardgames by GMT."
]

results = []

for q in questions:

    print("Processing:", q)

    params = {
        "question": q,
        "dataset": sparql_endpoint
    }

    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        query = data.get("query")

        print("Generated query received")

        query_result = None
        execution_error = None

        if query:
            try:
                sparql_response = requests.get(
                    sparql_endpoint,
                    params={"query": query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json"}
                )

                if sparql_response.status_code == 200:
                    query_result = sparql_response.json()
                    print("Query executed successfully")
                else:
                    execution_error = sparql_response.text
                    print("Query execution failed")

            except Exception as e:
                execution_error = str(e)

        results.append({
            "question": q,
            "query": query,
            "result": query_result,
            "execution_error": execution_error
        })

    else:
        results.append({
            "question": q,
            "error": response.text
        })

    time.sleep(1)

with open("sparql_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to sparql_results.json")