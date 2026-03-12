import requests
import json
import time

api_url = "http://localhost:8000/api"
sparql_endpoint = "https://dbpedia.org/sparql"

INPUT_FILE = "qald_9_plus_train_dbpedia_en.json"
OUTPUT_FILE = "qald_9_sparql_results.json"
ERROR_FILE = "qald_9_f1_less_than_1.json"
LANGUAGE = "en"

# Load QALD questions
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    questions_data = data.get("questions", [])
else:
    questions_data = data

results = []
errors = []

# Running totals for micro scores
running_tp = 0
running_fp = 0
running_fn = 0


def extract_question(q, language="en"):
    if isinstance(q, str):
        return q

    if isinstance(q, list):
        for item in q:
            if item.get("language") == language:
                return item.get("string")

    if isinstance(q, dict):
        return q.get("string") or q.get("text")

    return str(q)


def extract_gold_sparql(item):
    """Extract the reference SPARQL query from QALD item."""
    query_block = item.get("query")

    if isinstance(query_block, dict):
        return query_block.get("sparql")

    if "sparql" in item:
        return item.get("sparql")

    return None


def execute_sparql(query):
    """Execute SPARQL query."""
    if not query:
        return None, "No query provided"

    try:
        response = requests.get(
            sparql_endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=60
        )

        if response.status_code == 200:
            return response.json(), None

        return None, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return None, str(e)


def extract_values(result_json):
    """Extract returned values from SPARQL JSON."""
    if not result_json:
        return set()

    values = set()
    bindings = result_json.get("results", {}).get("bindings", [])

    for row in bindings:
        for value_obj in row.values():
            value = value_obj.get("value")
            if value is not None:
                values.add(value)

    return values


def compute_metrics(predicted_set, gold_set):
    """Compute TP, FP, FN, precision, recall, F1."""
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    if len(predicted_set) == 0 and len(gold_set) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def compute_running_scores(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


for i, item in enumerate(questions_data):

    question_text = extract_question(item.get("question"), LANGUAGE)
    gold_query = extract_gold_sparql(item)

    print(f"\n[{i + 1}/{len(questions_data)}] {question_text}")

    generated_query = None
    generated_result = None
    generated_error = None

    gold_result = None
    gold_error = None

    generated_values = set()
    gold_values = set()

    try:
        response = requests.get(
            api_url,
            params={"question": question_text, "dataset": sparql_endpoint},
            timeout=60
        )

        if response.status_code == 200:
            api_data = response.json()
            generated_query = api_data.get("query")

            if generated_query:
                generated_result, generated_error = execute_sparql(generated_query)

            if gold_query:
                gold_result, gold_error = execute_sparql(gold_query)

            generated_values = extract_values(generated_result)
            gold_values = extract_values(gold_result)

        else:
            generated_error = response.text

    except Exception as e:
        generated_error = str(e)

    tp, fp, fn, precision, recall, f1 = compute_metrics(generated_values, gold_values)

    running_tp += tp
    running_fp += fp
    running_fn += fn

    running_precision, running_recall, running_f1 = compute_running_scores(
        running_tp, running_fp, running_fn
    )

    print(f"Question P/R/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"Running  P/R/F1: {running_precision:.4f} / {running_recall:.4f} / {running_f1:.4f}")

    result_entry = {
        "index": i,
        "question": question_text,

        "generated_query": generated_query,
        "generated_result": generated_result,
        "generated_execution_error": generated_error,

        "gold_query": gold_query,
        "gold_result": gold_result,
        "gold_execution_error": gold_error,

        "tp": tp,
        "fp": fp,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "running_tp": running_tp,
        "running_fp": running_fp,
        "running_fn": running_fn,

        "running_precision": running_precision,
        "running_recall": running_recall,
        "running_f1": running_f1,

        "generated_values": sorted(generated_values),
        "gold_values": sorted(gold_values),
        "only_in_generated": sorted(generated_values - gold_values),
        "only_in_gold": sorted(gold_values - generated_values)
    }

    results.append(result_entry)

    if f1 < 1.0:
        errors.append(result_entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)

    time.sleep(1)

final_precision, final_recall, final_f1 = compute_running_scores(
    running_tp, running_fp, running_fn
)

summary = {
    "total_questions": len(results),
    "final_tp": running_tp,
    "final_fp": running_fp,
    "final_fn": running_fn,
    "final_precision": final_precision,
    "final_recall": final_recall,
    "final_f1": final_f1,
    "questions_with_f1_less_than_1": len(errors)
}

print("\nFinished processing all questions")
print(json.dumps(summary, indent=4))