#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


DEFAULT_API_URL = "http://localhost:8000/api"
DEFAULT_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def flush_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def load_questions(input_file: Path) -> List[Dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("questions", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {input_file}")


def extract_question(question_field: Any, lang: str = "en") -> str:
    if isinstance(question_field, str):
        return question_field.strip()

    if isinstance(question_field, list):
        for item in question_field:
            if item.get("language") == lang and item.get("string"):
                return item["string"].strip()
        for item in question_field:
            if item.get("string"):
                return item["string"].strip()

    if isinstance(question_field, dict):
        return (question_field.get("string") or question_field.get("text") or "").strip()

    return ""


def extract_gold_sparql(item: Dict[str, Any]) -> Optional[str]:
    query_block = item.get("query")

    if isinstance(query_block, dict):
        sparql = query_block.get("sparql")
        return sparql.strip() if sparql else None

    if "sparql" in item and item["sparql"]:
        return str(item["sparql"]).strip()

    return None


def call_generation_api(api_url: str, question_text: str, dataset_url: str, timeout: int = 120) -> Tuple[Optional[str], Optional[str]]:
    try:
        response = requests.get(
            api_url,
            params={"question": question_text, "dataset": dataset_url},
            timeout=timeout,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}"

        api_data = response.json()
        generated_query = api_data.get("query")
        if not generated_query:
            return None, "API returned empty query"

        return str(generated_query).strip(), None

    except Exception as e:
        return None, str(e)


def execute_sparql(query: Optional[str], endpoint: str, timeout: int = 60) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not query:
        return None, "No query provided"

    try:
        response = requests.get(
            endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )

        if response.status_code == 200:
            return response.json(), None

        return None, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return None, str(e)


def normalize_binding_value(value_obj: Dict[str, Any]) -> str:
    vtype = value_obj.get("type", "")
    value = str(value_obj.get("value", "")).strip()

    if vtype == "uri":
        return value

    if vtype in {"literal", "typed-literal"}:
        datatype = value_obj.get("datatype")
        lang = value_obj.get("xml:lang") or value_obj.get("lang")
        if datatype:
            return f'"{value}"^^<{datatype}>'
        if lang:
            return f'"{value}"@{lang}'
        return f'"{value}"'

    if vtype == "bnode":
        return f"_:{value}"

    return value


def extract_values(result_json: Optional[Dict[str, Any]]) -> Set[str]:
    if not result_json:
        return set()

    if "boolean" in result_json:
        return {str(bool(result_json["boolean"])).lower()}

    values: Set[str] = set()
    bindings = result_json.get("results", {}).get("bindings", [])

    for row in bindings:
        normalized_row = []
        for var_name in sorted(row.keys()):
            normalized_row.append(f"{var_name}={normalize_binding_value(row[var_name])}")
        if normalized_row:
            values.add(" | ".join(normalized_row))

    return values


def compute_metrics(predicted_set: Set[str], gold_set: Set[str]) -> Tuple[int, int, int, float, float, float]:
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    if not predicted_set and not gold_set:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def compute_micro_scores(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_macro_scores(precisions: List[float], recalls: List[float], f1s: List[float]) -> Tuple[float, float, float]:
    n = len(f1s)
    if n == 0:
        return 0.0, 0.0, 0.0
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def make_summary(
    total_questions: int,
    processed_questions: int,
    running_tp: int,
    running_fp: int,
    running_fn: int,
    per_question_precisions: List[float],
    per_question_recalls: List[float],
    per_question_f1s: List[float],
    errors_count: int,
    started_at: float,
) -> Dict[str, Any]:
    micro_precision, micro_recall, micro_f1 = compute_micro_scores(running_tp, running_fp, running_fn)
    macro_precision, macro_recall, macro_f1 = compute_macro_scores(
        per_question_precisions, per_question_recalls, per_question_f1s
    )

    return {
        "total_questions": total_questions,
        "processed_questions": processed_questions,
        "remaining_questions": total_questions - processed_questions,
        "final_tp_so_far": running_tp,
        "final_fp_so_far": running_fp,
        "final_fn_so_far": running_fn,
        "micro_precision_so_far": micro_precision,
        "micro_recall_so_far": micro_recall,
        "micro_f1_so_far": micro_f1,
        "macro_precision_so_far": macro_precision,
        "macro_recall_so_far": macro_recall,
        "macro_f1_so_far": macro_f1,
        "questions_with_f1_less_than_1": errors_count,
        "elapsed_seconds": time.time() - started_at,
        "done": processed_questions == total_questions,
    }


def evaluate(
    input_file: Path,
    output_dir: Path,
    api_url: str,
    sparql_endpoint: str,
    lang: str,
    limit: Optional[int],
    sleep_seconds: float,
) -> None:
    questions_data = load_questions(input_file)
    if limit is not None:
        questions_data = questions_data[:limit]

    output_file = output_dir / "results.json"
    error_file = output_dir / "errors.json"
    summary_file = output_dir / "summary.json"

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    running_tp = 0
    running_fp = 0
    running_fn = 0

    per_question_precisions: List[float] = []
    per_question_recalls: List[float] = []
    per_question_f1s: List[float] = []

    started_at = time.time()

    flush_json(output_file, results)
    flush_json(error_file, errors)
    flush_json(
        summary_file,
        make_summary(
            total_questions=len(questions_data),
            processed_questions=0,
            running_tp=0,
            running_fp=0,
            running_fn=0,
            per_question_precisions=[],
            per_question_recalls=[],
            per_question_f1s=[],
            errors_count=0,
            started_at=started_at,
        ),
    )

    for i, item in enumerate(questions_data, start=1):
        question_text = extract_question(item.get("question"), lang=lang)
        gold_query = extract_gold_sparql(item)
        qid = item.get("id", i)

        logging.info("(%d/%d) QID=%s | %s", i, len(questions_data), qid, question_text)

        generated_query = None
        generated_result = None
        generated_error = None

        gold_result = None
        gold_error = None

        generated_values: Set[str] = set()
        gold_values: Set[str] = set()

        try:
            generated_query, api_error = call_generation_api(
                api_url=api_url,
                question_text=question_text,
                dataset_url=sparql_endpoint,
            )
            generated_error = api_error

            if generated_query:
                generated_result, exec_error = execute_sparql(generated_query, sparql_endpoint)
                if exec_error:
                    generated_error = exec_error

            if gold_query:
                gold_result, gold_error = execute_sparql(gold_query, sparql_endpoint)

            generated_values = extract_values(generated_result)
            gold_values = extract_values(gold_result)

        except Exception as e:
            generated_error = str(e)

        tp, fp, fn, precision, recall, f1 = compute_metrics(generated_values, gold_values)

        running_tp += tp
        running_fp += fp
        running_fn += fn

        per_question_precisions.append(precision)
        per_question_recalls.append(recall)
        per_question_f1s.append(f1)

        running_micro_precision, running_micro_recall, running_micro_f1 = compute_micro_scores(
            running_tp, running_fp, running_fn
        )
        running_macro_precision, running_macro_recall, running_macro_f1 = compute_macro_scores(
            per_question_precisions, per_question_recalls, per_question_f1s
        )

        logging.info(
            "QID=%s done | P=%.4f R=%.4f F1=%.4f | gold=%d pred=%d",
            qid, precision, recall, f1, len(gold_values), len(generated_values)
        )

        result_entry = {
            "index": i,
            "id": qid,
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

            "running_micro_precision": running_micro_precision,
            "running_micro_recall": running_micro_recall,
            "running_micro_f1": running_micro_f1,

            "running_macro_precision": running_macro_precision,
            "running_macro_recall": running_macro_recall,
            "running_macro_f1": running_macro_f1,

            "generated_values": sorted(generated_values),
            "gold_values": sorted(gold_values),
            "only_in_generated": sorted(generated_values - gold_values),
            "only_in_gold": sorted(gold_values - generated_values),
        }

        results.append(result_entry)
        flush_json(output_file, results)

        if f1 < 1.0:
            errors.append(result_entry)
            flush_json(error_file, errors)

        summary = make_summary(
            total_questions=len(questions_data),
            processed_questions=i,
            running_tp=running_tp,
            running_fp=running_fp,
            running_fn=running_fn,
            per_question_precisions=per_question_precisions,
            per_question_recalls=per_question_recalls,
            per_question_f1s=per_question_f1s,
            errors_count=len(errors),
            started_at=started_at,
        )
        flush_json(summary_file, summary)

        print(f"\n[{i}/{len(questions_data)}] {question_text}")
        print(f"Question P/R/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print(f"Running micro P/R/F1: {running_micro_precision:.4f} / {running_micro_recall:.4f} / {running_micro_f1:.4f}")
        print(f"Running macro P/R/F1: {running_macro_precision:.4f} / {running_macro_recall:.4f} / {running_macro_f1:.4f}")
        print(f"Results written live to: {output_file}")
        print(f"Summary written live to: {summary_file}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    final_summary = make_summary(
        total_questions=len(questions_data),
        processed_questions=len(questions_data),
        running_tp=running_tp,
        running_fp=running_fp,
        running_fn=running_fn,
        per_question_precisions=per_question_precisions,
        per_question_recalls=per_question_recalls,
        per_question_f1s=per_question_f1s,
        errors_count=len(errors),
        started_at=started_at,
    )

    print("\nFinished processing all questions")
    print(json.dumps(final_summary, indent=4, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live QALD evaluator for DBpedia")
    parser.add_argument("--input-file", type=Path, required=True, help="Path to QALD JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("./qald_eval_live"), help="Directory for live output files")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Your FastAPI /api endpoint")
    parser.add_argument("--sparql-endpoint", type=str, default=DEFAULT_SPARQL_ENDPOINT, help="SPARQL endpoint to execute against")
    parser.add_argument("--lang", type=str, default="en", help="Question language")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between questions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    evaluate(
        input_file=args.input_file,
        output_dir=args.output_dir,
        api_url=args.api_url,
        sparql_endpoint=args.sparql_endpoint,
        lang=args.lang,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()