import json

INPUT_FILE = "qald_9_sparql_results.json"
SUMMARY_FILE = "qald_9_macro_f1_summary.json"
LOG_FILE = "qald_9_macro_f1_log.txt"


def to_value_set(value_list):
    """Convert stored value list to a set safely."""
    if not value_list:
        return set()
    return set(value_list)


def compute_metrics(predicted_set, gold_set):
    """Compute TP, FP, FN, precision, recall, F1 for one question."""
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

included = []
skipped = []
log_lines = []

log_lines.append(f"Reading: {INPUT_FILE}")
log_lines.append("")

for item in results:
    index = item.get("index")
    question = item.get("question", "")

    predicted_values = to_value_set(item.get("generated_values", []))
    gold_values = to_value_set(item.get("gold_values", []))

    # Skip if gold result is empty
    if len(gold_values) == 0:
        skipped_entry = {
            "index": index,
            "question": question,
            "reason": "gold_values is empty"
        }
        skipped.append(skipped_entry)
        log_lines.append(
            f"[SKIP] index={index} | question={question} | reason=gold_values is empty"
        )
        continue

    tp, fp, fn, precision, recall, f1 = compute_metrics(predicted_values, gold_values)

    entry = {
        "index": index,
        "question": question,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(predicted_values),
        "gold_count": len(gold_values)
    }
    included.append(entry)

    log_lines.append(
        f"[USE ] index={index} | question={question} | "
        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} | "
        f"pred={len(predicted_values)} gold={len(gold_values)}"
    )

# Macro metrics: every included question has equal weight
if included:
    macro_precision = sum(x["precision"] for x in included) / len(included)
    macro_recall = sum(x["recall"] for x in included) / len(included)
    macro_f1 = sum(x["f1"] for x in included) / len(included)
else:
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

summary = {
    "input_file": INPUT_FILE,
    "total_questions_in_file": len(results),
    "included_questions": len(included),
    "skipped_questions": len(skipped),
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "included": included,
    "skipped": skipped
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
    f.write("\n\n===== SUMMARY =====\n")
    f.write(f"Total questions in file: {len(results)}\n")
    f.write(f"Included questions: {len(included)}\n")
    f.write(f"Skipped questions: {len(skipped)}\n")
    f.write(f"Macro Precision: {macro_precision:.6f}\n")
    f.write(f"Macro Recall:    {macro_recall:.6f}\n")
    f.write(f"Macro F1:        {macro_f1:.6f}\n")

print("Finished.")
print(f"Total questions in file: {len(results)}")
print(f"Included questions: {len(included)}")
print(f"Skipped questions: {len(skipped)}")
print(f"Macro Precision: {macro_precision:.6f}")
print(f"Macro Recall:    {macro_recall:.6f}")
print(f"Macro F1:        {macro_f1:.6f}")
print(f"Summary saved to: {SUMMARY_FILE}")
print(f"Log saved to: {LOG_FILE}")