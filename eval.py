"""Evaluator: runs an extractor on receipts and measures accuracy per field."""
import re
import statistics
import sys
import time
from difflib import SequenceMatcher

from docagent.extract import extract
from docagent.line_extract import extract_with_classifier
from docagent.loader import list_ids, load_ground_truth, load_ocr_text

FIELDS = ["company", "date", "address", "total"]
N_RECEIPTS = 50
FUZZY_THRESHOLD = 0.9


def run_llm(text: str) -> dict | None:
    receipt = extract(text)
    return receipt.model_dump() if receipt else None


EXTRACTORS = {"llm": run_llm, "classifier": extract_with_classifier}


def normalize_text(value) -> str:
    return " ".join(str(value).upper().split())


def normalize_total(value) -> float | None:
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None


def compare(field: str, predicted, truth) -> tuple[bool, bool]:
    """Return (exact_match, fuzzy_match) for one field."""
    if field == "total":
        p, t = normalize_total(predicted), normalize_total(truth)
        ok = p is not None and t is not None and abs(p - t) < 0.005
        return ok, ok
    p, t = normalize_text(predicted), normalize_text(truth)
    return p == t, SequenceMatcher(None, p, t).ratio() >= FUZZY_THRESHOLD


def main(output_path: str, extractor_name: str, split: str) -> None:
    run = EXTRACTORS[extractor_name]
    ids = list_ids(split)[:N_RECEIPTS]
    scored = {field: 0 for field in FIELDS}
    exact = {field: 0 for field in FIELDS}
    fuzzy = {field: 0 for field in FIELDS}
    latencies, mistakes, failures = [], [], 0

    for i, rid in enumerate(ids, start=1):
        truth = load_ground_truth(rid, split)
        start = time.perf_counter()
        result = run(load_ocr_text(rid, split))
        latencies.append((time.perf_counter() - start) * 1000)
        if result is None:
            failures += 1

        for field in FIELDS:
            if field not in truth:
                continue
            scored[field] += 1
            predicted = result.get(field, "") if result else ""
            is_exact, is_fuzzy = compare(field, predicted, truth[field])
            exact[field] += is_exact
            fuzzy[field] += is_fuzzy
            if not is_fuzzy:
                mistakes.append((rid, field, predicted, truth[field]))
        print(f"[{i}/{len(ids)}] {rid}  {latencies[-1]:.1f} ms")

    summary = [
        f"# Evaluation: {output_path}",
        "",
        f"Extractor: {extractor_name}. Receipts: {len(ids)} (first {N_RECEIPTS} {split} IDs). Extraction failures: {failures}.",
        "",
        "| Field | Scored | Exact | Fuzzy (>= 0.9) |",
        "|---|---|---|---|",
    ]
    for field in FIELDS:
        summary.append(f"| {field} | {scored[field]} | {exact[field] / scored[field]:.0%} | {fuzzy[field] / scored[field]:.0%} |")
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    summary += ["", f"Latency per receipt: p50 {p50:.1f} ms, p95 {p95:.1f} ms"]

    details = ["", "## Mistakes (fuzzy match failed)", "", "| Receipt | Field | Predicted | Truth |", "|---|---|---|---|"]
    for rid, field, predicted, true_value in mistakes:
        details.append(f"| {rid} | {field} | {predicted} | {true_value} |")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(summary + details) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "results/eval_v1.md"
    extractor_name = sys.argv[2] if len(sys.argv) > 2 else "llm"
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    main(output, extractor_name, split)