"""Extractor B: labels each OCR line with the ONNX classifier, then combines lines into fields."""
import re

import joblib
import numpy as np
import onnxruntime as ort

from docagent.features import to_matrix
from docagent.labels import LABELS, to_number
from docagent.loader import DATA_DIR

MODEL_DIR = DATA_DIR.parent.parent / "models"
DATE_PATTERN = re.compile(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2} ?[A-Z]{3} ?\d{2,4}")

SESSION = ort.InferenceSession(str(MODEL_DIR / "line_classifier_int8.onnx"))
LINE_VEC, PREV_VEC = joblib.load(MODEL_DIR / "vectorizers.joblib")


def softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def label_lines(text: str) -> list[tuple[str, str, float]]:
    """Return (line, label, confidence) for every line of the receipt."""
    lines = text.split("\n")
    rows = [{"line": line, "previous": lines[i - 1] if i > 0 else ""} for i, line in enumerate(lines)]
    probs = softmax(SESSION.run(None, {"features": to_matrix(rows, LINE_VEC, PREV_VEC)})[0])
    best = probs.argmax(axis=1)
    return [(line, LABELS[k], float(probs[i, k])) for i, (line, k) in enumerate(zip(lines, best))]


def extract_with_classifier(text: str) -> dict:
    labelled = label_lines(text)

    def most_confident(label: str) -> str:
        candidates = [(conf, line) for line, lab, conf in labelled if lab == label]
        return max(candidates)[1] if candidates else ""

    date_line = most_confident("date")
    match = DATE_PATTERN.search(date_line.upper())
    total = to_number(most_confident("total"))

    return {
        "company": most_confident("company"),
        "date": match.group(0) if match else date_line,
        "address": " ".join(line for line, lab, _ in labelled if lab == "address"),
        "total": total if total is not None else "",
    }