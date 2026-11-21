from docagent.loader import list_ids, load_ground_truth, load_ocr_lines


"""Line labeller: uses the ground truth to label every OCR line."""
import re
from difflib import SequenceMatcher

LABELS = ["other", "company", "address", "date", "total"]


def clean(text: str) -> str:
    return " ".join(str(text).upper().split())


def to_number(text: str) -> float | None:
    digits = re.sub(r"[^0-9.]", "", text)
    try:
        return float(digits)
    except ValueError:
        return None


def label_line(line: str, truth: dict) -> str:
    text = clean(line)

    if "date" in truth and clean(truth["date"]) in text:
        return "date"

    if "total" in truth:
        number = to_number(text)
        true_total = to_number(truth["total"])
        if number is not None and true_total is not None and abs(number - true_total) < 0.005:
            return "total"

    if "company" in truth and SequenceMatcher(None, text, clean(truth["company"])).ratio() >= 0.8:
        return "company"

    if "address" in truth and len(text) > 3 and text in clean(truth["address"]):
        return "address"

    return "other"
def build_dataset(split: str = "train") -> list[dict]:
    rows = []
    for rid in list_ids(split):
        truth = load_ground_truth(rid, split)
        lines = load_ocr_lines(rid, split)
        for i, line in enumerate(lines):
            previous = lines[i - 1] if i > 0 else ""
            rows.append({"receipt": rid, "line": line, "previous": previous, "label": label_line(line, truth)})
    return rows