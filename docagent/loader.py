"""Loader: reads SROIE receipts (OCR lines + ground truth) from disk."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sroie"
def list_ids(split: str = "train") -> list[str]:
    """Return sorted receipt IDs for a split ("train" or "test")."""
    box_dir = DATA_DIR / split / "box"
    return sorted(p.stem for p in box_dir.glob("*.txt"))
def load_ocr_lines(receipt_id: str, split: str = "train") -> list[str]:
    """Return the text of each OCR line, dropping the 8 box coordinates."""
    path = DATA_DIR / split / "box" / f"{receipt_id}.txt"
    lines = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.strip().split(",", 8)
            if len(parts) < 9:
                continue
            text = parts[8].strip()
            if text:
                lines.append(text)
    return lines
def load_ocr_text(receipt_id: str, split: str = "train") -> str:
    """Return the receipt as one string, one OCR line per row (the LLM input)."""
    return "\n".join(load_ocr_lines(receipt_id, split))


def load_ground_truth(receipt_id: str, split: str = "train") -> dict:
    """Return the answer key: {"company", "date", "address", "total"}."""
    path = DATA_DIR / split / "entities" / f"{receipt_id}.txt"
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)