"""Features: turns OCR lines into numbers (character n-grams)."""
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def split_by_receipt(rows: list[dict], val_fraction: float = 0.2, seed: int = 42):
    receipts = sorted({row["receipt"] for row in rows})
    random.Random(seed).shuffle(receipts)
    val_ids = set(receipts[: int(len(receipts) * val_fraction)])
    train = [row for row in rows if row["receipt"] not in val_ids]
    val = [row for row in rows if row["receipt"] in val_ids]
    return train, val


def make_vectorizers():
    line_vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=1500, binary=True)
    prev_vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=500, binary=True)
    return line_vec, prev_vec


def to_matrix(rows: list[dict], line_vec, prev_vec) -> np.ndarray:
    current = line_vec.transform([row["line"] for row in rows]).toarray()
    previous = prev_vec.transform([row["previous"] for row in rows]).toarray()
    return np.hstack([current, previous]).astype(np.float32)