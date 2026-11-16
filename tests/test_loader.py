"""Tests for docagent.loader. Run from project root: python -m pytest -v"""
from docagent.loader import (
    DATA_DIR,
    list_ids,
    load_ground_truth,
    load_ocr_lines,
    load_ocr_text,
)

SAMPLE_ID = "X00016469612"
FIELDS = {"company", "date", "address", "total"}

def test_split_sizes():
    assert len(list_ids("train")) == 626
    assert len(list_ids("test")) == 347


def test_every_receipt_has_ground_truth():
    for split in ("train", "test"):
        box_ids = set(list_ids(split))
        entity_ids = {p.stem for p in (DATA_DIR / split / "entities").glob("*.txt")}
        assert box_ids == entity_ids

def test_coordinates_are_removed():
    lines = load_ocr_lines(SAMPLE_ID)
    assert lines[0] == "TAN WOON YANN"
    assert lines[1] == "BOOK TA .K(TAMAN DAYA) SDN BND"


def test_commas_inside_text_are_kept():
    lines = load_ocr_lines(SAMPLE_ID)
    assert "NO.53 55,57 & 59, JALAN SAGU 18," in lines


def test_text_keeps_one_line_per_row():
    lines = load_ocr_lines(SAMPLE_ID)
    text = load_ocr_text(SAMPLE_ID)
    assert text.count("\n") == len(lines) - 1

def test_ground_truth_sample():
    gt = load_ground_truth(SAMPLE_ID)
    assert set(gt) == FIELDS
    assert gt["date"] == "25/12/2018"
    assert gt["total"] == "9.00"



def test_ground_truth_has_no_unknown_fields():
    for split in ("train", "test"):
        for rid in list_ids(split):
            unknown = set(load_ground_truth(rid, split)) - FIELDS
            assert not unknown, f"{split}/{rid} has unknown keys {sorted(unknown)}"


def test_only_known_receipt_is_missing_a_field():
    incomplete = []
    for split in ("train", "test"):
        for rid in list_ids(split):
            if set(load_ground_truth(rid, split)) != FIELDS:
                incomplete.append(f"{split}/{rid}")
    assert incomplete == ["train/X51005663280"]