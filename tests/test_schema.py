"""Tests for docagent.schema."""
import pytest
from pydantic import ValidationError

from docagent.schema import Receipt

GOOD = '{"company": "BOOK TA .K (TAMAN DAYA) SDN BHD", "date": "25/12/2018", "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.", "total": "9.00"}'


def test_string_total_becomes_float():
    receipt = Receipt.model_validate_json(GOOD)
    assert receipt.total == 9.0


def test_currency_in_total_is_rejected():
    with pytest.raises(ValidationError):
        Receipt.model_validate_json(GOOD.replace('"9.00"', '"RM 9.00"'))


def test_missing_field_is_rejected():
    with pytest.raises(ValidationError):
        Receipt.model_validate_json('{"company": "X", "date": "1/1/2018", "total": 1}')


def test_negative_total_is_rejected():
    with pytest.raises(ValidationError):
        Receipt.model_validate_json(GOOD.replace('"9.00"', '"-9.00"'))