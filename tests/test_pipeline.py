"""Run: python -m pytest -q"""
import pytest
from pydantic import ValidationError

from pipeline import parse_rules
from pipeline.models import Extraction, SymptomMention
from pipeline.validate import validate


def test_schema_rejects_absent_symptom_with_intensity():
    with pytest.raises(ValidationError):
        SymptomMention(symptom="nausea", present=False, intensity="severe", evidence="x")


def test_schema_rejects_out_of_range_pain():
    with pytest.raises(ValidationError):
        SymptomMention(symptom="pain", present=True, value_num=14, evidence="pain 14")


def test_schema_rejects_unknown_symptom():
    with pytest.raises(ValidationError):
        Extraction.model_validate({"symptoms": [{"symptom": "headache", "present": True, "evidence": "x"}]})


def test_grounding_rejects_invented_evidence():
    ext = Extraction(symptoms=[SymptomMention(symptom="vomiting", present=True, count=3,
                                              evidence="vomited three times")])
    acc, rej = validate(ext, "felt a bit queasy after lunch", "llm:test")
    assert not acc and rej and "evidence not found" in rej[0]["review_reason"]


def test_fever_always_needs_review():
    ext = parse_rules.parse("had a fever of 100.9 last night")
    acc, _ = validate(ext, "had a fever of 100.9 last night", "rules")
    assert acc[0]["review_status"] == "needs_review"


def test_negation_scope():
    ext = parse_rules.parse("No vomiting but nauseous all morning")
    got = {(m.symptom, m.present) for m in ext.symptoms}
    assert got == {("vomiting", False), ("nausea", True)}


def test_vague_count_stays_null():
    ext = parse_rules.parse("threw up a few times")
    assert ext.symptoms[0].count is None and ext.symptoms[0].present


def test_cross_check_flags_disagreement():
    text = "no nausea today"
    llm = Extraction(symptoms=[SymptomMention(symptom="nausea", present=True, evidence="nausea today")])
    acc, _ = validate(llm, text, "llm:test", cross_check=parse_rules.parse(text))
    assert acc[0]["review_status"] == "needs_review"
