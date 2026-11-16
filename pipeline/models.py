"""Extraction contract. Both parsers must return an `Extraction`; Pydantic rejects
anything outside the vocabulary or ranges before it reaches the database."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SymptomCode = Literal["mouth_sores", "pain", "swallowing", "nausea", "vomiting", "dry_mouth",
                      "taste_change", "thick_saliva", "fever", "fatigue", "general_unwell"]
Intensity = Literal["none", "mild", "moderate", "severe"]
TimeOfDay = Literal["morning", "afternoon", "evening", "night"]
MedCode = Literal["cisplatin", "ondansetron", "olanzapine", "oxycodone", "acetaminophen",
                  "lidocaine_rinse"]
MedAction = Literal["took", "missed", "started", "stopped", "helped", "did_not_help"]


class SymptomMention(BaseModel):
    symptom: SymptomCode
    present: bool = Field(description="false only when the patient explicitly says it is absent")
    intensity: Optional[Intensity] = Field(None, description="only if the patient's words state it")
    count: Optional[int] = Field(None, ge=0, le=30, description="episodes, vomiting only, only if stated")
    value_num: Optional[float] = Field(None, description="a number the patient gave (pain 0-10, sores 0-4, temp F)")
    relative_day: int = Field(0, ge=-7, le=0, description="0 = day of report, -1 = yesterday/last night")
    time_of_day: Optional[TimeOfDay] = None
    evidence: str = Field(description="exact words copied from the report")
    confidence: float = Field(0.8, ge=0, le=1)

    @model_validator(mode="after")
    def _coherent(self):
        if not self.present and self.intensity not in (None, "none"):
            raise ValueError("absent symptom cannot have an intensity")
        if self.count is not None and self.symptom != "vomiting":
            raise ValueError("count is only valid for vomiting")
        if self.value_num is not None:
            limits = {"pain": (0, 10), "mouth_sores": (0, 4), "nausea": (0, 4),
                      "swallowing": (0, 3), "fever": (95, 108)}
            lo, hi = limits.get(self.symptom, (None, None))
            if lo is None or not (lo <= self.value_num <= hi):
                raise ValueError(f"value_num {self.value_num} out of range for {self.symptom}")
        return self


class MedicationMention(BaseModel):
    medication: MedCode
    action: MedAction
    evidence: str


class Extraction(BaseModel):
    symptoms: List[SymptomMention] = []
    medications: List[MedicationMention] = []
