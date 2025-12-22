"""
Structured data models for two-stage Reader/Decider RAG flow.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    stage: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    post_surgery: Optional[bool] = None
    comorbidities: List[str] = Field(default_factory=list)
    special_notes: List[str] = Field(default_factory=list)


class PICOEvidence(BaseModel):
    evidence_id: str
    source_type: Optional[str] = None
    reference: Optional[str] = None
    year: Optional[int] = None
    locale: Optional[str] = None
    P: Optional[str] = None
    I: Optional[str] = None
    C: Optional[str] = None
    O: Optional[str] = None
    effect_size: Optional[Dict[str, Any]] = None
    grade: Optional[str] = None
    raw_excerpt: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_id: Optional[str] = None


class Applicability(BaseModel):
    population_match: float = 0.0
    intervention_match: float = 0.0
    comorbidity_risk_note: Optional[str] = None
    setting: Optional[str] = None


class EvidenceCard(BaseModel):
    evidence_id: str
    source_type: Optional[str] = None
    reference: Optional[str] = None
    year: Optional[int] = None
    locale: Optional[str] = None
    pico: Dict[str, Optional[str]] = Field(default_factory=dict)
    grade: Optional[str] = None
    effect_size_text: Optional[str] = None
    key_finding: Optional[str] = None
    applicability: Applicability = Field(default_factory=Applicability)
    safety_note: Optional[str] = None
    limitations: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class EvidenceOverview(BaseModel):
    summary: Optional[str] = None
    consistency: Optional[str] = None
    num_high_grade: int = 0
    num_moderate_grade: int = 0
    notes: Optional[str] = None


class EvidenceBundle(BaseModel):
    question_id: str
    normalized_question: Optional[str] = None
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    primary_evidence: List[EvidenceCard] = Field(default_factory=list)
    supporting_evidence: List[EvidenceCard] = Field(default_factory=list)
    conflicting_evidence: List[EvidenceCard] = Field(default_factory=list)
    evidence_overview: EvidenceOverview = Field(default_factory=EvidenceOverview)


class ReaderInput(BaseModel):
    question_text: str
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    retrieved_items: List[Any] = Field(default_factory=list)


class DeciderInput(BaseModel):
    question_text: str
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    evidence_bundle: EvidenceBundle


class DeciderOutput(BaseModel):
    final_answer: Optional[str] = None
    reasoning_outline: Optional[str] = None
    cited_evidence_ids: List[str] = Field(default_factory=list)
    uncertainty_note: Optional[str] = None
