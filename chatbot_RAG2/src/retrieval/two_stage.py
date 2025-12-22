"""
Reader/Decider scaffolding for two-stage RAG flow.
"""
from typing import List, Dict, Any
from loguru import logger
from uuid import uuid4
from src.retrieval.evidence_models import (
    ReaderInput,
    EvidenceBundle,
    EvidenceCard,
    DeciderInput,
    DeciderOutput,
    EvidenceOverview,
)
from src.retrieval.evidence_scoring import grade_to_numeric, compute_match_score


def _trim_text(text: str, max_len: int = 220) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


class Reader:
    """Collects retrieval outputs and shapes them into an EvidenceBundle."""

    def __init__(self, max_primary: int = 5, max_supporting: int = 3):
        self.max_primary = max_primary
        self.max_supporting = max_supporting

    def build_evidence_bundle(self, inp: ReaderInput) -> EvidenceBundle:
        logger.debug("Reader.build_evidence_bundle called.")
        scored_items: List[Dict[str, Any]] = []
        seen_content = set()

        for idx, item in enumerate(inp.retrieved_items):
            content = item.get("content", "")
            if content in seen_content:
                continue
            seen_content.add(content)

            metadata: Dict[str, Any] = item.get("metadata") or {}
            pico_ev = {
                "P": metadata.get("disease"),
                "I": metadata.get("drug") or metadata.get("intervention"),
                "C": metadata.get("comparator"),
                "O": metadata.get("outcome"),
                "grade": metadata.get("grade"),
                "year": metadata.get("year"),
            }
            score = compute_match_score(
                pico_question={
                    "P": inp.question_text,
                    "I": inp.question_text,
                    "C": inp.question_text,
                    "O": inp.question_text,
                },
                evidence=pico_ev,
                patient_profile=inp.patient_profile.model_dump() if hasattr(inp.patient_profile, "model_dump") else {},
            )
            scored_items.append({"item": item, "score": score, "idx": idx})

        scored_items.sort(key=lambda x: (-x["score"], x["idx"]))

        primary_cards: List[EvidenceCard] = []
        supporting_cards: List[EvidenceCard] = []

        for rank, scored in enumerate(scored_items):
            if len(primary_cards) >= self.max_primary and len(supporting_cards) >= self.max_supporting:
                break

            item = scored["item"]
            metadata: Dict[str, Any] = item.get("metadata") or {}
            evidence_id = (
                metadata.get("chunk_id")
                or metadata.get("source_id")
                or f"chunk-{rank}"
            )
            pico = {
                "P": metadata.get("disease"),
                "I": metadata.get("drug") or metadata.get("intervention"),
                "C": metadata.get("comparator"),
                "O": metadata.get("outcome"),
            }
            card = EvidenceCard(
                evidence_id=str(evidence_id),
                source_type=metadata.get("source_type"),
                reference=metadata.get("source"),
                year=metadata.get("year"),
                locale=metadata.get("locale"),
                pico=pico,
                grade=metadata.get("grade"),
                effect_size_text=metadata.get("effect_size_text"),
                key_finding=_trim_text(
                    metadata.get("key_finding")
                    or metadata.get("summary")
                    or item.get("content", "")
                ),
                applicability={
                    "population_match": float(metadata.get("population_match", 0.0) or 0.0),
                    "intervention_match": float(metadata.get("intervention_match", 0.0) or 0.0),
                    "comorbidity_risk_note": metadata.get("comorbidity_risk_note"),
                    "setting": metadata.get("setting"),
                },
                safety_note=_trim_text(metadata.get("safety_note") or ""),
                limitations=_trim_text(metadata.get("limitations") or ""),
                tags=metadata.get("tags") or [],
            )

            if len(primary_cards) < self.max_primary:
                primary_cards.append(card)
            elif len(supporting_cards) < self.max_supporting:
                supporting_cards.append(card)

        overview = EvidenceOverview(
            summary="Evidence bundle generated from retrieval results.",
            consistency=None,
            num_high_grade=sum(1 for c in primary_cards if grade_to_numeric(c.grade) >= 0.9),
            num_moderate_grade=sum(
                1 for c in primary_cards if 0.5 <= grade_to_numeric(c.grade) < 0.9
            ),
            notes=None,
        )

        return EvidenceBundle(
            question_id=str(uuid4()),
            normalized_question=inp.question_text,
            patient_profile=inp.patient_profile,
            primary_evidence=primary_cards,
            supporting_evidence=supporting_cards,
            conflicting_evidence=[],
            evidence_overview=overview,
        )


class Decider:
    """Consumes an EvidenceBundle and produces the final answer."""

    def answer(self, inp: DeciderInput) -> DeciderOutput:
        logger.debug("Decider.answer called (stub).")
        note = (
            "Two-stage Reader/Decider is scaffolded; legacy single-stage LLM "
            "response should be used for now."
        )
        ids = select_evidence_ids(inp.evidence_bundle)
        return DeciderOutput(
            final_answer=note,
            reasoning_outline=note,
            cited_evidence_ids=ids,
            uncertainty_note=None,
        )


def select_evidence_ids(evidence_bundle: EvidenceBundle) -> List[str]:
    ids: List[str] = []
    for ev in (
        evidence_bundle.primary_evidence
        + evidence_bundle.supporting_evidence
        + evidence_bundle.conflicting_evidence
    ):
        if ev.evidence_id:
            ids.append(ev.evidence_id)
    return ids
