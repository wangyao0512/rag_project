"""Scoring helpers for two-stage Reader/Decider pipeline."""
from typing import Dict, Optional
from datetime import datetime


def compute_match_score(pico_question: Dict[str, Optional[str]],
                        evidence: Dict[str, Optional[str]],
                        patient_profile: Optional[Dict] = None,
                        *,
                        weight_p: float = 0.4,
                        weight_i: float = 0.35,
                        weight_c: float = 0.15,
                        weight_o: float = 0.1,
                        grade_boost_high: float = 0.1,
                        recency_half_life_years: int = 6) -> float:
    score = 0.0
    pq = {k.lower(): (v or "").lower() for k, v in (pico_question or {}).items()}
    ev = {k.lower(): (v or "").lower() for k, v in (evidence or {}).items()}

    def _hit(query_val: str, ev_val: str) -> bool:
        return bool(query_val and ev_val and (query_val in ev_val or ev_val in query_val))

    if _hit(pq.get("p"), ev.get("p")):
        score += weight_p
    if _hit(pq.get("i"), ev.get("i")):
        score += weight_i
    if _hit(pq.get("c"), ev.get("c")):
        score += weight_c
    if _hit(pq.get("o"), ev.get("o")):
        score += weight_o

    grade = (evidence or {}).get("grade", "") or ""
    gl = grade.lower()
    if "high" in gl or gl == "a":
        score += grade_boost_high
    elif "moderate" in gl or gl == "b":
        score += grade_boost_high * 0.6
    elif gl:
        score += grade_boost_high * 0.3

    year = None
    try:
        year = int((evidence or {}).get("year") or 0)
    except ValueError:
        year = None
    if year:
        age = max(0, datetime.now().year - year)
        decay = 0.5 ** (age / float(recency_half_life_years))
        score *= (0.7 + 0.3 * decay)

    if patient_profile:
        _ = patient_profile

    return round(score, 4)


def grade_to_numeric(grade: Optional[str]) -> float:
    if not grade:
        return 0.0
    g = grade.lower()
    if "high" in g or g == "a":
        return 1.0
    if "moderate" in g or g == "b":
        return 0.7
    if "low" in g or g == "c":
        return 0.4
    return 0.2
