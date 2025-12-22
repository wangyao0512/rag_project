from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

# MCP FastMCP server (Python).
# Install: pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("melanoma-staging-ajcc8-csco2025")


# =========================
# References / citations
# =========================
CITE_TNM_DEF = "CSCO 2025 黑色素瘤 AJCC第8版：T/N/M 定义页（p14-17）"
CITE_STAGE_MATRIX = "CSCO 2025 黑色素瘤 AJCC第8版：病理分期分组矩阵（p18）"


def _trace(rule_id: str, cite: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"id": rule_id, "cite": cite, "detail": detail or {}}


# =========================
# Stage-group matrix (M0)
# =========================
# NOTE: This matrix is the "Stage Group" (pathologic stage grouping) for M0.
# Any M1* -> Stage IV (handled separately).
STAGE_MATRIX_M0: Dict[str, Dict[str, str]] = {
    "T1a": {"N0": "IA",  "N1a": "IIIA","N1b": "IIIB","N1c": "IIIB","N2a": "IIIA","N2b": "IIIB","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T1b": {"N0": "IA",  "N1a": "IIIA","N1b": "IIIB","N1c": "IIIB","N2a": "IIIA","N2b": "IIIB","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T2a": {"N0": "IB",  "N1a": "IIIA","N1b": "IIIB","N1c": "IIIB","N2a": "IIIA","N2b": "IIIB","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T2b": {"N0": "IIA", "N1a": "IIIB","N1b": "IIIB","N1c": "IIIB","N2a": "IIIB","N2b": "IIIB","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T3a": {"N0": "IIA", "N1a": "IIIB","N1b": "IIIB","N1c": "IIIB","N2a": "IIIB","N2b": "IIIB","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T3b": {"N0": "IIB", "N1a": "IIIC","N1b": "IIIC","N1c": "IIIC","N2a": "IIIC","N2b": "IIIC","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T4a": {"N0": "IIB", "N1a": "IIIC","N1b": "IIIC","N1c": "IIIC","N2a": "IIIC","N2b": "IIIC","N2c": "IIIC","N3a": "IIIC","N3b": "IIIC","N3c": "IIIC"},
    "T4b": {"N0": "IIC", "N1a": "IIIC","N1b": "IIIC","N1c": "IIIC","N2a": "IIIC","N2b": "IIIC","N2c": "IIIC","N3a": "IIID","N3b": "IIID","N3c": "IIID"},
}

VALID_T_FOR_MATRIX = set(STAGE_MATRIX_M0.keys())  # T1a..T4b only
VALID_N_FOR_MATRIX = set(next(iter(STAGE_MATRIX_M0.values())).keys())  # N0..N3c (no Nx)


# =========================
# 1) Derive T (AJCC8 melanoma; MVP but aligned with thickness+ulceration)
# =========================
def derive_T(
    primary_present: Literal["yes", "no", "unknown"] = "unknown",
    in_situ: Literal["yes", "no", "unknown"] = "unknown",
    breslow_mm: Optional[float] = None,
    ulceration: Literal["present", "absent", "unknown"] = "unknown",
) -> Tuple[str, List[str], List[str], List[Dict[str, Any]]]:
    """
    Returns: (T, missing_fields, warnings, trace)

    MVP rules consistent with AJCC8 concept:
      - primary_present == no -> T0
      - in_situ == yes -> Tis
      - if thickness missing -> Tx
      - otherwise classify by thickness bands + ulceration:
          <=1.0:
            <0.8 and no ulcer -> T1a
            else -> T1b
          (1.0,2.0] -> T2a/T2b (by ulceration)
          (2.0,4.0] -> T3a/T3b
          >4.0      -> T4a/T4b
    """
    missing: List[str] = []
    warnings: List[str] = []
    tr: List[Dict[str, Any]] = []

    if primary_present == "no":
        tr.append(_trace("T_RULE_T0_NO_PRIMARY", CITE_TNM_DEF, {"primary_present": primary_present}))
        return "T0", [], [], tr

    if in_situ == "yes":
        tr.append(_trace("T_RULE_TIS_IN_SITU", CITE_TNM_DEF, {"in_situ": in_situ}))
        return "Tis", [], [], tr

    if breslow_mm is None:
        missing.append("primary_tumor.breslow_mm")
        tr.append(_trace("T_RULE_TX_UNKNOWN_THICKNESS", CITE_TNM_DEF, {"breslow_mm": None}))
        return "Tx", missing, [], tr

    # ulceration is required for a/b split for T2+
    if ulceration == "unknown":
        missing.append("primary_tumor.ulceration")

    # Determine T
    if breslow_mm <= 1.0:
        if breslow_mm < 0.8 and ulceration == "absent":
            tr.append(_trace("T_RULE_T1A_LT0_8_NO_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm, "ulceration": ulceration}))
            return "T1a", missing, warnings, tr
        tr.append(_trace("T_RULE_T1B_WITHIN_1_0_ELSE", CITE_TNM_DEF, {"breslow_mm": breslow_mm, "ulceration": ulceration}))
        return "T1b", missing, warnings, tr

    if 1.0 < breslow_mm <= 2.0:
        if ulceration == "present":
            tr.append(_trace("T_RULE_T2B_1_0_2_0_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
            return "T2b", missing, warnings, tr
        if ulceration == "absent":
            tr.append(_trace("T_RULE_T2A_1_0_2_0_NO_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
            return "T2a", missing, warnings, tr
        # ulceration unknown -> return Tx or provisional? safer: Tx-like
        warnings.append("ulceration_unknown_cannot_finalize_T2a_vs_T2b")
        tr.append(_trace("T_RULE_T2_UNDETERMINED_ULC_UNKNOWN", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
        return "Tx", missing, warnings, tr

    if 2.0 < breslow_mm <= 4.0:
        if ulceration == "present":
            tr.append(_trace("T_RULE_T3B_2_0_4_0_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
            return "T3b", missing, warnings, tr
        if ulceration == "absent":
            tr.append(_trace("T_RULE_T3A_2_0_4_0_NO_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
            return "T3a", missing, warnings, tr
        warnings.append("ulceration_unknown_cannot_finalize_T3a_vs_T3b")
        tr.append(_trace("T_RULE_T3_UNDETERMINED_ULC_UNKNOWN", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
        return "Tx", missing, warnings, tr

    # >4.0
    if ulceration == "present":
        tr.append(_trace("T_RULE_T4B_GT4_0_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
        return "T4b", missing, warnings, tr
    if ulceration == "absent":
        tr.append(_trace("T_RULE_T4A_GT4_0_NO_ULC", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
        return "T4a", missing, warnings, tr

    warnings.append("ulceration_unknown_cannot_finalize_T4a_vs_T4b")
    tr.append(_trace("T_RULE_T4_UNDETERMINED_ULC_UNKNOWN", CITE_TNM_DEF, {"breslow_mm": breslow_mm}))
    return "Tx", missing, warnings, tr


# =========================
# 2) Derive N (complete for N0..N3c + Nx per your CSCO table concepts)
# =========================
def derive_N(
    nodes_positive_count: Optional[int] = None,
    nodes_clinically_occult: Literal["yes", "no", "unknown"] = "unknown",  # yes=microscopic/occult; no=clinically evident
    matted_nodes: Literal["yes", "no", "unknown"] = "unknown",
    in_transit_satellite_microsatellite: Literal["present", "absent", "unknown"] = "unknown",
    # optional backward compat:
    clinically_apparent_nodes: Optional[int] = None,
) -> Tuple[str, List[str], List[str], List[Dict[str, Any]]]:
    """
    Returns: (N, missing_fields, warnings, trace)

    Rules (AJCC8 melanoma style; matches your CSCO table structure):
      N0: 0 LN and no IT/satellite/microsatellite
      N1c: 0 LN but IT/satellite/microsatellite present

      If matted nodes => N3b

      If IT present and LN >=2 => N3c
      If LN >=4:
         occult -> N3a
         clinical -> N3b
      If IT present and LN >=1 (and LN<2 already) => N2c
      If LN in [2,3]:
         occult -> N2a
         clinical -> N2b
      If LN == 1:
         occult -> N1a
         clinical -> N1b

      Otherwise Nx when insufficient.
    """
    missing: List[str] = []
    warnings: List[str] = []
    tr: List[Dict[str, Any]] = []

    if nodes_positive_count is None:
        missing.append("regional_nodes.nodes_positive_count")
    if in_transit_satellite_microsatellite == "unknown":
        missing.append("regional_nodes.in_transit_satellite_microsatellite")

    # Determine occult/evident
    if nodes_clinically_occult == "unknown":
        if clinically_apparent_nodes is None:
            missing.append("regional_nodes.nodes_clinically_occult")
        else:
            nodes_clinically_occult = "no" if clinically_apparent_nodes >= 1 else "yes"
            warnings.append("nodes_clinically_occult_inferred_from_clinically_apparent_nodes")
            tr.append(_trace(
                "N_RULE_OCCULT_INFERRED_FROM_CLINICAL_COUNT",
                CITE_TNM_DEF,
                {"clinically_apparent_nodes": clinically_apparent_nodes, "nodes_clinically_occult": nodes_clinically_occult},
            ))

    if nodes_positive_count is None or in_transit_satellite_microsatellite == "unknown" or nodes_clinically_occult == "unknown":
        tr.append(_trace(
            "N_RULE_NX_INSUFFICIENT_INFO",
            CITE_TNM_DEF,
            {
                "nodes_positive_count": nodes_positive_count,
                "nodes_clinically_occult": nodes_clinically_occult,
                "in_transit_satellite_microsatellite": in_transit_satellite_microsatellite,
            },
        ))
        return "Nx", missing, warnings, tr

    it_present = (in_transit_satellite_microsatellite == "present")
    occult = (nodes_clinically_occult == "yes")
    matted = (matted_nodes == "yes")

    # N0 / N1c
    if nodes_positive_count == 0 and not it_present:
        tr.append(_trace("N_RULE_N0_NO_NODES_NO_IT", CITE_TNM_DEF, {"nodes_positive_count": 0}))
        return "N0", [], warnings, tr

    if nodes_positive_count == 0 and it_present:
        tr.append(_trace("N_RULE_N1C_IT_NO_NODES", CITE_TNM_DEF, {"nodes_positive_count": 0}))
        return "N1c", [], warnings, tr

    # matted nodes => N3b
    if matted:
        tr.append(_trace("N_RULE_N3B_MATTED", CITE_TNM_DEF, {"matted_nodes": True}))
        return "N3b", [], warnings, tr

    # N3c: >=2 LN + IT
    if nodes_positive_count >= 2 and it_present:
        tr.append(_trace("N_RULE_N3C_GE2_NODES_WITH_IT", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
        return "N3c", [], warnings, tr

    # >=4 LN
    if nodes_positive_count >= 4:
        if occult:
            tr.append(_trace("N_RULE_N3A_GE4_OCCULT", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
            return "N3a", [], warnings, tr
        tr.append(_trace("N_RULE_N3B_GE4_CLINICAL", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
        return "N3b", [], warnings, tr

    # N2c: >=1 LN + IT (here LN is 1, because >=2 already caught by N3c)
    if nodes_positive_count >= 1 and it_present:
        tr.append(_trace("N_RULE_N2C_GE1_NODE_WITH_IT", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
        return "N2c", [], warnings, tr

    # 2-3 LN
    if 2 <= nodes_positive_count <= 3:
        if occult:
            tr.append(_trace("N_RULE_N2A_2_3_OCCULT", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
            return "N2a", [], warnings, tr
        tr.append(_trace("N_RULE_N2B_2_3_CLINICAL", CITE_TNM_DEF, {"nodes_positive_count": nodes_positive_count}))
        return "N2b", [], warnings, tr

    # 1 LN
    if nodes_positive_count == 1:
        if occult:
            tr.append(_trace("N_RULE_N1A_1_OCCULT", CITE_TNM_DEF, {"nodes_positive_count": 1}))
            return "N1a", [], warnings, tr
        tr.append(_trace("N_RULE_N1B_1_CLINICAL", CITE_TNM_DEF, {"nodes_positive_count": 1}))
        return "N1b", [], warnings, tr

    warnings.append("unexpected_N_inputs_fell_through")
    tr.append(_trace(
        "N_RULE_FALLBACK",
        CITE_TNM_DEF,
        {
            "nodes_positive_count": nodes_positive_count,
            "nodes_clinically_occult": nodes_clinically_occult,
            "it_present": it_present,
            "matted_nodes": matted_nodes,
        },
    ))
    return "Nx", [], warnings, tr


# =========================
# 3) Derive M (complete for M0/M1a/M1b/M1c/M1d + LDH (0/1/?), Mx)
# =========================
def derive_M(
    distant_metastasis_status: Literal["none", "present", "unknown"] = "unknown",
    site: Literal["skin_soft_tissue", "nonregional_ln", "lung", "other_viscera", "cns", "unknown"] = "unknown",
    sites: Optional[List[Literal["skin_soft_tissue", "nonregional_ln", "lung", "other_viscera", "cns"]]] = None,
    ldh: Literal["normal", "elevated", "unknown"] = "unknown",
) -> Tuple[Dict[str, str], List[str], List[str], List[Dict[str, Any]]]:
    """
    Returns: (M_obj, missing_fields, warnings, trace)

    Worst-site hierarchy:
      CNS => M1d
      other viscera (non-CNS) => M1c
      lung => M1b
      skin/soft tissue and/or nonregional LN => M1a

    LDH suffix:
      normal => (0)
      elevated => (1)
      unknown => (?) + missing_fields includes ldh
    """
    missing: List[str] = []
    warnings: List[str] = []
    tr: List[Dict[str, Any]] = []

    if distant_metastasis_status == "none":
        tr.append(_trace("M_RULE_M0_NO_DISTANT", CITE_TNM_DEF, {"status": "none"}))
        return {"M_base": "M0", "M_with_ldh": "M0"}, [], [], tr

    if distant_metastasis_status == "unknown":
        missing.append("distant_metastasis.status")
        tr.append(_trace("M_RULE_MX_UNKNOWN_STATUS", CITE_TNM_DEF, {"status": "unknown"}))
        return {"M_base": "Mx", "M_with_ldh": "Mx"}, missing, [], tr

    # present
    norm_sites: List[str] = []
    if sites:
        norm_sites = list(sites)
    elif site != "unknown":
        norm_sites = [site]

    if not norm_sites:
        missing.append("distant_metastasis.site_or_sites")
        tr.append(_trace("M_RULE_MX_PRESENT_BUT_SITE_UNKNOWN", CITE_TNM_DEF, {"status": "present"}))
        return {"M_base": "Mx", "M_with_ldh": "Mx"}, missing, [], tr

    has_cns = "cns" in norm_sites
    has_other_viscera = "other_viscera" in norm_sites
    has_lung = "lung" in norm_sites
    has_m1a_bucket = any(s in ("skin_soft_tissue", "nonregional_ln") for s in norm_sites)

    if has_cns:
        m_base = "M1d"
        tr.append(_trace("M_RULE_M1D_CNS", CITE_TNM_DEF, {"sites": norm_sites, "ldh": ldh}))
    elif has_other_viscera:
        m_base = "M1c"
        tr.append(_trace("M_RULE_M1C_OTHER_VISCERA_NON_CNS", CITE_TNM_DEF, {"sites": norm_sites, "ldh": ldh}))
    elif has_lung:
        m_base = "M1b"
        tr.append(_trace("M_RULE_M1B_LUNG", CITE_TNM_DEF, {"sites": norm_sites, "ldh": ldh}))
    elif has_m1a_bucket:
        m_base = "M1a"
        tr.append(_trace("M_RULE_M1A_SKIN_SOFT_TISSUE_OR_NONREGIONAL_LN", CITE_TNM_DEF, {"sites": norm_sites, "ldh": ldh}))
    else:
        warnings.append("unexpected_metastasis_sites")
        tr.append(_trace("M_RULE_FALLBACK_UNEXPECTED_SITES", CITE_TNM_DEF, {"sites": norm_sites, "ldh": ldh}))
        return {"M_base": "Mx", "M_with_ldh": "Mx"}, [], warnings, tr

    if ldh == "normal":
        m_with_ldh = f"{m_base}(0)"
    elif ldh == "elevated":
        m_with_ldh = f"{m_base}(1)"
    else:
        m_with_ldh = f"{m_base}(?)"
        missing.append("distant_metastasis.ldh")
        warnings.append("ldh_unknown_for_M1_subcategory")

    return {"M_base": m_base, "M_with_ldh": m_with_ldh}, missing, warnings, tr


# =========================
# 4) Stage group from TNM (uses M_base)
# =========================
def stage_group_from_TNM(T: str, N: str, M_base: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    warnings: List[str] = []
    tr: List[Dict[str, Any]] = []

    if M_base in ("M1a", "M1b", "M1c", "M1d"):
        tr.append(_trace("STAGE_RULE_M1_TO_IV", CITE_STAGE_MATRIX, {"M_base": M_base}))
        return "IV", warnings, tr

    if M_base in ("Mx",):
        warnings.append("M_unknown_stage_group_undetermined")
        tr.append(_trace("STAGE_RULE_MX_UNDETERMINED", CITE_STAGE_MATRIX, {"M_base": M_base}))
        return "UNDETERMINED", warnings, tr

    # M0 -> matrix lookup requires T in T1a..T4b and N in N0..N3c
    if M_base != "M0":
        warnings.append(f"unexpected_M_base:{M_base}")
        tr.append(_trace("STAGE_RULE_UNEXPECTED_M_BASE", CITE_STAGE_MATRIX, {"M_base": M_base}))
        return "UNDETERMINED", warnings, tr

    if T not in VALID_T_FOR_MATRIX:
        warnings.append(f"unsupported_T_for_stage_matrix:{T}")
        tr.append(_trace("STAGE_RULE_UNSUPPORTED_T", CITE_STAGE_MATRIX, {"T": T}))
        return "UNDETERMINED", warnings, tr

    if N not in VALID_N_FOR_MATRIX:
        warnings.append(f"unsupported_N_for_stage_matrix:{N}")
        tr.append(_trace("STAGE_RULE_UNSUPPORTED_N", CITE_STAGE_MATRIX, {"N": N}))
        return "UNDETERMINED", warnings, tr

    sg = STAGE_MATRIX_M0[T][N]
    tr.append(_trace(f"STAGE_RULE_M0_MATRIX:{T}:{N}", CITE_STAGE_MATRIX, {"T": T, "N": N}))
    return sg, warnings, tr


# =========================
# 5) MCP Tool: full pipeline (recommended for chatbot)
# =========================
@mcp.tool()
def melanoma_staging_pipeline_ajcc8_csco2025(
    context: Dict[str, Any],
    primary_tumor: Dict[str, Any],
    regional_nodes: Dict[str, Any],
    distant_metastasis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pipeline:
      observations -> derive T/N/M -> stage group

    Inputs are structured for extraction from a clinical/path report or user description.
    Outputs include missing_fields and trace to support safe chatbot behavior.
    """
    # ---- T ----
    T, miss_T, warn_T, tr_T = derive_T(
        primary_present=primary_tumor.get("primary_present", "unknown"),
        in_situ=primary_tumor.get("in_situ", "unknown"),
        breslow_mm=primary_tumor.get("breslow_mm", None),
        ulceration=primary_tumor.get("ulceration", "unknown"),
    )

    # ---- N ----
    N, miss_N, warn_N, tr_N = derive_N(
        nodes_positive_count=regional_nodes.get("nodes_positive_count", None),
        nodes_clinically_occult=regional_nodes.get("nodes_clinically_occult", "unknown"),
        matted_nodes=regional_nodes.get("matted_nodes", "unknown"),
        in_transit_satellite_microsatellite=regional_nodes.get("in_transit_satellite_microsatellite", "unknown"),
        clinically_apparent_nodes=regional_nodes.get("clinically_apparent_nodes", None),  # optional backward compat
    )

    # ---- M ----
    M_obj, miss_M, warn_M, tr_M = derive_M(
        distant_metastasis_status=distant_metastasis.get("status", "unknown"),
        site=distant_metastasis.get("site", "unknown"),
        sites=distant_metastasis.get("sites", None),
        ldh=distant_metastasis.get("ldh", "unknown"),
    )
    M_base = M_obj["M_base"]
    M_with_ldh = M_obj["M_with_ldh"]

    missing_fields = miss_T + miss_N + miss_M
    warnings = warn_T + warn_N + warn_M

    stage_group, warn_stage, tr_stage = stage_group_from_TNM(T, N, M_base)
    warnings += warn_stage

    result: Dict[str, Any] = {
        "version": "ajcc8_csco2025_mvp_v2",
        "context": context,
        "tnm": {"T": T, "N": N, "M_base": M_base, "M_with_ldh": M_with_ldh},
        "stage_group": stage_group,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "trace": {
            "T_rules": tr_T,
            "N_rules": tr_N,
            "M_rules": tr_M,
            "stage_rules": tr_stage,
        },
    }

    # Optional: conditional results when M is unknown
    if M_base == "Mx":
        result["conditional_results"] = [
            {
                "if": {"distant_metastasis.status": "present"},
                "tnm": {"T": T, "N": N, "M_base": "M1c", "M_with_ldh": "M1c(?)"},
                "stage_group": "IV",
                "note": "一旦明确存在远处转移（任一 M1 亚类），分期分组进入 IV 期。",
                "trace": [_trace("STAGE_RULE_M1_TO_IV", CITE_STAGE_MATRIX)],
            },
            {
                "if": {"distant_metastasis.status": "none"},
                "note": "若明确无远处转移（M0），则可用 T×N 矩阵确定 I–III/IIID 分期分组。",
                "trace": [_trace("STAGE_RULE_M0_MATRIX_LOOKUP", CITE_STAGE_MATRIX)],
            },
        ]

    return result


# =========================
# 6) Optional sub-tools (debug / modular use)
# =========================
@mcp.tool()
def melanoma_derive_T_ajcc8(primary_tumor: Dict[str, Any]) -> Dict[str, Any]:
    T, missing, warnings, trace = derive_T(
        primary_present=primary_tumor.get("primary_present", "unknown"),
        in_situ=primary_tumor.get("in_situ", "unknown"),
        breslow_mm=primary_tumor.get("breslow_mm", None),
        ulceration=primary_tumor.get("ulceration", "unknown"),
    )
    return {"T": T, "missing_fields": missing, "warnings": warnings, "trace": trace, "version": "ajcc8_csco2025_mvp_v2"}


@mcp.tool()
def melanoma_derive_N_ajcc8(regional_nodes: Dict[str, Any]) -> Dict[str, Any]:
    N, missing, warnings, trace = derive_N(
        nodes_positive_count=regional_nodes.get("nodes_positive_count", None),
        nodes_clinically_occult=regional_nodes.get("nodes_clinically_occult", "unknown"),
        matted_nodes=regional_nodes.get("matted_nodes", "unknown"),
        in_transit_satellite_microsatellite=regional_nodes.get("in_transit_satellite_microsatellite", "unknown"),
        clinically_apparent_nodes=regional_nodes.get("clinically_apparent_nodes", None),
    )
    return {"N": N, "missing_fields": missing, "warnings": warnings, "trace": trace, "version": "ajcc8_csco2025_mvp_v2"}


@mcp.tool()
def melanoma_derive_M_ajcc8(distant_metastasis: Dict[str, Any]) -> Dict[str, Any]:
    M_obj, missing, warnings, trace = derive_M(
        distant_metastasis_status=distant_metastasis.get("status", "unknown"),
        site=distant_metastasis.get("site", "unknown"),
        sites=distant_metastasis.get("sites", None),
        ldh=distant_metastasis.get("ldh", "unknown"),
    )
    return {"M": M_obj, "missing_fields": missing, "warnings": warnings, "trace": trace, "version": "ajcc8_csco2025_mvp_v2"}


@mcp.tool()
def melanoma_stage_group_from_tnm_ajcc8(tnm: Dict[str, Any]) -> Dict[str, Any]:
    T = tnm.get("T")
    N = tnm.get("N")
    M_base = tnm.get("M_base") or tnm.get("M")
    if not T or not N or not M_base:
        return {
            "stage_group": "UNDETERMINED",
            "missing_fields": [k for k in ["T", "N", "M_base"] if not (tnm.get(k) or (k == "M_base" and tnm.get("M")))],
            "warnings": ["missing_TNM"],
            "trace": [],
        }
    sg, warnings, trace = stage_group_from_TNM(str(T), str(N), str(M_base))
    return {"stage_group": sg, "warnings": warnings, "trace": trace, "version": "ajcc8_csco2025_mvp_v2"}


if __name__ == "__main__":
    # Run MCP server (usually via stdio).
    # Depending on your MCP runner:
    #   python melanoma_staging_mcp_server.py
    # or:
    #   mcp run melanoma_staging_mcp_server.py
    mcp.run()
