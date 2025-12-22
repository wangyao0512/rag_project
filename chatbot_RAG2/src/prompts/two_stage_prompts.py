"""
Prompt templates for two-stage Reader/Decider RAG flow.

These are text scaffolds to be used by lighter-weight Reader LLM calls and the
final Decider call. They are designed to encourage evidence synthesis rather
than verbatim repetition to avoid distribution collapse.
"""
from textwrap import dedent


READER_CARD_PROMPT = dedent(
    """
    你将看到一条 PICO 结构化医疗证据，以及当前患者/问题信息。
    请你根据 PICO 和原始摘要，生成一条紧凑的“证据卡片”。

    【当前问题】
    {normalized_question}

    【患者信息】
    {patient_profile}

    【证据 PICO】
    P: {P}
    I: {I}
    C: {C}
    O: {O}

    【研究摘要】
    {summary}
    【若有原文片段】
    {raw_excerpt}

    请输出 JSON，字段包括：
    - key_finding: 用1-2句总结该证据对当前问题最关键的结论
    - effect_size_text: 用自然语言简要描述效应量（如“HR≈0.65，复发风险降低约35%”）
    - safety_note: 与当前问题相关的安全性要点
    - limitations: 该研究在外推到当前患者时的主要局限
    - applicability.population_match: 0~1 之间的数
    - applicability.intervention_match: 0~1 之间的数
    - applicability.comorbidity_risk_note: 文字说明
    """
).strip()


DECIDER_PROMPT = dedent(
    """
    你是一名循证医学导向的智能助手，面向临床医生回答问题。
    你只能基于提供的 EvidenceBundle 和你自身的医学常识进行推理，
    需要明确说明证据支持程度和不确定性。

    【医生原始问题】
    {question_text}

    【患者信息】
    {patient_profile}

    【证据总览】
    {evidence_overview}

    【主要证据】
    {primary_evidence}

    【辅助与冲突证据】
    {other_evidence}

    请按以下步骤思考并输出（用中文回答）：
    1. 先简要总结：对当前患者，这些证据整体支持哪些治疗策略？是否存在明显冲突？
    2. 给出你对医生问题的直接回答，面向有一定专业背景的临床医生，语言简洁但专业。
    3. 说明你的结论主要依据哪些证据（引用 EID，如 E2025-0001），以及这些证据的 GRADE。
    4. 如果证据之间存在冲突或不足，请明确指出不确定性来源，并建议“需要多学科讨论 / 查阅最新指南 / 个体化评估”等。
    5. 避免逐字重复证据卡片的内容，而是用你的语言进行综合和解释。
    """
).strip()

# Simple extractor prompt for melanoma staging fields
MELANOMA_STAGING_EXTRACTION_PROMPT = dedent(
    """
    从下列用户问题中提取与黑色素瘤分期相关的结构化字段，直接输出纯JSON（不要额外文字）。
    输出字段：
    {{
      "context": {{"mode": "mixed", "language": "zh"}},
      "primary_tumor": {{
        "primary_present": "yes/no/unknown",
        "in_situ": "yes/no/unknown",
        "breslow_mm": float or null,
        "ulceration": "present/absent/unknown"
      }},
      "regional_nodes": {{
        "nodes_positive_count": int or null,
        "clinically_apparent_nodes": int or null,
        "matted_nodes": "yes/no/unknown",
        "in_transit_satellite_microsatellite": "present/absent/unknown"
      }},
      "distant_metastasis": {{
        "status": "none/present/unknown",
        "site": "skin_soft_tissue/nonregional_ln/lung/other_viscera/cns/unknown",
        "ldh": "normal/elevated/unknown"
      }}
    }}
    如果问题未提及某字段，用 null 或 "unknown"。

    json格式举例：
    {{
      "context": {{"mode":"mixed","language":"zh"}},
      "primary_tumor": {{
        "primary_present":"yes",
        "in_situ":"no",
        "breslow_mm": 2.4,
        "ulceration":"present"
      }},
      "regional_nodes": {{
        "nodes_positive_count": 1,
        "clinically_apparent_nodes": 0,
        "matted_nodes":"no",
        "in_transit_satellite_microsatellite":"absent"
      }},
      "distant_metastasis": {{
        "status":"none",
        "site":"unknown",
        "ldh":"unknown"
      }}
    }}

    用户问题:
    {question}
    """
).strip()
