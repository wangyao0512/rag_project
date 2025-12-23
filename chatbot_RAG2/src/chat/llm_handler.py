"""
LLM handler using OpenAI-compatible Qwen HTTP API
"""
from typing import Dict, List, Optional, Generator, Union, Any
import json
import re
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from config import Config
from src.chat.prompts import PromptTemplates, RefusalMessages
from src.prompts.two_stage_prompts import DECIDER_PROMPT, MELANOMA_STAGING_EXTRACTION_PROMPT
from src.retrieval.evidence_models import EvidenceBundle

# 从全局配置里拿参数
DEFAULT_CHAT_BASE_URL = Config.CHAT_BASE_URL
DEFAULT_CHAT_MODEL = Config.CHAT_MODEL
DEFAULT_API_KEY = Config.CHAT_API_KEY
DEFAULT_ENTITY_BASE_URL = Config.ENTITY_BASE_URL
DEFAULT_ENTITY_MODEL = Config.ENTITY_MODEL


class QwenLLM:
    """
    对外接口保持不变：
    - QwenLLM(...) 仍然可以接收 model_path 等参数（为了兼容老代码），但**完全不用本地文件**
    - 公开方法：extract_entities, generate_answer
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        entity_model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        chat_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        entity_base_url: Optional[str] = None,
        entity_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # 这些是兼容用的属性，不再真正使用文件
        self.model_path = model_path
        self.entity_model_path = entity_model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        base_url = chat_base_url or DEFAULT_CHAT_BASE_URL
        api_key = api_key or DEFAULT_API_KEY
        chat_model = chat_model or DEFAULT_CHAT_MODEL

        if entity_base_url is None:
            entity_base_url = DEFAULT_ENTITY_BASE_URL or base_url
        if entity_model is None:
            entity_model = DEFAULT_ENTITY_MODEL or chat_model

        logger.info(
            f"Initializing QwenLLM with OpenAI-compatible API: "
            f"chat_base_url={base_url}, chat_model={chat_model}, entity_model={entity_model}"
        )

        # 主模型 & 实体抽取模型（可以用同一个）
        self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=chat_model)
        self.entity_llm = ChatOpenAI(base_url=entity_base_url, api_key=api_key, model=entity_model)

        # 医学实体规范化映射（保留原来）
        self.entity_mappings = {
            "糖尿病": "Diabetes mellitus",
            "2型糖尿病": "Type 2 diabetes",
            "高血压": "Hypertension",
            "心肌梗死": "Myocardial infarction",
            "心梗": "Myocardial infarction",
            "冠心病": "Coronary heart disease",
            "脑卒中": "Stroke",
            "中风": "Stroke",
            "老年": "Elderly",
            "儿童": "Pediatric",
            "孕妇": "Pregnant",
        }
    # ==============PICO识别============
# ============ PICO 抽取 ============

    def extract_pico(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract PICO from query text.

        Returns:
            {"P": str|None, "I": str|None, "C": str|None, "O": str|None}
        """
        if not self.entity_llm and not self.llm:
            logger.warning("No LLM loaded, returning empty PICO")
            return {"P": None, "I": None, "C": None, "O": None}

        try:
            raw = self._run_chain(
                RunnableLambda(lambda value: self._pico_extraction_prompt(value))
                | (self.entity_llm or self.llm).bind(
                    max_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["<|im_end|>", "<|im_start|>", "\n\n"],
                    repeat_penalty=1.05
                )
                | StrOutputParser(),
                text,
            ).strip()

            pico = self._parse_pico_json(raw)
            if pico is not None:
                return pico

            # 解析失败：尝试二次“纠错输出”提示（仍然不走 stream）
            logger.warning("PICO JSON parse failed, trying repair prompt...")
            repaired_raw = self._run_chain(
                RunnableLambda(lambda value: self._pico_repair_prompt(value))
                | (self.entity_llm or self.llm).bind(
                    max_tokens=256,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["<|im_end|>", "<|im_start|>", "\n\n"],
                    repeat_penalty=1.0
                )
                | StrOutputParser(),
                raw,
            ).strip()
            pico = self._parse_pico_json(repaired_raw)
            if pico is not None:
                return pico

            # 最后 fallback：规则抽取（尽量给出 I/C/O/P 的关键词）
            logger.warning("PICO extraction failed after repair, using fallback rules.")
            return self._fallback_pico_extraction(text)

        except Exception as e:
            logger.error(f"Error in PICO extraction: {e}")
            return self._fallback_pico_extraction(text)
    def _pico_extraction_prompt(self, text: str) -> str:
        # 这里不依赖你 prompts.py 的模板，避免你还要改 PromptTemplates
        # 要求：严格 JSON + 只输出 JSON
        return f"""
你是医学信息抽取助手。请从“用户问题”中抽取 PICO（Population/Intervention/Comparison/Outcome）。
只输出**严格 JSON**，不要输出任何解释、不要 Markdown、不要代码块。
【PICO框架定义】
- P (Population/Patient/Problem)：人群、患者或医学问题描述，例如“2型糖尿病患者”或“中年女性高血压患者”。
- I (Intervention/Exposure/Diagnostic Test)：干预措施、暴露或检验检查，例如“阿司匹林治疗”或“疫苗接种”。
- C (Comparator)：比较组，例如“安慰剂”、“标准治疗”或“无干预”。如果隐含（如“随机对照试验”中默认有对照组），请积极推断。
- O (Outcome)：有比较意义的结局，例如“心血管事件发生率降低”或“生存率改善”。
规则：
- 必须输出 4 个键：P, I, C, O
- 值为字符串或 null
- 不要输出数组；多个内容用中文逗号连接成一个字符串
- 不确定就填 null
- 如果一个片段缺少某个PICO元素（如C），则输出该元素为 null
- 严格按照PICO定义，如果不存在PICO则输出null


输出格式示例：
{{"P":"...","I":"...","C":null,"O":"..."}}

用户问题：{text}
""".strip()

    def _pico_repair_prompt(self, raw: str) -> str:
        # 专门用来把“非严格 JSON”纠正成严格 JSON
        return f"""
将下面内容纠正为**严格 JSON**，且只输出 JSON：
- 必须包含键 P, I, C, O
- 值为字符串或 null
- 不要任何额外文本

原始内容：
{raw}
""".strip()

    def _parse_pico_json(self, raw: str) -> Optional[Dict[str, Optional[str]]]:
        """
        Robust JSON parsing:
        - strip code fences
        - extract first {{...}} block
        - normalize keys to P/I/C/O
        - normalize null-ish values
        """
        if not raw:
            return None

        # 去掉 ```json ... ``` 之类
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 尝试直接 loads
        obj = None
        try:
            obj = json.loads(cleaned)
        except Exception:
            # 提取第一个 JSON 对象块
            m = re.search(r"\{{.*\}}", cleaned, flags=re.DOTALL)
            if not m:
                return None
            candidate = m.group(0)
            try:
                obj = json.loads(candidate)
            except Exception:
                return None

        if not isinstance(obj, dict):
            return None

        # key 规范化
        out: Dict[str, Optional[str]] = {"P": None, "I": None, "C": None, "O": None}
        for k, v in obj.items():
            kk = str(k).strip().upper()
            if kk not in out:
                continue

            if v is None:
                out[kk] = None
                continue

            # 列表转字符串（虽然我们 prompt 禁止，但模型可能会输出）
            if isinstance(v, list):
                vv = "，".join([str(x).strip() for x in v if x is not None and str(x).strip()])
            else:
                vv = str(v).strip()

            if not vv or vv.lower() in ("null", "none", "n/a", "na", "未知", "不确定", "无"):
                out[kk] = None
            else:
                out[kk] = vv

        return out

    def _fallback_pico_extraction(self, text: str) -> Dict[str, Optional[str]]:
        """
        Very lightweight rule-based fallback.
        目标：宁可保守返回 null，也不要乱填。
        """
        t = text.strip()

        # 粗糙抽 P：人群/患者
        P = None
        m = re.search(r"(.{{0,20}}?(患者|人群|病人|成人|儿童|老年|孕妇).{{0,20}}?)", t)
        if m:
            P = m.group(1).strip()

        # 粗糙抽 I/C：常见比较句式
        I = None
        C = None
        # “A vs B”
        m = re.search(r"(.+?)\s*(vs|VS|对比|比较|相比)\s*(.+)", t)
        if m:
            I = m.group(1).strip()
            C = m.group(3).strip()
        else:
            # “使用/给予/治疗/用药 …”
            m = re.search(r"(使用|给予|治疗|用药|应用|采用)(.{{1,30}})", t)
            if m:
                I = m.group(2).strip()

        # 粗糙抽 O：结局关键词
        O = None
        outcome_keywords = ["疗效", "生存", "OS", "PFS", "复发", "缓解", "不良反应", "毒性", "死亡", "风险", "并发症", "效果", "结局", "预后"]
        for kw in outcome_keywords:
            if kw in t:
                O = kw
                break

        return {"P": P, "I": I, "C": C, "O": O}

 

    # ============ 实体抽取 ============

    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract medical entities from text using dedicated entity model

        Args:
            text: User query text

        Returns:
            Dictionary with disease, drug, and population entities
        """
        # Use entity model if available, otherwise use main model
        if not self.entity_llm and not self.llm:
            logger.warning("No LLM loaded, returning empty entities")
            return {"disease": None, "drug": None, "population": None}

        try:
            result_text = self._run_chain(
                RunnableLambda(lambda value: PromptTemplates.entity_extraction_prompt(value))
                | (self.entity_llm or self.llm).bind(
                    max_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["<|im_end|>", "<|im_start|>", "\n\n"],
                    repeat_penalty=1.1
                )
                | StrOutputParser(),
                text,
            ).strip()

            # Try to parse JSON
            try:
                entities = json.loads(result_text)
                # Ensure all values are strings or None (convert lists to comma-separated strings)
                for key in ['disease', 'drug', 'population']:
                    if key in entities:
                        if isinstance(entities[key], list):
                            # Convert list to comma-separated string
                            entities[key] = ','.join([str(v) for v in entities[key] if v and str(v).lower() != 'null'])
                        elif entities[key] and str(entities[key]).lower() == 'null':
                            entities[key] = None
                return entities
            except Exception as e:
                # Fallback: simple entity extraction
                logger.warning(f"Failed to parse JSON from LLM: {e}, using fallback")
                return self._fallback_entity_extraction(text)

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return self._fallback_entity_extraction(text)

    def _fallback_entity_extraction(self, text: str) -> Dict[str, Optional[str]]:
        """
        Fallback entity extraction using simple rules

        Args:
            text: User query text

        Returns:
            Dictionary with extracted entities using keyword matching
        """
        # Simple keyword matching for common medical terms
        entities: Dict[str, Optional[str]] = {"disease": None, "drug": None, "population": None}

        # Common diseases
        diseases = ["2型糖尿病", "糖尿病", "高血压", "心脏病", "心梗", "心肌梗死", "中风", "感冒", "发烧", "乙肝"]
        for disease in diseases:
            if disease in text:
                entities["disease"] = disease
                break

        # Common drugs - check for multiple drugs
        drugs = ["二甲双胍", "格列美脲", "阿司匹林", "华法林", "降压药", "降糖药", "乙肝疫苗"]
        found_drugs = []
        for drug in drugs:
            if drug in text:
                found_drugs.append(drug)

        if found_drugs:
            entities["drug"] = ",".join(found_drugs)

        # Common populations
        populations = ["老年", "儿童", "孕妇", "成人", "患者"]
        for pop in populations:
            if pop in text:
                entities["population"] = pop
                break

        return entities

    # ============ 生成回答（RAG 用） ============

    def generate_answer(self, query: str, context: List[Dict], max_tokens: int = 3072,
                       stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate answer based on retrieved context

        Args:
            query: User query
            context: List of retrieved context documents
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated answer as string or generator
        """
        if stream:
            return self._generate_answer_stream(query, context, max_tokens)
        else:
            return self._generate_answer_sync(query, context, max_tokens)

    def _generate_answer_sync(self, query: str, context: List[Dict], max_tokens: int = 4096) -> str:
        """
        Synchronous answer generation (non-streaming)

        Args:
            query: User query
            context: List of retrieved context documents
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer as string
        """
        if not self.llm:
            logger.warning("LLM not loaded, returning default message")
            return RefusalMessages.model_not_loaded()

        prompt = self._build_prompt(query, context)
        stop_sequences = PromptTemplates.get_stop_sequences()

        try:
            answer = self._invoke_chat(
                self.llm,
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                stop=stop_sequences,
                repeat_penalty=1.1
            ).strip()
            # Clean up any remaining stop sequences
            for stop in stop_sequences:
                answer = answer.split(stop)[0]
            # Remove Qwen3 thinking tags
            answer = self._clean_qwen3_output(answer)

            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def _generate_answer_stream(self, query: str, context: List[Dict], max_tokens: int = 3072) -> Generator[str, None, None]:
        """
        Streaming answer generation (generator)

        Args:
            query: User query
            context: List of retrieved context documents
            max_tokens: Maximum tokens to generate

        Yields:
            Generated tokens
        """
        if not self.llm:
            logger.warning("LLM not loaded, returning default message")
            yield RefusalMessages.model_not_loaded()
            return

        prompt = self._build_prompt(query, context)
        stop_sequences = PromptTemplates.get_stop_sequences()

        try:
            # Simply yield all tokens - we'll clean think tags from the accumulated text in the app
            token_count = 0
            for token in self._stream_chat(
                self.llm,
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                stop=stop_sequences,
                repeat_penalty=1.1
            ):
                token_count += 1

                # Just yield the token - llama.cpp will handle stop sequences
                yield token

            logger.info(f"LLM generated {token_count} tokens")

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            yield f"抱歉，生成回答时出现错误：{str(e)}"

    # ============ Prompt 构造 & 清理 ============

    def _build_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Build prompt from query and context

        Args:
            query: User query
            context: List of retrieved context documents

        Returns:
            Formatted prompt string
        """
        # Format context using template
        context_text = PromptTemplates.format_context_for_prompt(context, max_chars=800, max_docs=15)
        num_refs = len([doc for doc in context[:15] if doc.get('content')])

        return PromptTemplates.answer_generation_prompt(query, context_text, num_refs)
    def _clean_qwen3_output(self, text: str) -> str:
        """
        Remove Qwen3 thinking tags from output

        Args:
            text: Text potentially containing thinking tags

        Returns:
            Cleaned text
        """
        # Remove <think>...</think> blocks (including newlines inside)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        text = re.sub(r'\n\n\n+', '\n\n', text)
        return text.strip()
    def generate_decider_answer(self,
                                question: str,
                                evidence_bundle: EvidenceBundle,
                                max_tokens: int = 2048,
                                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate answer using the structured EvidenceBundle (two-stage path).

        Args:
            question: Original user question
            evidence_bundle: Structured evidence bundle from Reader
            max_tokens: Generation cap
            stream: Whether to stream tokens
        """
        if not self.llm:
            logger.warning("LLM not loaded, returning default message")
            return RefusalMessages.model_not_loaded()

        prompt = self._build_decider_prompt(question, evidence_bundle)
        stop_sequences = PromptTemplates.get_stop_sequences()

        if stream:
            return self._stream_with_prompt(prompt, max_tokens, stop_sequences)

        try:
            answer = self._invoke_chat(
                self.llm,
                prompt,
                max_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9,
                stop=stop_sequences,
                repeat_penalty=1.05
            ).strip()
            for stop in stop_sequences:
                answer = answer.split(stop)[0]
            return self._clean_qwen3_output(answer)
        except Exception as e:
            logger.error(f"Error generating decider answer: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def _stream_with_prompt(self, prompt: str, max_tokens: int, stop_sequences: List[str]) -> Generator[str, None, None]:
        """Helper for streaming a custom prompt."""
        try:
            for token in self._stream_chat(
                self.llm,
                prompt,
                max_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9,
                stop=stop_sequences,
                repeat_penalty=1.05
            ):
                yield token
        except Exception as e:
            logger.error(f"Error in streaming decider answer: {e}")
            yield f"抱歉，生成回答时出现错误：{str(e)}"

    def _build_decider_prompt(self, question: str, evidence_bundle: EvidenceBundle) -> str:
        """Format EvidenceBundle into the decider prompt."""
        def _fmt_card(card) -> str:
            return (
                f"EID: {card.evidence_id}\n"
                f"来源: {card.source_type or ''} - {card.reference or ''} ({card.year or '未知'})\n"
                f"P: {card.pico.get('P')}\n"
                f"I: {card.pico.get('I')}\n"
                f"C: {card.pico.get('C')}\n"
                f"O: {card.pico.get('O')}\n"
                f"GRADE: {card.grade or ''}\n"
                f"关键结论: {card.key_finding or ''}\n"
                f"效应量: {card.effect_size_text or ''}\n"
                f"适用性: P匹配 {card.applicability.population_match}, I匹配 {card.applicability.intervention_match}\n"
                f"安全性: {card.safety_note or ''}\n"
                f"局限: {card.limitations or ''}\n"
            )

        primary_block = "\n\n".join(_fmt_card(c) for c in evidence_bundle.primary_evidence) or "无"
        other_cards = evidence_bundle.supporting_evidence + evidence_bundle.conflicting_evidence
        other_block = "\n\n".join(_fmt_card(c) for c in other_cards) or "无"
        overview_text = (
            f"{evidence_bundle.evidence_overview.summary or ''}\n"
            f"一致性: {evidence_bundle.evidence_overview.consistency or '未知'}; "
            f"高GRADE: {evidence_bundle.evidence_overview.num_high_grade}; "
            f"中GRADE: {evidence_bundle.evidence_overview.num_moderate_grade}"
        ).strip()

        patient_profile = evidence_bundle.patient_profile.model_dump() if hasattr(evidence_bundle.patient_profile, "model_dump") else {}

        return DECIDER_PROMPT.format(
            question_text=question,
            patient_profile=patient_profile,
            evidence_overview=overview_text,
            primary_evidence=primary_block,
            other_evidence=other_block,
        )

    def extract_melanoma_staging_fields(self, question: str) -> Dict[str, Any]:
        """
        Extract structured melanoma staging fields from a question.
        Falls back to defaults if parsing fails.
        """
        if not self.llm:
            return self._default_staging_payload()

        prompt = MELANOMA_STAGING_EXTRACTION_PROMPT.format(question=question)
        stop_sequences = PromptTemplates.get_stop_sequences()
        try:
            text = self._invoke_chat(
                self.llm,
                prompt,
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
                stop=stop_sequences,
                repeat_penalty=1.05
            ).strip()
            # Clean any stop tokens
            for stop in stop_sequences:
                text = text.split(stop)[0]
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to extract melanoma staging fields: {e}")
            return self._default_staging_payload()

    @staticmethod
    def _default_staging_payload() -> Dict[str, Any]:
        return {
            "context": {"mode": "mixed", "language": "zh"},
            "primary_tumor": {
                "primary_present": "unknown",
                "in_situ": "unknown",
                "breslow_mm": None,
                "ulceration": "unknown"
            },
            "regional_nodes": {
                "nodes_positive_count": None,
                "clinically_apparent_nodes": None,
                "matted_nodes": "unknown",
                "in_transit_satellite_microsatellite": "unknown"
            },
            "distant_metastasis": {
                "status": "unknown",
                "site": "unknown",
                "ldh": "unknown"
            }
        }

    @staticmethod
    def _invoke_chat(
        llm: ChatOpenAI,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        repeat_penalty: Optional[float],
    ) -> str:
        params: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            params["stop"] = stop
        if repeat_penalty is not None:
            params["repeat_penalty"] = repeat_penalty
        response = llm.invoke(prompt, **params)
        return response.content or ""

    @staticmethod
    def _stream_chat(
        llm: ChatOpenAI,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        repeat_penalty: Optional[float],
    ) -> Generator[str, None, None]:
        params: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            params["stop"] = stop
        if repeat_penalty is not None:
            params["repeat_penalty"] = repeat_penalty
        for chunk in llm.stream(prompt, **params):
            token = chunk.content or ""
            if token:
                yield token

    @staticmethod
    def _run_chain(chain, payload: str) -> str:
        return chain.invoke(payload)
