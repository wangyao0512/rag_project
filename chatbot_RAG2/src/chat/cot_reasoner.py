"""
Chain-of-Thought Reasoner for Medical Query Classification
"""
from typing import Dict, Optional
from loguru import logger
import json
from src.chat.prompts import PromptTemplates, RefusalMessages


class CoTReasoner:
    """Analyzes user queries and determines optimal response strategy"""

    def __init__(self, llm):
        self.llm = llm

    def analyze_query(self, query: str) -> Dict:
        """
        Perform Chain-of-Thought analysis on user query to check if it's medical-related

        Returns:
        {
            'is_medical': bool,
            'reasoning': str
        }
        """
        if not self.llm.llm:
            logger.warning("LLM not loaded for CoT reasoning")
            return self._default_analysis()

        prompt = PromptTemplates.cot_analysis_prompt(query)

        try:
            response = self.llm.llm(
                prompt,
                max_tokens=128,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>", "\n\n\n"],
                repeat_penalty=1.1
            )

            result_text = response['choices'][0]['text'].strip()
            original_response = result_text  # Save original for debugging

            # Log the raw response for debugging
            logger.debug(f"Raw CoT response: {result_text}")

            # Try to parse JSON (handle thinking tags and markdown code blocks)
            try:
                # Remove thinking tags if present
                import re
                # Remove <think>...</think> blocks and everything inside them
                result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()

                # Remove markdown code blocks if present
                if result_text.startswith('```'):
                    # Extract JSON from code block
                    lines = result_text.split('\n')
                    json_lines = []
                    for line in lines[1:]:
                        if line.strip() == '```':
                            break
                        json_lines.append(line)
                    result_text = '\n'.join(json_lines).strip()

                # Log the processed text before parsing
                logger.debug(f"Processed CoT text for parsing: {result_text}")
                analysis = self._safe_parse_json(result_text)
                is_medical = analysis.get('is_medical', True)
                reasoning = analysis.get('reasoning', '')
                needs_staging = analysis.get('needs_melanoma_staging', False)
                staging_reason = analysis.get('staging_reason', '')

                # Heuristic: melanoma + treatment intent => staging needed even if LLM missed it
                if not needs_staging and self._heuristic_staging_needed(query):
                    needs_staging = True
                    staging_reason = staging_reason or "黑色素瘤治疗相关问题，需先确认分期"

                logger.info(f"CoT Analysis: is_medical={is_medical}, reasoning={reasoning}, needs_staging={needs_staging}")
                return {
                    'is_medical': is_medical,
                    'reasoning': reasoning,
                    'needs_melanoma_staging': needs_staging,
                    'staging_reason': staging_reason
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse CoT JSON: {e}. Original: '{original_response[:200]}'. Processed: '{result_text[:200]}'")
                return self._fallback_analysis(query)

        except Exception as e:
            logger.error(f"Error in CoT analysis: {e}")
            return self._fallback_analysis(query)

    def _fallback_analysis(self, query: str) -> Dict:
        """Rule-based fallback when LLM analysis fails"""
        heuristic_staging = self._heuristic_staging_needed(query)
        # Check if non-medical
        non_medical_keywords = ['天气', '新闻', '股票', '电影', '音乐', '游戏', '美食', '体育', '娱乐']
        if any(kw in query for kw in non_medical_keywords):
            return {
                'is_medical': False,
                'reasoning': '问题不涉及医学内容',
                'needs_melanoma_staging': False,
                'staging_reason': ''
            }

        # Medical keywords
        medical_keywords = ['疾病', '药', '治疗', '症状', '诊断', '患者', '血压', '糖尿病', '医生', '健康', '病']
        is_medical = any(kw in query for kw in medical_keywords)

        if is_medical:
            return {
                'is_medical': True,
                'reasoning': '检测到医学相关关键词',
                'needs_melanoma_staging': heuristic_staging,
                'staging_reason': '黑色素瘤治疗相关问题，需先确认分期' if heuristic_staging else ''
            }
        else:
            return {
                'is_medical': False,
                'reasoning': '未检测到医学相关关键词',
                'needs_melanoma_staging': False,
                'staging_reason': ''
            }
    @staticmethod
    def _safe_parse_json(text: str) -> Dict:
        """
        Try to parse JSON; on failure return minimal defaults.
        """
        try:
            return json.loads(text)
        except Exception:
            return {
                'is_medical': True,
                'reasoning': '',
                'needs_melanoma_staging': False,
                'staging_reason': ''
            }
    def _default_analysis(self) -> Dict:
        """Default analysis when LLM not available"""
        return {
            'is_medical': True,
            'reasoning': 'LLM未加载，默认允许查询'
        }

    def should_refuse(self, analysis: Dict) -> bool:
        """Check if query should be refused (not medical-related)"""
        return not analysis.get('is_medical', True)

    def get_refusal_message(self, analysis: Dict) -> str:
        """Generate appropriate refusal message"""
        reasoning = analysis.get('reasoning', '')
        return RefusalMessages.non_medical_query(reasoning)
    def _heuristic_staging_needed(self, query: str) -> bool:
        """Detect melanoma treatment intent that should trigger staging."""
        lowered = query.lower()
        melanoma_hit = ("黑色素瘤" in query) or ("melanoma" in lowered)
        treatment_keywords = ["治疗", "方案", "用药", "手术", "免疫", "靶向", "化疗", "放疗", "辅助", "adjuvant", "neoadjuvant", "metastatic", "复发", "转移"]
        treatment_hit = any(k in lowered for k in treatment_keywords)
        return melanoma_hit and treatment_hit
