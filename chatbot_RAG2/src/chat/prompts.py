"""
Prompt templates for LLM interactions
"""
from typing import List, Dict


class PromptTemplates:
    """Collection of prompt templates for medical RAG system"""

    @staticmethod
    def entity_extraction_prompt(text: str) -> str:
        """
        Prompt for extracting medical entities from user query

        Args:
            text: User query text

        Returns:
            Formatted prompt for entity extraction
        """
        return f"""<|im_start|>system
/no_think
你是一个医学实体识别助手。从问题中提取疾病、药物和人群特征，以JSON格式返回。如果有多个药物，用逗号分隔。<|im_end|>
<|im_start|>user
从以下问题中提取实体：{text}

返回JSON格式：{{"disease": "疾病名或null", "drug": "药物名或null", "population": "人群特征或null"}}

注意：只返回JSON，不要有其他内容。<|im_end|>
<|im_start|>assistant
"""

    @staticmethod
    def answer_generation_prompt(query: str, context_text: str, num_refs: int) -> str:

#         return f"""<|im_start|>system
# /think
# 你是一个专业的医学问答助手。请基于提供的医学文献，结合用户问题，进行深度的分析和思考，然后回答用户问题。

# 重要要求：

# 1. 提供详细、全面、深入的回答，充分整合所有相关文献的观点和数据
# 2. 每个重要论述、数据、建议都必须用方括号标注来源，格式为[1][2]而非[1,2]，多引用不同文献以增强论证
# 3. 回答要包含足够的细节和具体信息，使内容更加充实和专业
# 4. 禁止在正文中写"文献1"、"文献2和8"这类表述，只能在句尾用[1]、[2][8]标注
# 5. 如需提及来源时，用"相关研究"、"多项研究"等通用表述
# 6. 可以适当展开说明机制、原因、注意事项等，让回答更有价值

# 正确示例：相关研究表明这种方法有效[1][2]
# 错误示例：文献1和2表明这种方法有效[1][2]<|im_end|>
# <|im_start|>user
# 以下是{num_refs}篇相关医学文献：

# {context_text}

# 问题：{query}

# 请给出详细、全面的专业回答，充分利用所有相关文献信息。重要：引用格式为[1][2]不是[1,2]。严禁在正文写"文献X"，只能在句尾用方括号标注。<|im_end|>
# <|im_start|>assistant
# """
    
#         return f"""<|im_start|>system
# /think
# 你是一个专业的医学问答助手。请仅依据我提供的医学文献内容，围绕用户问题进行有条理的推理与整合后作答。

# 重要要求：
# 1.先给出结论清单（编号）：结论要简洁明确、带方向性；各结论之间需体现因果/递进/互补关系；每条结论可独立成立但彼此支撑。
# 2.再给出详细解析：按段落展开，每段只讨论一个逻辑主题（如适用人群、疗效证据、安全性、方案选择、证据质量与局限等），层次清晰。
# 3.数字使用规则：回答中凡出现具体数字（比例、风险、剂量、时间、样本量、P 值等），必须在同一句或句末给出对应来源；若无法提供来源，则不得出现该数字。
# 4.内容深度与覆盖：回答需全面、深入、可落地，充分整合所有相关文献的观点、结论与数据，并指出一致与分歧之处。
# 5.引用格式：关键论述、数据与建议必须用方括号标注来源，格式为 [1][2]（不得写成 [1,2]）；尽量交叉引用多篇文献以增强论证力度。
# 6.正文表述限制：正文中禁止出现“文献1”“文献2和8”等说法；只能在句末用 [1]、[2][8] 标注。
# 7.提及来源的措辞：需要概括来源时，用“相关研究”“多项研究”“指南建议”等通用表述，不点名编号或作者。

# 正确示例：相关研究表明这种方法有效[1][2]
# 错误示例：文献1和2表明这种方法有效[1][2]<|im_end|>
# <|im_start|>user
# 以下是{num_refs}篇相关医学文献：

# {context_text}

# 问题：{query}

# 请给出详细、全面的专业回答，充分利用所有相关文献信息。重要：引用格式为[1][2]不是[1,2]。严禁在正文写"文献X"，只能在句尾用方括号标注。<|im_end|>
# <|im_start|>assistant
# """
        return f"""<|im_start|>system
/think
你是一名专业的医学问答助手。请基于检索到的医学文献，在严格依托证据的前提下，回答用户问题。

重要要求：
1.先给结论（要点式回答）
•开头先总结 2–5 条编号结论（如：结论1、结论2……）。
•每条结论要简洁明确，体现因果或递进关系：
•既可以单独成立，又能互相支撑。
•结论中如出现任何数字（如比例、风险、疗效提高多少等），必须有对应文献来源标注。
2.再给详细解释（分段论证）
•在结论之后，写“详细解析”部分。
•每一段只聚焦一个逻辑点：例如
•机制与原理
•具体证据数据（不同研究的结果）
•适用人群与限制条件
•临床应用建议（在证据范围内）
•详细解释中如出现数字，同样必须带文献标注，否则不要写具体数字。
3.数字必须有证据来源
•所有数字，包括：发生率、相对风险、HR/OR/RR、百分比、生存率、剂量、时间等，都必须直接来自给定文献。
•如果文献没有具体数字，只能用定性表述（如“更高风险”“明显增加”“轻度改善”），不要虚构任何数值。
4.文献引用格式与用法
•引用格式统一用方括号，例如：[1]、[2][5]。
•同一句话可引用多篇文献，用相邻方括号表示，如：[1][3][7]。
•禁止使用“文献1”“文献2和8”这类说法，只能写“相关研究”“多项研究”“随访研究”等通用表述，然后在句尾用 [1] 或 [1][2][8] 标注。
•对于重要论点、关键数据和核心建议，尽量引用多篇不同文献以增强可信度。
5.整合多篇文献，给出临床可用答案
•尽量综合所有相关文献的结果，指出：
•一致结论
•有分歧的地方
•证据的强弱和局限
•回答要详细、全面、深入，但保持结构清晰，方便临床医生或科研人员直接使用。
•不得超出文献证据随意推断结论，也不得给出超出证据范围的“绝对性结论”。

正确示例：相关研究表明这种方法有效[1][2]
错误示例：文献1和2表明这种方法有效[1][2]<|im_end|>
<|im_start|>user
以下是当前数据库中检索到的相关医学文献：

{context_text}

问题：{query}

请给出详细、全面的专业回答，充分利用所有相关文献信息，不显示检索到的文献数量信息。重要：引用格式为[1][2]不是[1,2]。严禁在正文写"文献X"，只能在句尾用方括号标注。<|im_end|>
<|im_start|>assistant
"""
    @staticmethod
    def cot_analysis_prompt(query: str) -> str:
        """
        Prompt for Chain-of-Thought query analysis

        Args:
            query: User query

        Returns:
            Formatted prompt for CoT analysis
        """
        return f"""<|im_start|>system
/no_think
你是一个医学问答系统的分析模块。请判断用户的问题是否与医学相关。

医学相关问题包括：疾病、症状、诊断、治疗、药物、医疗程序、健康状况等。
非医学问题包括：天气、新闻、娱乐、体育、技术、日常生活等。

以JSON格式返回分析结果。<|im_end|>
<|im_start|>user
用户问题：{query}

请分析并返回JSON格式：
{{
  "is_medical": true/false,
  "reasoning": "判断理由"
}}<|im_end|>
<|im_start|>assistant
"""

    @staticmethod
    def simple_chat_prompt(message: str) -> str:
        """
        Prompt for simple chat without context

        Args:
            message: User message

        Returns:
            Formatted prompt for simple chat
        """
        return f"""你是一个专业的医学助手。请回答以下问题：

{message}

回答："""

    @staticmethod
    def format_context_for_prompt(context: List[Dict], max_chars: int = 800, max_docs: int = 15) -> str:
        """
        Format retrieved context documents for inclusion in prompt

        Args:
            context: List of retrieved documents with content
            max_chars: Maximum characters per document
            max_docs: Maximum number of documents to include

        Returns:
            Formatted context string
        """
        context_text = "\n\n".join([
            f"文献 {i+1}:\n{doc.get('content', '')[:max_chars]}"
            for i, doc in enumerate(context[:max_docs])
        ])

        return context_text.strip() or "暂无相关文献。"

    @staticmethod
    def get_stop_sequences() -> List[str]:
        """
        Get stop sequences for LLM generation

        Returns:
            List of stop sequences
        """
        return [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>"
        ]


class RefusalMessages:
    """Collection of refusal messages for non-medical queries"""

    @staticmethod
    def non_medical_query(reasoning: str = "") -> str:
        """
        Generate refusal message for non-medical queries

        Args:
            reasoning: Optional reasoning for refusal

        Returns:
            Formatted refusal message
        """
        return """抱歉，我是一个专业的医学问答助手，只能回答医学相关的问题。

您的问题似乎不涉及医学内容。如果您有医学方面的疑问（如疾病、药物、治疗等），我很乐意为您解答。"""

    @staticmethod
    def model_not_loaded() -> str:
        """Message when model is not loaded"""
        return "抱歉，语言模型未加载。请先运行 setup_models.py 下载模型。"

    @staticmethod
    def system_not_initialized() -> str:
        """Message when system is not initialized"""
        return "抱歉，系统未能正确初始化。请检查模型是否已下载。\n\n运行 `python scripts/setup_models.py` 来下载所需模型。"
