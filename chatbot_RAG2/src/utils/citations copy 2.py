"""
Citation management utilities for medical references
"""
from typing import List, Dict, Tuple, Set
import re
import json
import base64
from collections import defaultdict
from loguru import logger
from typing import Dict
import html

def extract_citation_order(text: str) -> List[int]:
    """按出现顺序提取 [N] 引用编号，支持 [1,2] 这种逗号形式"""
    # 先把 [1,2] 这种展开成 [1][2]
    def _expand(match):
        nums = re.findall(r"\d+", match.group(0))
        return "".join([f"[{n}]" for n in nums])

    normalized = re.sub(r"\[\d+(?:,\s*\d+)+\]", _expand, text)

    citations = re.findall(r"\[(\d+)\]", normalized)

    seen: Set[str] = set()
    ordered: List[int] = []
    for cite in citations:
        if cite not in seen:
            seen.add(cite)
            ordered.append(int(cite))
    return ordered


def group_sources_by_article(sources: List[Dict]) -> Dict[str, Dict]:
    """
    Group sources by article, keeping track of paragraph content

    Args:
        sources: List of source dictionaries with content and metadata

    Returns:
        Dictionary mapping article keys to grouped content and metadata
    """
    article_groups = defaultdict(lambda: {'paragraphs': [], 'metadata': None, 'contents': []})

    for source in sources:
        metadata = source.get('metadata', {})
        source_info = metadata.get('source', '未知来源')
        year = metadata.get('year', '')
        original_content = metadata.get('llm_text', '') or source.get('content', '')

        # Create unique article key
        article_key = f"{source_info}|{year}"

        # Add paragraph content
        article_groups[article_key]['paragraphs'].append(source['content'])
        article_groups[article_key]['contents'].append(original_content)

        # Store metadata (same for all paragraphs from same article)
        if article_groups[article_key]['metadata'] is None:
            article_groups[article_key]['metadata'] = {
                'source': source_info,
                'year': year
            }

    return article_groups


def find_most_similar_content(target_content: str, contents: List[str]) -> str:
    """
    Find the content most similar to target_content from a list of contents.
    Uses simple string matching as a proxy for similarity.
    """
    if not contents:
        return ""
    
    if len(contents) == 1:
        return contents[0]
    
    # Calculate similarity scores based on common characters
    best_match = contents[0]
    best_score = 0
    
    for content in contents:
        # Simple similarity measure: count common characters
        common_chars = sum(1 for c1, c2 in zip(target_content, content) if c1 == c2)
        score = common_chars
        
        if score > best_score:
            best_score = score
            best_match = content
    
    return best_match


def format_sources_with_hover(sources: List[Dict], answer_text: str) -> Tuple[str, Dict, Dict]:
    """
    Format sources with hover tooltips, ordered by citation appearance

    Args:
        sources: List of source dictionaries
        answer_text: The answer text containing citations

    Returns:
        Tuple of (formatted_references, hover_data, citation_remap)
    """
    if not sources:
        return "📚 **参考文献：** 无相关文献", {}, {}

    # Group sources by article
    article_groups = group_sources_by_article(sources)

    # Extract citation order from answer
    citation_order = extract_citation_order(answer_text)

    # ===== 关键修复：强制初始化所有映射表 =====
    ordered_sources = []
    seen_articles: Set[str] = set()
    article_to_new_idx: Dict[str, int] = {}
    old_to_new_map: Dict[int, int] = {}  # 清空旧映射
    individual_hover_data: Dict[int, Dict] = {}

    # First pass: Process citations that appear in the answer (in [N] format)
    for old_idx in citation_order:
        if old_idx <= len(sources):
            source = sources[old_idx - 1]
            metadata = source.get('metadata', {})
            source_info = metadata.get('source', '未知来源')
            year = metadata.get('year', '')
            article_key = f"{source_info}|{year}"
            original_content = metadata.get('llm_text', '') or source.get('content', '')

            # Only add each article once to the reference list
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                new_idx = len(ordered_sources) + 1
                article_to_new_idx[article_key] = new_idx
                
                # Select the most representative content for this article
                article_contents = article_groups[article_key]['contents']
                selected_content = find_most_similar_content(original_content, article_contents)
                
                ordered_sources.append({
                    'article_key': article_key,
                    'source': source_info,
                    'year': year,
                    'paragraphs': article_groups[article_key]['paragraphs'],
                    'selected_content': selected_content
                })

            # Map this old citation number to its article's new number
            old_to_new_map[old_idx] = article_to_new_idx[article_key]

            # Store the specific paragraph content for this citation
            individual_hover_data[old_idx] = {
                'source': source_info,
                'year': year,
                'content': original_content
            }

    # Second pass: Map ALL sources (even those not cited in [N] format)
    for old_idx in range(1, len(sources) + 1):
        if old_idx not in old_to_new_map:
            source = sources[old_idx - 1]
            metadata = source.get('metadata', {})
            source_info = metadata.get('source', '未知来源')
            year = metadata.get('year', '')
            article_key = f"{source_info}|{year}"
            original_content = metadata.get('llm_text', '') or source.get('content', '')

            # Get or create article number for this source
            if article_key in article_to_new_idx:
                new_idx = article_to_new_idx[article_key]
                # Update the selected content if this one is better than current one
                current_selected_content = ordered_sources[new_idx - 1].get('selected_content', '')
                article_contents = article_groups[article_key]['contents']
                new_selected_content = find_most_similar_content(original_content, article_contents)
                
                # Check if the new content is better than current one
                if current_selected_content != new_selected_content:
                    ordered_sources[new_idx - 1]['selected_content'] = new_selected_content
            else:
                new_idx = len(ordered_sources) + 1
                article_to_new_idx[article_key] = new_idx
                
                # Select the most representative content for this article
                article_contents = article_groups[article_key]['contents']
                selected_content = find_most_similar_content(original_content, article_contents)
                
                ordered_sources.append({
                    'article_key': article_key,
                    'source': source_info,
                    'year': year,
                    'paragraphs': article_groups[article_key]['paragraphs'],
                    'selected_content': selected_content
                })

            old_to_new_map[old_idx] = new_idx

            # Store paragraph content for tooltip
            individual_hover_data[old_idx] = {
                'source': source_info,
                'year': year,
                'content': original_content
            }

    # Format references
    formatted = "📚 **参考文献：**\n\n"

    for i, article in enumerate(ordered_sources, 1):
        source_info = article['source']
        year = article['year']

        # Format reference line
        if year:
            formatted += f"**[{i}]** {source_info}\n"
        else:
            formatted += f"**[{i}]** {source_info}\n"

    return formatted, individual_hover_data, old_to_new_map

def add_citation_tooltips(text: str, hover_data: Dict, citation_remap: Dict, msg_id: str) -> str:
    """
    将文本中的 [N] 引用替换为带 data-* 的 HTML span。
    ✅ 改进：不再依赖隐藏 div（容易被 gradio sanitize 掉），而是把内容直接写入 data-content-b64。
    ✅ msg_id 隔离仍保留（data-key）。
    """

    def _key(old_num: int) -> str:
        return f"{msg_id}:{old_num}"

    def _replace_wenxian_with_list(match):
        full_text = match.group(0)
        nums = re.findall(r"\d+", full_text)
        remapped = [str(citation_remap.get(int(n), int(n))) for n in nums]

        seen = set()
        uniq = []
        for n in remapped:
            if n not in seen:
                seen.add(n)
                uniq.append(n)

        if len(uniq) == 1:
            return f"文献{uniq[0]}"

        out = full_text
        for old, new in zip(nums, remapped):
            out = out.replace(old, new, 1)
        return out

    def _replace_wenxian_single(match):
        old = int(match.group(1))
        new = citation_remap.get(old, old)
        return f"文献{new}"

    def _expand_comma_citations(match):
        nums = re.findall(r"\d+", match.group(0))
        return "".join([f"[{n}]" for n in nums])

    # 1) 文献X 相关映射
    result = re.sub(r"文献\d+[和、及]\d+", _replace_wenxian_with_list, text)
    result = re.sub(r"文献\d+[和、及]文献\d+", _replace_wenxian_with_list, result)
    result = re.sub(r"文献(\d+)", _replace_wenxian_single, result)
    result = re.sub(r"文献(\d+)[和、及]文献\1", r"文献\1", result)
    result = re.sub(r"文献(\d+)[和、及]\1", r"文献\1", result)

    # 2) [1,2] -> [1][2]
    result = re.sub(r"\[\d+(?:,\s*\d+)+\]", _expand_comma_citations, result)

    # 3) [N] -> <span ... data-content-b64="...">
    last_new = None

    def _make_b64_payload(old_int: int) -> str:
        d = (hover_data or {}).get(old_int) or (hover_data or {}).get(str(old_int)) or {}
        citation_obj = {
            "source": d.get("source", "未知来源"),
            "year": d.get("year", ""),
            "content": d.get("content", "") or "",
        }
        # 可选：避免属性过大（太长会影响渲染/传输）
        max_len = 2000
        if len(citation_obj["content"]) > max_len:
            citation_obj["content"] = citation_obj["content"][:max_len] + "…"

        payload = json.dumps(citation_obj, ensure_ascii=False)
        return base64.b64encode(payload.encode("utf-8")).decode("ascii")

    def _replace_bracket_citation(match):
        nonlocal last_new
        old = int(match.group(1))
        new = int(citation_remap.get(old, old))

        if last_new == new:
            return ""
        last_new = new

        b64 = _make_b64_payload(old)
        cid = f"cite-{msg_id}-{old}"
        return (
        f'<a class="citation-link" href="#{cid}">'
        f"<sup>[{new}]</sup></a >"
        f'<template class="citation-payload" id="{cid}">{b64}</template>'
        )   
        # return (
        #     f'<a class="citation-link" href=" ">'
        #     f"<sup>[{new}]</sup></a >"
        #     f'<template class="citation-payload" id="{cid}">{b64}</template>'
        # )
        # return (
        #     f'<span class="citation-link" '
        #     f'data-cite="{new}" data-old="{old}" data-key="{_key(old)}" '
        #     f'data-content-b64="{b64}">'
        #     f"<sup>[{new}]</sup></span>"
        # )

    result = re.sub(r"\[(\d+)\]", _replace_bracket_citation, result)
    print(f"[add_citation_tooltips] result: {result}")
    return result