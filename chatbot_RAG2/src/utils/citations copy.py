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

def extract_citation_order(text: str) -> List[int]:
    """Extract citation numbers from text in order of appearance"""
    # ===== 关键修复：先清空文本中的旧引用标记 =====
    # 移除所有非数字的干扰字符，只保留 [N] 格式
    clean_text = re.sub(r'\[(\d+)\]', r'[\1]', text)
    citations = re.findall(r'\[(\d+)\]', clean_text)

    seen: Set[str] = set()
    ordered = []
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


def add_citation_tooltips(text: str, hover_data: Dict, citation_remap: Dict) -> str:
    """
    Add markdown tooltips to citations in the text using superscript and footnotes

    Args:
        text: The text containing citations
        hover_data: Dictionary mapping citation numbers to tooltip data
        citation_remap: Dictionary mapping old citation numbers to new ones

    Returns:
        Text with HTML citation tooltips and hidden data
    """

    def replace_wenxian_with_list(match):
        """Replace 文献X和Y patterns with remapped numbers"""
        full_text = match.group(0)
        numbers = re.findall(r'\d+', full_text)
        remapped_nums = [str(citation_remap.get(int(n), int(n))) for n in numbers]

        # Remove duplicates while preserving order
        seen = set()
        unique_remapped = []
        for num in remapped_nums:
            if num not in seen:
                seen.add(num)
                unique_remapped.append(num)

        # If all numbers map to the same reference, just say "文献"
        if len(unique_remapped) == 1:
            return '文献'

        # Otherwise, reconstruct with unique numbers
        result_text = full_text
        for old_num, new_num in zip(numbers, remapped_nums):
            result_text = result_text.replace(old_num, new_num, 1)
        return result_text

    # Match patterns like "文献3和7", "文献3、7", "文献3及7"
    result = re.sub(r'文献\d+[和、及]\d+', replace_wenxian_with_list, text)
    result = re.sub(r'文献\d+[和、及]文献\d+', replace_wenxian_with_list, text)

    # Handle simple "文献X" patterns
    def replace_wenxian(match):
        old_cite_num = int(match.group(1))
        new_cite_num = citation_remap.get(old_cite_num, old_cite_num)
        return f'文献{new_cite_num}'

    result = re.sub(r'文献(\d+)', replace_wenxian, result)

    # Clean up redundant patterns
    result = re.sub(r'文献(\d+)[和、及]文献\1', r'文献\1', result)
    result = re.sub(r'文献(\d+)[和、及]\1', r'文献\1', result)

    # Normalize comma-separated citations: [5,12] -> [5][12]
    def expand_comma_citations(match):
        numbers = re.findall(r'\d+', match.group(0))
        return ''.join([f'[{num}]' for num in numbers])

    result = re.sub(r'\[\d+(?:,\s*\d+)+\]', expand_comma_citations, result)

    # Extract all citations to sort them by their remapped values
    all_citations = re.findall(r'\[(\d+)\]', result)
    if all_citations:
        # Create a mapping from original citation numbers to their remapped values
        citation_to_remap = {}
        for old_cite in all_citations:
            old_cite_int = int(old_cite)
            new_cite = citation_remap.get(old_cite_int, old_cite_int)
            citation_to_remap[old_cite_int] = new_cite
        
        # Sort citations by their remapped values
        sorted_citations = sorted(citation_to_remap.items(), key=lambda x: x[1])
        
        # Now process the text to replace citations in the correct order
        # First, temporarily replace all citations with placeholders
        temp_placeholders = {}
        temp_result = result
        
        # Create unique placeholders to avoid conflicts
        for i, (old_cite, new_cite) in enumerate(sorted_citations):
            placeholder = f"__CITATION_PLACEHOLDER_{i}__"
            temp_result = re.sub(rf'\[{re.escape(str(old_cite))}\]', placeholder, temp_result)
            temp_placeholders[placeholder] = (old_cite, new_cite)
        
        # Replace placeholders with the actual citation HTML
        for placeholder, (old_cite, new_cite) in temp_placeholders.items():
            replacement = (f'<span class="citation-link cite-{new_cite} old-cite-{old_cite}" '
                          f'cite="{new_cite}" old-cite="{old_cite}">'
                          f'<sup>[{new_cite}]</sup></span>')
            temp_result = temp_result.replace(placeholder, replacement)
        
        result = temp_result
    else:
        # Original logic for remapping citation numbers when no citations found
        def replace_citation(match):
            old_cite_num = int(match.group(1))
            new_cite_num = citation_remap.get(old_cite_num, old_cite_num)
            return (f'<span class="citation-link cite-{new_cite_num} old-cite-{old_cite_num}" '
                    f'cite="{new_cite_num}" old-cite="{old_cite_num}">'
                    f'<sup>[{new_cite_num}]</sup></span>')

        result = re.sub(r'\[(\d+)\]', replace_citation, result)

    # Remove duplicate consecutive citations
    def deduplicate_citations(text):
        pattern = r'(<span class="citation-link cite-(\d+)"[^>]*><sup>\[\2\]</sup></span>)(?:<span class="citation-link cite-\2"[^>]*><sup>\[\2\]</sup></span>)+'
        return re.sub(pattern, r'\1', text)

    result = deduplicate_citations(result)

    # Create hidden lookup table for citation data
    citation_lookup = []
    for old_cite_num, data in hover_data.items():
        citation_obj = {
            'source': data.get('source', '未知来源'),
            'year': data.get('year', ''),
            'content': data['content']
        }
        json_str = json.dumps(citation_obj, ensure_ascii=False)
        content_b64 = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
        citation_lookup.append(
            f'<div class="citation-data-{old_cite_num}" style="display:none;">{content_b64}</div>'
        )

    # Append lookup table to the result
    lookup_html = '\n'.join(citation_lookup)
    result = result + '\n' + lookup_html
    # ✅ lookup 也用同一个 key
    # citation_lookup = []
    # for old_cite_num, data in hover_data.items():
    #     citation_obj = {
    #         "source": data.get("source", "未知来源"),
    #         "year": data.get("year", ""),
    #         "content": data.get("content", "")
    #     }
    #     json_str = json.dumps(citation_obj, ensure_ascii=False)
    #     content_b64 = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    #     key = f"{msg_id}:{old_cite_num}"
    #     citation_lookup.append(
    #         f'<div class="citation-data" data-key="{key}" style="display:none;">{content_b64}</div>'
    #     )

    # return result + "\n" + "\n".join(citation_lookup)
    return result






# def add_citation_tooltips(text: str, hover_data: Dict, citation_remap: Dict, msg_id: str) -> str:
    """
    将文本中的 [N] 引用替换为带 data-key 的 HTML span，并在末尾附加隐藏的 citation-data。
    修复点：
      1) 每条消息 msg_id 隔离，避免第二轮 querySelector 取到上一轮的数据；
      2) 不再用“按 remap 排序 + placeholder 全局替换”的方式（会把同号引用全部替成一个占位符，导致位置/顺序错乱）；
      3) 先把逗号引用 [1,2] 展开成 [1][2]，并支持“文献3和7”等文本引用号重映射。
    """

    # ---------- helpers ----------
    def _key(old_num: int) -> str:
        return f"{msg_id}:{old_num}"

    def _replace_wenxian_with_list(match):
        full_text = match.group(0)
        nums = re.findall(r"\d+", full_text)
        remapped = [str(citation_remap.get(int(n), int(n))) for n in nums]

        # 去重但保序
        seen = set()
        uniq = []
        for n in remapped:
            if n not in seen:
                seen.add(n)
                uniq.append(n)

        # 全映射到同一个就只保留“文献”
        if len(uniq) == 1:
            return "文献"

        # 逐个替换，保持原文本结构
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

    # ---------- 1) 处理“文献X/文献X和Y” ----------
    result = re.sub(r"文献\d+[和、及]\d+", _replace_wenxian_with_list, text)
    result = re.sub(r"文献\d+[和、及]文献\d+", _replace_wenxian_with_list, result)
    result = re.sub(r"文献(\d+)", _replace_wenxian_single, result)
    result = re.sub(r"文献(\d+)[和、及]文献\1", r"文献\1", result)
    result = re.sub(r"文献(\d+)[和、及]\1", r"文献\1", result)

    # ---------- 2) 展开逗号引用：[5,12] -> [5][12] ----------
    result = re.sub(r"\[\d+(?:,\s*\d+)+\]", _expand_comma_citations, result)

    # ---------- 3) 把 [N] 替换成 span（逐次替换，保留位置/顺序） ----------
    # 同时去掉连续重复引用（例如 [3][3]）
    last_new = None

    def _replace_bracket_citation(match):
        nonlocal last_new
        old = int(match.group(1))
        new = int(citation_remap.get(old, old))

        # 连续重复引用：直接丢弃后一个
        if last_new == new:
            return ""

        last_new = new
        return (
            f'<span class="citation-link" '
            f'data-cite="{new}" data-old="{old}" data-key="{_key(old)}">'
            f"<sup>[{new}]</sup></span>"
        )

    # 用回调逐个匹配替换，避免 placeholder 把同号引用一锅端
    result = re.sub(r"\[(\d+)\]", _replace_bracket_citation, result)

    # ---------- 4) 构造隐藏 lookup（用 msg_id:old_idx 唯一 key） ----------
    # 注意：hover_data 的 key 是 old 引用号（你在 format_sources_with_hover 里就是这么返回的）
    lookup_divs = []
    for old_num, data in (hover_data or {}).items():
        try:
            old_int = int(old_num)
        except Exception:
            continue

        citation_obj = {
            "source": data.get("source", "未知来源"),
            "year": data.get("year", ""),
            "content": data.get("content", ""),
        }
        payload = json.dumps(citation_obj, ensure_ascii=False)
        b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        lookup_divs.append(
            f'<div class="citation-data" data-key="{_key(old_int)}" style="display:none;">{b64}</div>'
        )

    if lookup_divs:
        result = result + "\n" + "\n".join(lookup_divs)

    return result