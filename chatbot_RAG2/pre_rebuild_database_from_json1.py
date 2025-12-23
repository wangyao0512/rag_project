"""
Rebuild medical database from processed JSON files in 1215_MVP_processed_final
Converts 'score' field to 'grade' field
"""
import sys
import os
import json
import uuid
from pathlib import Path
from loguru import logger
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time
import re
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

json_folder = Path("/data_vision9/ywang/heisesuliuMVP_PICO_final_processed")
json_files = list(json_folder.glob("*_pico.json"))
json_meta_path = Path("/data_vision9/ywang/heisesuliuMVP_PICO_final")

def extract_json_from_response(response_text):
    """
    从AI响应中提取JSON格式的数据
    """
    # 尝试查找JSON格式的响应
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"无法解析JSON: {json_str}")
            return None
    return None

def process_single_reference(reference):
    """
    处理单个参考信息，获取评分、国家和类型
    """
    base_url = "http://10.6.52.31:1025/v1"
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    openai_api_key = "-"
    
    client = ChatOpenAI(
        base_url=base_url,
        api_key=openai_api_key,
        model=model_name,
    )
    
    # 构建提示模板，使用当前的references
    prompt_template = '''
你是一位资深医学专家，请根据以下提供的参考信息 {reference}，严格按照下列规则评估：

【评分规则（1–10分）】
- 10分：NCCN/ESMO最新版（2024或2025年发布），基于高证据等级（I类/RCT）
- 9分：中国国家级临床指南（2023–2025年发布，如CSCO、卫健委、中华医学会）
- 8分：中国国家级指南（2019–2022年发布）或近3年高质量分子检测指导
- 7分：多中心专家共识（2020–2025年），具强临床相关性
- 6分：小众领域共识（2020–2025年），聚焦操作/诊断，证据有限
- 5分：任何指南/共识发布于2016–2019年（距今6–9年）
- 4分及以下：发布于2015年或更早（距今≥10年），无论来源权威性
  - 2010–2015年：4分
  - 2005–2009年：2–3分
  - 2004年及以前：1分

【重要约束】
- 所有2015年及以前发布的指南，最高不超过4分；2009年及以前（如2008年）不得高于3分。
- 中国指南（CN）可在上述分数基础上+1分，但不得突破上述时效上限（例如2008年CN指南仍≤3分）。

【输出要求】
1. 国家/地区代码：使用ISO两位字母简写（如CN, US, EN, JP, KR, INT）
2. 指南焦点类型：从以下选一：treatment, diagnostic, prognosis, follow-up, survival
3. 仅输出一个JSON对象，不要任何解释、注释或额外文本。

输出格式：
{{
  "grade": <整数>,
  "country": "<代码>",
  "guideline_focus": "<类型>"
}}
'''

    # 正确地格式化提示内容
    formatted_prompt = prompt_template.format(reference=reference)
    
    messages = [
        SystemMessage(content="你是一个专业的医学专家助手，请根据以下信息进行输出，不输出额外内容，请输出json格式，格式为：{\"grade\": xxx, \"country\": xxx, \"guideline_focus\": xxx}"),
        HumanMessage(content=formatted_prompt),
    ]
    
    try:
        response = client.invoke(
            messages,
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
        )
        response_text = response.content

        
        # 尝试解析JSON响应
        result = extract_json_from_response(response_text)
        if result and 'grade' in result and 'country' in result and 'guideline_focus' in result:
            return result
        else:
            logger.error(f"无法解析AI响应为有效JSON: {response_text}")
            return {"grade": 5, "country": "UNKNOWN", "guideline_focus": "unknown"}  # 默认值
            
    except Exception as e:
        logger.error(f"调用AI API时出错: {e}")
        traceback.print_exc()
        return {"grade": 5, "country": "UNKNOWN", "guideline_focus": "unknown"}  # 默认值

def main():
    # 把所有的reference都读出来，以及读出meta json信息
    all_references = []
    file_mapping = {}  # 映射文件到其数据
    
    for json_file in json_files:
        logger.info(f"正在处理文件: {json_file}")
        
        try:
            with open(json_file, "r", encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败: {json_file}, 错误: {e}")
            continue
        
        meta_file = json_meta_path / (json_file.stem.replace("_pico", "_meta") + ".json")
        if os.path.exists(meta_file):
            logger.info(f"Meta file found: {meta_file}")
        else:
            logger.error(f"Meta file not found: {meta_file}")
            continue
        
        try:
            with open(meta_file, "r", encoding='utf-8') as f:
                # 检查文件是否为空
                content = f.read().strip()
                if not content:
                    logger.warning(f"Meta file is empty: {meta_file}")
                    continue
                meta = yaml.safe_load(content)
        except Exception as e:
            logger.error(f"读取meta文件失败: {meta_file}, 错误: {e}")
            continue
        
        # 创建一个空的list，用于存储所有的reference
        references = []
        references.append(meta.get("publishing_organization", ""))
        references.append(meta.get("title", ""))

        # reference要改写，llm_text要重写
        ##################
        #################
        #################
        #################
        ### 要补上相关信息
        
        if data and len(data) > 0:
            references.append(data[0].get("reference", ""))
        else:
            references.append("")
        references.append(meta.get("publication_date", ""))
        
        all_references.append({
            'references': references,
            'json_file': json_file,
            'data': data
        })
        
        # # 添加到文件映射
        # file_mapping[str(json_file)] = {
        #     'references': references,
        #     'data': data
        # }

    # 逐个处理每个文件的参考信息
    for item in all_references:
        json_file = item['json_file']
        data = item['data']
        references = item['references']
        
        logger.info(f"正在为文件 {json_file} 获取评分信息...")
        
        # 获取评分、国家和类型
        grade_info = process_single_reference(references)
        
        # 更新数据中的每个条目
        for item_data in data:
            item_data["grade"] = grade_info["grade"]
            item_data["country"] = grade_info["country"]
            item_data["guideline_focus"] = grade_info["guideline_focus"]
        
        # 写回JSON文件
        output_file = json_file  # 可以选择输出到不同路径
        try:
            # 确保目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"成功更新文件: {output_file}")
        except Exception as e:
            logger.error(f"写入文件失败: {output_file}, 错误: {e}")
        
        # 添加延时以避免API调用过于频繁
        time.sleep(1)

    logger.info("所有文件处理完成！")

if __name__ == "__main__":
    main()

