#!/usr/bin/env python3
"""
Medical RAG System - Command Line Version
"""

import sys
import json
from datetime import datetime
from loguru import logger
import uuid

from src.retrieval.rag_pipeline import MedicalRAG
from src.utils.citations import (
    format_sources_with_hover,
    add_citation_tooltips
)
from src.utils.usage_logger import get_usage_logger
from config import Config

# Setup logging
logger.remove()
logger.add(sys.stderr, level=Config.LOG_LEVEL)
logger.add(Config.LOG_FILE, rotation=Config.LOG_ROTATION, level="DEBUG")

# Initialize RAG system
logger.info("Initializing Medical RAG system...")
try:
    rag = MedicalRAG()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag = None

# Initialize usage logger
usage_logger = get_usage_logger("logs/usage_log.txt")
logger.info("Usage logger initialized")

def process_query_answer(message):
    """Process a single query and print the response"""
    session_id = str(uuid.uuid4())

    try:
        logger.info(f"User query: {message}")
        
        # Process query through RAG with streaming
        result = rag.process_query(message, stream=True, use_cot=True, extract_entities=False, max_context_chunks=15)

        # Check if query was refused
        if isinstance(result, dict) and result.get('refused'):
            print(result['answer'])
            return

        # Collect response data
        accumulated_text = ""
        raw_text = ""
        entities = {}
        sources = []
        cot_analysis = None
        chunk_count = 0
        inside_think = False
        
        import re
        buffer = ""
        buffer_size = 20
        
        for chunk in result:
            token = chunk['token']
            entities = chunk.get('entities', {})
            sources = chunk.get('sources', [])
            cot_analysis = chunk.get('cot_analysis')
            chunk_count += 1
            
            raw_text += token
            
            # Handle think tags
            if '<think>' in token:
                inside_think = True
            if '</think>' in token:
                inside_think = False
                continue
            
            # Accumulate tokens outside think blocks
            if not inside_think and 'think>' not in token:
                buffer += token
                
                # Output when buffer is full or at sentence end
                if len(buffer) >= buffer_size or token in ['.', '!', '?', '\n']:
                    accumulated_text += buffer
                    print(buffer, end='', flush=True)
                    buffer = ""
        
        # Print remaining buffer content
        if buffer:
            accumulated_text += buffer
            print(buffer, end='', flush=True)
            
        logger.info(f"Streamed {chunk_count} chunks, final sources count: {len(sources)}")
        logger.info(f"Raw text length: {len(raw_text)} chars, cleaned length: {len(accumulated_text)} chars")
        
        # Clean up think tags
        accumulated_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
        accumulated_text = re.sub(r'\n\n\n+', '\n\n', accumulated_text).strip()
        
        # Print additional information
        print("\n")
        
        # Add CoT analysis
        if cot_analysis and cot_analysis.get('is_medical'):
            cot_info = f"🤔 查询分析：医学相关查询 ({cot_analysis.get('reasoning', '')})\n"
            print(cot_info)
        
        # Add entity information
        if any(entities.values()):
            entity_info = "🔍 识别的实体："
            entity_parts = []
            if entities.get('disease'):
                entity_parts.append(f"疾病: {entities['disease']}")
            if entities.get('drug'):
                entity_parts.append(f"药物: {entities['drug']}")
            if entities.get('population'):
                entity_parts.append(f"人群: {entities['population']}")
            entity_info += " | ".join(entity_parts) + "\n"
            print(entity_info)
        
        # Format and display sources
        sources_text, hover_data, citation_remap = format_sources_with_hover(sources, accumulated_text)
        html_response = add_citation_tooltips(accumulated_text, hover_data, citation_remap)
        
        if sources_text:
            print("---")
            print("📚 参考文献:")
            print(sources_text)
        
        # Log interaction
        try:
            usage_logger.log_interaction(
                question=message,
                answer=accumulated_text,
                raw_answer=raw_text,
                sources=sources,
                entities=entities,
                cot_analysis=cot_analysis,
                session_id=session_id
            )
        except Exception as log_error:
            logger.error(f"Failed to log interaction: {log_error}")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"抱歉，处理您的问题时出现错误：{str(e)}")

def main():
    """Main command line interface"""
    print("🏥 L1循证医学问答系统（命令行版）")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 50)
    
    while True:
        try:
            query = "陈先生，70 岁，有类风湿关节炎，长期服用类固醇及甲氨蝶呤。近期病理确诊 III 期黑色素瘤，淋巴结肿大但无远处转移。他担心免疫治疗会加重原有疾病。他问：在免疫抑制状态下还能使用 PD-1 抑制剂吗？是否更推荐靶向治疗或手术/放疗优先？"
            
            if query.lower() in ['quit', 'exit']:
                print("感谢使用，再见！")
                break
                
            if not query:
                continue
                
            print("\n正在生成回答...")
            print("-" * 30)
            process_query_answer(query)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except EOFError:
            print("\n\n输入结束，再见！")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    logger.info("Starting command line Medical RAG app...")
    main()