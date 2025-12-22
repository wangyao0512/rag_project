"""
Simple Gradio chat interface for Medical RAG System
"""
import gradio as gr
import copy, re, uuid
from src.retrieval.rag_pipeline import MedicalRAG
from src.utils.citations import (
    extract_citation_order,
    group_sources_by_article,
    format_sources_with_hover,
    add_citation_tooltips
)
from src.utils.usage_logger import get_usage_logger
import json
from datetime import datetime
from loguru import logger
import sys
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

# Citation functions are now imported from src.utils.citations
# No need to redefine them here
def clear_history():
    global chat_history
    chat_history = {}  # 保持字典类型一致性
    return "对话历史已清空"
chat_history = {}
def chat_response(message, history, request: gr.Request):
    global chat_history
    session_id = getattr(request, "session_hash", None) or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    # 初始化当前会话的历史
    if session_id not in chat_history:
        chat_history[session_id] = []
    accumulated_text = ""  # 移到开头重置
    raw_text = ""
    entities = {}
    sources = []
    
    cot_analysis = None
    chunk_count = 0
    inside_think = False
    buffer = ""  # 缓冲区重置
    buffer_size = 40

    try:
        # Process query through RAG with streaming
        logger.info(f"User query: {message}")
        # 清空 rag 实例的 sources 缓存（如果有）
        if hasattr(rag, 'sources'):
            rag.sources = []  # 关键：清空上一轮的来源缓存
        result = rag.process_query(
            message, 
            stream=True, 
            use_cot=True, 
            extract_entities=False, 
            max_context_chunks=Config.MAX_CONTEXT_SOURCES,
            session_id=session_id
        )

        # Check if query was refused (returns dict instead of generator)
        if isinstance(result, dict) and result.get('refused'):
            yield result['answer']
            return
        if isinstance(result, dict) and result.get('need_followup'):
            # Persist staging input state for the session
            yield result.get('answer', "请补充分期所需信息。")
            return
        final_sources = []
        for chunk in result:
            if isinstance(chunk, dict):
                if 'token' in chunk:
                    token = chunk['token']
                    entities = chunk.get('entities', {})
                    if "sources" in chunk and chunk["sources"] is not None:
                        final_sources = copy.deepcopy(chunk["sources"])
                    cot_analysis = chunk.get('cot_analysis')
                elif 'choices' in chunk and chunk['choices']:
                    token = chunk['choices'][0]['text']
                else:
                    continue
            else:
                token = str(chunk)
            
            chunk_count += 1

            raw_text += token

            if '<think>' in token:
                inside_think = True
            if '</think>' in token:
                inside_think = False
                continue  

            if not inside_think and 'think>' not in token:
                buffer += token
                
                if len(buffer) >= buffer_size or token in ['.', '!', '?', '\n']:
                    accumulated_text += buffer
                    yield accumulated_text
                    buffer = ""

        if buffer:
            accumulated_text += buffer
            yield accumulated_text
        
        sources = final_sources
        logger.info(f"Streamed {chunk_count} chunks, final sources count: {len(sources)}")
        logger.info(f"Raw text length: {len(raw_text)} chars, cleaned length: {len(accumulated_text)} chars")

        # Clean up any remaining think tags (in case of unclosed tags)
        accumulated_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
        accumulated_text = re.sub(r'\n\n\n+', '\n\n', accumulated_text).strip()

        final_response = accumulated_text

        # Add CoT analysis information
        if cot_analysis and cot_analysis.get('is_medical'):
            cot_info = f"\n\n🤔 **查询分析：** 医学相关查询 ({cot_analysis.get('reasoning', '')})"
            final_response += cot_info

        # Add entity information
        if any(entities.values()):
            entity_info = "\n\n🔍 **识别的实体：** "
            entity_parts = []
            if entities.get('disease'):
                entity_parts.append(f"疾病: {entities['disease']}")
            if entities.get('drug'):
                entity_parts.append(f"药物: {entities['drug']}")
            if entities.get('population'):
                entity_parts.append(f"人群: {entities['population']}")
            entity_info += " | ".join(entity_parts)
            final_response += entity_info

        # Format sources with new citation system
        sources_text, hover_data, citation_remap = format_sources_with_hover(final_sources, accumulated_text)

        # Add HTML for hover tooltips and remap citation numbers
        html_response = add_citation_tooltips(accumulated_text, hover_data, citation_remap, request_id)

        # Don't wrap in extra div - let Gradio handle it
        final_response = html_response + "\n\n---\n\n" + sources_text

        # Log the interaction to usage log (thread-safe)
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


        yield final_response

        # Save to history
        chat_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'query': message,
            'response': accumulated_text,
            'entities': entities,
            'num_sources': len(sources)
        })

        logger.info("Response generated successfully")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        yield f"抱歉，处理您的问题时出现错误：{str(e)}"


def export_chat_history():
    """Export chat history as JSON"""
    return json.dumps(chat_history, ensure_ascii=False, indent=2)



with open('static/custom.css', 'r', encoding='utf-8') as f:
    custom_css = f.read()
with open('static/custom.js', 'r', encoding='utf-8') as f:
    custom_js = f.read()

# Create Gradio interface
demo = gr.Blocks()

with demo:
    # Add custom CSS
    gr.HTML(f"<style>{custom_css}</style>")
    # gr.HTML(f"<style>{dark_theme_css}{custom_css}</style>")
    gr.Markdown("""
    # 🏥 L1循证医学问答系统（原型版-黑色素瘤问答）

    基于本地医学知识库的医学问答助手

    **功能特点：**
    - 🔍 循证医学知识检索
    - 📚 文献来源追踪
    - 🏥 疾病诊疗对话咨询（目前仅支持单轮问答）

    **⚠️ 免责声明：** 本系统仅供研究和参考使用，不能替代专业医疗建议。
    """)

    with gr.Tab("💬 对话"):
        chatbot = gr.ChatInterface(
            chat_response,
            # examples=[
            #     "陈先生，70 岁，有类风湿关节炎，长期服用类固醇及甲氨蝶呤。近期病理确诊 III 期黑色素瘤，淋巴结肿大但无远处转移。他担心免疫治疗会加重原有疾病。请问：在免疫抑制状态下还能使用 PD-1 抑制剂吗？是否更推荐靶向治疗或手术/放疗优先？",
            #     "赵先生，38岁，男性，无明显基础疾病。职业为户外工程师，长期暴露在阳光下。两个月前发现左肩部出现一颗黑色斑点，逐渐增大并伴有边缘模糊、颜色不均的情况。最近在医院进行了皮肤镜检查，提示怀疑恶性黑色素瘤。医生建议进一步切除，但患者担心手术会造成疤痕和肿瘤扩散。请问：皮肤镜检查是否已经可以确诊黑色素瘤？还需要做切除活检吗？",
            #     "刘先生，57岁，两年前确诊皮肤黑色素瘤并手术切除。近半年出现头痛和短暂性视物模糊，MRI 显示右顶叶 0.9 cm 的增强病灶，考虑脑转移。医生建议立体定向放疗并结合免疫治疗。请问：脑转移患者的治疗策略是否不同于肺转移？放疗和免疫治疗的顺序如何安排更合适？",
            #     "李女士，52岁，女性。一年前右前臂出现一块黑色斑片，逐渐增大并呈不规则形态。半年后活检确诊为皮肤黑色素瘤，Breslow 厚度 2.3 mm，未见溃破。行手术切除后哨兵淋巴结活检阳性，分期为 IIIA 期。术后半年复查 PET-CT 显示腋窝区小淋巴结增大。基因检测提示 BRAF V600K 突变。她现在是否需要进行辅助免疫治疗？PD-1 单药与 PD-1 联合 CTLA-4 治疗在 III 期患者中的效果差异大吗？",
            #     "手术切除和激光治疗哪个更适合治疗色素痣？",
            #     "黑色素瘤是怎么治疗的？",
            #     "黑色素瘤是怎么诊断的？"
            # ],
            examples=[
                "黑色素瘤应该怎么治疗？",
                "黑色素瘤厚度1.9mm，原发灶无溃疡，手术完成，淋巴结阴性，未发现区域淋巴结转移，未发现临床隐匿性转移，也未发现远端转移，Ki-67为100%，S100为阳性，SOX同样是阳性，未发生基因突变，这位患者需要进行术后辅助治疗吗，如果需要，请给出推荐的辅助治疗方案。",
                "赵先生，38岁，男性，无明显基础疾病。职业为户外工程师，长期暴露在阳光下。两个月前发现左肩部出现一颗黑色斑点，逐渐增大并伴有边缘模糊、颜色不均的情况。最近在医院进行了皮肤镜检查，提示怀疑恶性黑色素瘤。医生建议进一步切除，但患者担心手术会造成疤痕和肿瘤扩散。他想了解，皮肤镜检查是否已经可以确诊黑色素瘤？还需要做切除活检吗？",
                "王先生，45岁，男性，无明显基础疾病。父亲曾患黑色素瘤。3年前发现右小腿一颗痣逐渐变大、变黑，并伴有不规则边缘。两年前确诊为皮肤黑色素瘤，Breslow 厚度 1.8 mm，溃破阳性。手术切除 + 哨兵淋巴结活检阴性。今年随访期间出现局部皮下小结节，活检提示复发。基因检测发现 BRAF V600E 突变。他要不要做免疫治疗作为辅助治疗？ PD-1 单药与联合治疗相比效果差别大吗？",
                "李女士，52岁，女性，有长期日晒史，年轻时经常去海边晒日光浴。无家族遗传病史。一年前右前臂出现一块黑色斑片，逐渐增大并呈不规则形态。半年后皮肤镜检查提示高度怀疑黑色素瘤，活检确诊为皮肤黑色素瘤，Breslow 厚度 2.3 mm，未见溃破。行手术切除后哨兵淋巴结活检阳性，分期为 IIIA 期。术后半年复查 PET-CT 显示腋窝区小淋巴结增大。基因检测提示 BRAF V600K 突变。她现在是否需要进行辅助免疫治疗？PD-1 单药与 PD-1 联合 CTLA-4 治疗在 III 期患者中的效果差异大吗？",
                "手术切除和激光治疗哪个更适合治疗色素痣？"
            ],
            title=None
        )

    with gr.Tab("📊 历史记录"):
        gr.Markdown("### 对话历史记录")

        with gr.Row():
            export_btn = gr.Button("📥 导出历史", scale=1)
            clear_btn = gr.Button("🗑️ 清空历史", scale=1)

        history_display = gr.JSON(label="历史记录（JSON格式）")
        clear_msg = gr.Textbox(label="操作结果", visible=False)

        export_btn.click(export_chat_history, outputs=history_display)
        clear_btn.click(clear_history, outputs=clear_msg)

    with gr.Tab("ℹ️ 关于"):
        gr.Markdown("""
        ### 系统信息

        **模型配置：**
        - **语言模型**: Qwen3-235B-256K 
        - **嵌入模型**: Qwen3-Embedding-8B
        - **数据库**: SQLite + ChromaDB
        - **向量维度**: 4096
        - **上下文窗口**: 256K  tokens
        - **GPU加速**: NVIDIA H20

        ### 📖 使用说明

        1. **提问技巧**：直接用自然语言提问，系统会通过语义理解匹配相关知识
        2. **结果解读**：每个回答都会显示参考文献来源和引用标注
        3. **问答轮次**：目前版本只支持单轮问答
        4. **知识范围**：当前聚焦于黑色素瘤和色素痣的诊疗知识

        ### ⚠️ 免责声明

        本系统为研究原型，提供的医学信息仅供参考：
        - ❌ 不能替代专业医疗诊断
        - ❌ 不能作为治疗依据
        - ❌ 如有健康问题，请咨询专业医生

        ### 📝 版本信息

        - **版本**: v0.3.0 alpha (Melanoma Knowledge Base)
        - **更新日期**: 2025-12-10
        - **知识库**: 黑色素瘤诊疗知识库
        """)

    with gr.Tab("⚙️ 系统状态"):
        gr.Markdown("### 系统运行状态")

        def get_system_status():
            try:
                db_count = rag.db_pico.conn.execute("SELECT COUNT(*) FROM medical_evidence").fetchone()[0]
                chunk_count = rag.db_pico.conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
                vector_count = rag.vector_store_pico.count()

                status = f"""
                **数据库状态：**
                - 医学证据条目: {db_count}
                - 向量数据量: {vector_count}

                **系统状态：** ✅ 运行正常
                - LLM: {'已加载' if rag.llm.llm else '未加载'}
                - 向量数据库: {'已连接' if rag.vector_store_pico else '未连接'}
                - SQL数据库: {'已连接' if rag.db_pico else '未连接'}
                """
                return status
            except Exception as e:
                return f"❌ 获取状态失败: {str(e)}"

        status_btn = gr.Button("🔄 刷新状态")
        status_display = gr.Markdown()

        status_btn.click(get_system_status, outputs=status_display)

    # Add JavaScript for clickable citations
    demo.load(None, None, None, js=custom_js)


if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    demo.launch(
        server_name=Config.SERVER_NAME,
        server_port=Config.SERVER_PORT,
        share=Config.SHARE_GRADIO,
        debug=True,
        show_error=True
    )
