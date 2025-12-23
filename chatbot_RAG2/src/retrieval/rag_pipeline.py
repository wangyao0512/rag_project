"""
Complete RAG pipeline
"""
from src.database.schema import MedicalDatabase
from src.retrieval.vector_store import VectorStore
from src.chat.llm_handler import QwenLLM
from src.chat.cot_reasoner import CoTReasoner
from src.chat.prompts import PromptTemplates
import json
from typing import List, Dict, Optional, Generator, Union, Any
from langchain_core.runnables import RunnableBranch, RunnableLambda
from src.retrieval.two_stage import Reader, Decider
from src.retrieval.evidence_models import ReaderInput
from loguru import logger
from config import Config
try:
    import src.retrieval.melanoma_staging_mcp_server as melanoma_staging_mcp_server
except Exception as e:
    melanoma_staging_mcp_server = None
    logger.error(f"Melanoma staging MCP server import failed: {e}")
    
class MedicalRAG:
    def __init__(
        self,
        llm_db_path: str = Config.LLM_DB_PATH,
        llm_vector_db_path: str = Config.LLM_VECTOR_DB_PATH,
        pico_db_path: str = Config.PICO_DB_PATH,
        pico_vector_db_path: str = Config.PICO_VECTOR_DB_PATH,
        # 下面两个参数仅为兼容旧代码，不再使用本地模型路径
        model_path: Optional[str] = None,
        entity_model_path: Optional[str] = None,
        enable_two_stage: bool = False
    ):
        logger.info("Initializing Medical RAG system...")

        # 结构化数据库
        self.db_llm = MedicalDatabase(llm_db_path)
        self.vector_store_llm = VectorStore(persist_dir=llm_vector_db_path)

        self.db_pico = MedicalDatabase(pico_db_path)
        self.vector_store_pico = VectorStore(persist_dir=pico_vector_db_path)

        # 大模型：内部已经使用远程 Qwen Chat
        self.llm = QwenLLM(
            model_path=model_path,
            entity_model_path=entity_model_path,
        )
        self.cot_reasoner = CoTReasoner(self.llm)

        self.use_two_stage = enable_two_stage
        self.reader = Reader(
            max_primary=Config.TWO_STAGE_MAX_PRIMARY,
            max_supporting=Config.TWO_STAGE_MAX_SUPPORTING
        ) if enable_two_stage else None
        self.decider = Decider() if enable_two_stage else None
        self.enable_melanoma_staging = Config.ENABLE_MELANOMA_STAGING and melanoma_staging_mcp_server is not None
        self.staging_sessions: Dict[str, Dict] = {}
        self.last_cot: Dict[str, Any] = {}

        # Load entity mappings
        self.entity_mappings = self._load_entity_mappings()

        if self.use_two_stage:
            logger.info("Two-stage Reader/Decider scaffolding is enabled (stub mode).")

        logger.info("Medical RAG system initialized")

    def _load_entity_mappings(self) -> Dict:
        """Load medical term mappings"""
        return {
            # Drug mappings (Chinese -> Standard)
            "阿司匹林": "Aspirin",
            "二甲双胍": "Metformin",
            "格列美脲": "Glimepiride",
            "华法林": "Warfarin",
            "降压药": "Antihypertensive",
            "降糖药": "Antidiabetic",

            # Disease mappings
            "糖尿病": "Diabetes Mellitus",
            "2型糖尿病": "Type 2 Diabetes",
            "高血压": "Hypertension",
            "心梗": "Myocardial Infarction",
            "心肌梗死": "Myocardial Infarction",
            "中风": "Stroke",

            # Population
            "老年": "Elderly",
            "儿童": "Pediatric",
            "孕妇": "Pregnant"
        }

    def normalize_entity(self, entity: Optional[str]) -> Optional[str]:
        """Normalize medical entity"""
        if not entity:
            return None
        return self.entity_mappings.get(entity, entity)


    def search_structured(self, db: MedicalDatabase, entities: Dict[str, Optional[str]]) -> List[Dict]:
        disease = entities.get('disease')
        drug = entities.get('drug')

        results = db.search_evidence(disease=disease, drug=drug, limit=10)
        logger.info(f"Found {len(results)} structured results")
        return results

    def get_chunks_from_evidence(self, db: MedicalDatabase, evidence_list: List[Dict],
                                 max_chunks_per_result: int = 2) -> List[Dict]:
        """Get text chunks from evidence results"""
        all_chunks: List[Dict] = []

        for result in evidence_list:
            chunk_ids = json.loads(result.get('chunk_ids', '[]'))
            for chunk_id in chunk_ids[:max_chunks_per_result]:
                chunk = db.get_chunk(chunk_id)
                if chunk:
                    all_chunks.append({
                        'content': chunk['content'],
                        'metadata': json.loads(chunk.get('metadata', '{}')),
                        'source': chunk.get('source_doc', 'unknown')
                    })

        return all_chunks

    def search_vector(self, vector_store: VectorStore, db: MedicalDatabase, query: str, n_results: int = 5) -> List[
        Dict]:
        """Search vector database with fallback to broader query"""
        try:
            results = vector_store.search(query, n_results=n_results)

            chunks = []
            if results and results.get('documents'):
                distances = results.get('distances', [[]])[0]
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    distance = distances[i] if i < len(distances) else 999
                    chunks.append({
                        'content': doc,
                        'metadata': metadata,
                        'source': metadata.get('source', 'vector_db'),
                        'distance': distance
                    })

            if not chunks or (chunks and chunks[0].get('distance', 0) > 0.5):
                logger.info("Initial search quality low, trying broader query...")
                broader_query = self._extract_core_medical_terms(query)
                if broader_query != query:
                    logger.info(f"Broader query: {broader_query}")
                    broader_results = vector_store.search(broader_query, n_results=n_results)

                    if broader_results and broader_results.get('documents'):
                        broader_distances = broader_results.get('distances', [[]])[0]
                        for i, doc in enumerate(broader_results['documents'][0]):
                            metadata = broader_results['metadatas'][0][i] if broader_results.get('metadatas') else {}
                            distance = broader_distances[i] if i < len(broader_distances) else 999
                            if doc not in [c['content'] for c in chunks]:
                                chunks.append({
                                    'content': doc,
                                    'metadata': metadata,
                                    'source': metadata.get('source', 'vector_db'),
                                    'distance': distance
                                })

                chunks.sort(key=lambda x: x.get('distance', 999))

            logger.info(f"Found {len(chunks)} vector search results")

            if len(chunks) > 0:
                chunks = self._rerank_by_grade(db, chunks, Config.VECTOR_SEARCH_TOP_K)

            return chunks

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _rerank_by_grade(self, db: MedicalDatabase, chunks: List[Dict], top_k: int) -> List[Dict]:
        # Get GRADE from metadata (already stored during ingestion)
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            grade = metadata.get('grade')
            cluster = metadata.get('guideline_focus')

            if grade is None:
                chunk_id = metadata.get('chunk_id', '')
                try:
                    evidence_row = db.conn.execute("""
                        SELECT grade, cluster, source FROM medical_evidence
                        WHERE chunk_ids LIKE ?
                    """, (f'%{chunk_id}%',)).fetchone()

                    if evidence_row:
                        grade = evidence_row[0]
                        cluster = evidence_row[1] if evidence_row[1] else cluster
                        source = evidence_row[2]
                        chunk['source'] = source if source else chunk.get('source', '')
                except Exception as e:
                    logger.debug(f"Could not lookup GRADE from database: {e}")

            chunk['grade'] = grade if grade is not None else 5
            chunk['guideline_focus'] = cluster

        for chunk in chunks:
            distance = chunk.get('distance', 1.0)
            grade = chunk.get('grade', 5)

            grade_score = (grade - 1) / 9.0
            normalized_distance = min(distance / 1.5, 1.0)
            relevance_score = 1.0 - normalized_distance

            # 你原来就是 0.8/0.2，我保持不动
            chunk['combined_score'] = 0.8 * relevance_score + 0.2 * grade_score

        chunks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        if chunks:
            logger.info(
                f"Reranked by GRADE. Top result: GRADE={chunks[0].get('grade')}, "
                f"cluster={chunks[0].get('guideline_focus')}, score={chunks[0].get('combined_score', 0):.3f}"
            )

        return chunks[:top_k]


    def _extract_core_medical_terms(self, query: str) -> str:
        """Extract core medical terms from query for broader search"""
        import re

        # Remove demographic details (age, gender, etc.)
        query = re.sub(r'\d+岁', '', query)  # Remove age
        query = re.sub(r'(男性|女性|患者|病人)', '患者', query)  # Generalize gender/patient

        # Keep disease/treatment terms
        medical_keywords = ['黑色素瘤', '黑素色瘤', '色素痣', '手术', '治疗', '诊断', '检查', '检验',
                          '复发', '转移', '切除', '激光', '药物', '化疗', '放疗']

        found_terms = []
        for keyword in medical_keywords:
            if keyword in query:
                found_terms.append(keyword)

        # If we found medical terms, use those; otherwise return simplified query
        if found_terms:
            return ' '.join(found_terms)

        # Fallback: just remove specific details
        return query.strip()

    def filter_chunks_by_entities(self, chunks: List[Dict], entities: Dict) -> List[Dict]:
        """Filter chunks to only keep those matching extracted entities"""
        if not entities or not any(entities.values()):
            return chunks

        filtered = []

        # Get entity values (clean up nulls and multiple values) - handle None safely
        disease = (entities.get('disease') or '').lower().replace('null', '').strip()
        drug = (entities.get('drug') or '').lower().replace('null', '').strip()
        population = (entities.get('population') or '').lower().replace('null', '').strip()

        # Split multiple drugs if comma-separated
        drugs = [d.strip() for d in drug.split(',') if d.strip()] if drug else []

        for chunk in chunks:
            content = chunk['content'].lower()
            metadata = chunk.get('metadata', {})

            # Check if chunk matches any extracted entity
            matches = False

            # Check disease match
            if disease and disease in content:
                matches = True

            # Check drug match (any of the drugs)
            if drugs:
                for drug_name in drugs:
                    if drug_name in content:
                        matches = True
                        break

            # Check population match
            if population and population in content:
                matches = True

            # Also check normalized entities in metadata
            if 'disease' in metadata and disease:
                if disease in str(metadata.get('disease', '')).lower():
                    matches = True

            if 'drug' in metadata and drugs:
                meta_drug = str(metadata.get('drug', '')).lower()
                for drug_name in drugs:
                    if drug_name in meta_drug:
                        matches = True
                        break

            if matches:
                filtered.append(chunk)

        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} matching entities")
        return filtered

    def combine_results(self, structured_chunks: List[Dict], vector_chunks: List[Dict],
                       max_total: int = 5) -> List[Dict]:
        """Combine and deduplicate results"""
        seen_contents = set()
        combined = []

        # Prioritize structured results
        for chunk in structured_chunks:
            content = chunk['content']
            if content not in seen_contents:
                seen_contents.add(content)
                combined.append(chunk)
                if len(combined) >= max_total:
                    return combined

        # Add vector results
        for chunk in vector_chunks:
            content = chunk['content']
            if content not in seen_contents:
                seen_contents.add(content)
                combined.append(chunk)
                if len(combined) >= max_total:
                    return combined

        return combined
    def _merge_staging_inputs(self, base: Dict, update: Dict) -> Dict:
        """Merge two staging payloads preferring non-null, non-unknown from update."""
        def pick(old_val, new_val):
            if new_val in (None, "", "unknown"):
                return old_val
            return new_val

        merged = base.copy() if base else {}
        merged.setdefault("context", {"mode": "mixed", "language": "zh"})
        for key in ["primary_tumor", "regional_nodes", "distant_metastasis"]:
            merged.setdefault(key, {})
            update_section = (update or {}).get(key, {})
            for k, old_v in merged[key].items():
                merged[key][k] = pick(old_v, update_section.get(k, old_v))
            for k, new_v in update_section.items():
                merged[key].setdefault(k, new_v)
        return merged
    def _maybe_compute_melanoma_staging(self, query: str, session_id: Optional[str]) -> Optional[Dict]:
        """
        If the query appears to ask about melanoma staging, call the staging tool.
        Returns a chunk-like dict for context injection.
        """
        if not self.enable_melanoma_staging or melanoma_staging_mcp_server is None:
            return None

        lowered = query.lower()
        staging_keywords = ["分期", "分组", "staging", "stage", "tnm", "ajcc", "csco"]
        treatment_keywords = ["治疗", "方案", "用药", "手术", "免疫", "靶向", "化疗", "放疗", "辅助", "adjuvant", "neoadjuvant", "metastatic", "复发", "转移"]
        melanoma_hit = ("黑色素瘤" in query) or ("melanoma" in lowered)
        staging_hit = any(k in lowered for k in staging_keywords)
        treatment_hit = any(k in lowered for k in treatment_keywords)
        cot_staging = False
        # If CoT flagged need for staging, honor it
        try:
            cot_staging = bool(self.last_cot.get("needs_melanoma_staging"))  # type: ignore[attr-defined]
        except Exception:
            cot_staging = False
        # Trigger staging if melanoma AND (explicit staging OR treatment intent OR CoT request)
        if not (melanoma_hit and (staging_hit or treatment_hit or cot_staging)):
            return None

        try:
            # Retrieve prior staged input if we already asked for follow-up
            staging_input = {}
            if session_id and session_id in self.staging_sessions:
                staging_input = self.staging_sessions.get(session_id, {})

            # Always ask LLM to extract from current query
            new_input = {}
            try:
                # print(query)
                new_input = self.llm.extract_melanoma_staging_fields(query)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Melanoma staging field extraction failed: {e}")
                new_input = {}

            staging_input = self._merge_staging_inputs(staging_input, new_input)

            if not staging_input:
                staging_input = {
                    "context": {"mode": "mixed", "language": "zh"},
                    "primary_tumor": {},
                    "regional_nodes": {},
                    "distant_metastasis": {},
                }

            staging = melanoma_staging_mcp_server.melanoma_staging_pipeline_ajcc8_csco2025(
                context=staging_input.get("context", {"mode": "mixed", "language": "zh"}),
                primary_tumor=staging_input.get("primary_tumor", {}),
                regional_nodes=staging_input.get("regional_nodes", {}),
                distant_metastasis=staging_input.get("distant_metastasis", {}),
            )
            tnm = staging.get("tnm", {})
            stage_group = staging.get("stage_group", "UNDETERMINED")
            missing = staging.get("missing_fields", [])
            warnings = staging.get("warnings", [])
            summary_lines = [
                "自动分期结果（AJCC8/CSCO2025 原型）：",
                f"TNM: T={tnm.get('T','?')} N={tnm.get('N','?')} M={tnm.get('M_base','?')}",
                f"分期分组: {stage_group}",
                "请在回答中明确提及上述 TNM 与分期分组，并基于该分期给出治疗建议。",
            ]
            if missing:
                summary_lines.append(f"缺失字段: {', '.join(missing)}")
            if warnings:
                summary_lines.append(f"注意事项: {', '.join(warnings)}")
            content = "\n".join(summary_lines)
            return {
                "content": content,
                "metadata": {
                    "source": "melanoma_staging_tool",
                    "staging": staging,
                    "staging_input": staging_input,
                }
            }
        except Exception as e:
            logger.error(f"Failed to run melanoma staging tool: {e}")
            return None

    def _should_use_staging_tool(self, query: str, cot_analysis: Optional[Dict]) -> bool:
        lowered = query.lower()
        staging_keywords = ["分期", "分组", "staging", "stage", "tnm", "ajcc", "csco"]
        treatment_keywords = ["治疗", "方案", "用药", "手术", "免疫", "靶向", "化疗", "放疗", "辅助", "adjuvant", "neoadjuvant", "metastatic", "复发", "转移"]
        melanoma_hit = ("黑色素瘤" in query) or ("melanoma" in lowered)
        staging_hit = any(k in lowered for k in staging_keywords)
        treatment_hit = any(k in lowered for k in treatment_keywords)
        cot_staging = bool(cot_analysis.get("needs_melanoma_staging")) if cot_analysis else False
        return melanoma_hit and (staging_hit or treatment_hit or cot_staging)

    def _build_refusal_payload(self, query: str, cot_analysis: Dict) -> Dict:
        refusal_msg = self.cot_reasoner.get_refusal_message(cot_analysis)
        return {
            'query': query,
            'entities': {},
            'normalized_entities': {},
            'answer': refusal_msg,
            'sources': [],
            'num_sources': 0,
            'cot_analysis': cot_analysis,
            'refused': True
        }

    def _stream_answer_payload(self, query: str, context: List[Dict], cot_analysis: Optional[Dict],
                               entities: Dict, normalized_entities: Dict, stream: bool):
        if not stream:
            answer = self.llm.generate_answer(query, context, stream=False)
            return {
                'query': query,
                'entities': entities,
                'normalized_entities': normalized_entities,
                'answer': answer,
                'sources': context,
                'num_sources': len(context),
                'cot_analysis': cot_analysis,
                'refused': False
            }

        def generate():
            for token in self.llm.generate_answer(query, context, stream=True):
                yield {
                    'token': token,
                    'entities': entities,
                    'normalized_entities': normalized_entities,
                    'sources': context,
                    'num_sources': len(context),
                    'cot_analysis': cot_analysis
                }
        return generate()

    def _run_tool_chain(self, payload: Dict) -> Union[Dict, Generator]:
        query = payload["query"]
        stream = payload["stream"]
        session_id = payload.get("session_id")
        cot_analysis = payload.get("cot_analysis")

        staging_chunk = self._maybe_compute_melanoma_staging(query, session_id=session_id)
        if not staging_chunk:
            return self._run_rag_chain(payload)

        staging_meta = staging_chunk.get("metadata", {}).get("staging", {})
        missing_fields = staging_meta.get("missing_fields", [])
        if missing_fields:
            if session_id:
                self.staging_sessions[session_id] = staging_chunk.get("metadata", {}).get("staging_input", {})
            followup = (
                "为了准确进行黑色素瘤分期，请补充以下信息："
                + "、".join(missing_fields)
                + "。例如：Breslow厚度(毫米)、是否有溃疡、阳性淋巴结个数、是否临床可见、是否有远处转移及部位、LDH情况。"
            )
            return {
                'query': query,
                'answer': followup,
                'need_followup': True,
                'missing_fields': missing_fields,
                'staging_input': staging_chunk.get("metadata", {}).get("staging_input", {}),
                'session_id': session_id,
                'refused': False
            }

        if session_id and session_id in self.staging_sessions:
            self.staging_sessions.pop(session_id, None)

        return self._stream_answer_payload(
            query=query,
            context=[staging_chunk],
            cot_analysis=cot_analysis,
            entities={},
            normalized_entities={},
            stream=stream
        )

    def _run_chat_chain(self, payload: Dict) -> Union[Dict, Generator]:
        query = payload["query"]
        stream = payload["stream"]
        cot_analysis = payload.get("cot_analysis")
        prompt = PromptTemplates.simple_chat_prompt(query)
        stop_sequences = PromptTemplates.get_stop_sequences()

        if not stream:
            response = self.llm.llm(
                prompt,
                max_tokens=Config.MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                stop=stop_sequences,
                repeat_penalty=1.05
            )
            answer = response['choices'][0]['text'].strip()
            return {
                'query': query,
                'entities': {},
                'normalized_entities': {},
                'answer': answer,
                'sources': [],
                'num_sources': 0,
                'cot_analysis': cot_analysis,
                'refused': False
            }

        def generate():
            for output in self.llm.llm(
                prompt,
                max_tokens=Config.MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                stop=stop_sequences,
                stream=True,
                repeat_penalty=1.05
            ):
                token = output['choices'][0]['text']
                yield {
                    'token': token,
                    'entities': {},
                    'normalized_entities': {},
                    'sources': [],
                    'num_sources': 0,
                    'cot_analysis': cot_analysis
                }
        return generate()

    def _run_rag_chain(self, payload: Dict) -> Union[Dict, Generator]:
        query = payload["query"]
        max_context_chunks = payload["max_context_chunks"]
        stream = payload["stream"]
        extract_entities = payload["extract_entities"]
        session_id = payload.get("session_id")
        cot_analysis = payload.get("cot_analysis")

        route_info = self._route_store(query)
        db = route_info["db"]
        vector_store = route_info["vector_store"]
        pico = route_info["pico"]
        route = route_info["route"]
        logger.info(f"[RAG route] route={route}, pico_keys={list(pico.keys())}")
        entities = {}
        normalized_entities = {}
        if extract_entities:
            entities = self.llm.extract_entities(query)
            for key, value in entities.items():
                if value:
                    normalized_entities[key] = self.normalize_entity(value)

            if any(entities.values()):
                structured_results = self.search_structured(db, entities)
                structured_chunks = self.get_chunks_from_evidence(db, structured_results)
            else:
                structured_chunks = []
        else:
            structured_chunks = []
        if route == "pico":
            pico_parts = []
            for key in ["P", "I", "C", "O"]:
                if pico.get(key):
                    pico_parts.append(pico[key])
            pico_query = ",".join(pico_parts) if pico_parts else query
            vector_chunks = self.search_vector(vector_store, db, pico_query, n_results=max_context_chunks * 2)
        else:
            vector_chunks = self.search_vector(vector_store, db, query, n_results=max_context_chunks * 2)

        if extract_entities and entities and any(entities.values()):
            vector_chunks = self.filter_chunks_by_entities(vector_chunks, entities)

        all_chunks = self.combine_results(structured_chunks, vector_chunks, max_total=max_context_chunks)

        evidence_bundle = None
        if self.use_two_stage and self.reader:
            try:
                reader_input = ReaderInput(
                    question_text=query,
                    patient_profile={},
                    retrieved_items=all_chunks
                )
                evidence_bundle = self.reader.build_evidence_bundle(reader_input)
            except Exception as bundle_err:
                logger.error(f"Failed to build evidence bundle: {bundle_err}")

        use_two_stage_path = self.use_two_stage and evidence_bundle is not None

        if stream:
            def generate():
                if use_two_stage_path:
                    try:
                        for token in self.llm.generate_decider_answer(query, evidence_bundle, stream=True):
                            payload = {
                                'token': token,
                                'entities': entities,
                                'normalized_entities': normalized_entities,
                                'sources': all_chunks,
                                'num_sources': len(all_chunks),
                                'cot_analysis': cot_analysis,
                                'evidence_bundle': evidence_bundle.model_dump()
                            }
                            yield payload
                        return
                    except Exception as decider_err:
                        logger.error(f"Two-stage streaming failed, falling back: {decider_err}")

                for token in self.llm.generate_answer(query, all_chunks, stream=True):
                    payload = {
                        'token': token,
                        'entities': entities,
                        'normalized_entities': normalized_entities,
                        'sources': all_chunks,
                        'num_sources': len(all_chunks),
                        'cot_analysis': cot_analysis
                    }
                    if evidence_bundle:
                        payload['evidence_bundle'] = evidence_bundle.model_dump()
                    yield payload
            return generate()

        try:
            if use_two_stage_path:
                answer = self.llm.generate_decider_answer(query, evidence_bundle, stream=False)
            else:
                answer = self.llm.generate_answer(query, all_chunks, stream=False)
        except Exception as gen_err:
            logger.error(f"Two-stage generation failed, falling back: {gen_err}")
            answer = self.llm.generate_answer(query, all_chunks, stream=False)

        result = {
            'query': query,
            'entities': entities,
            'normalized_entities': normalized_entities,
            'answer': answer,
            'sources': all_chunks,
            'num_sources': len(all_chunks),
            'cot_analysis': cot_analysis,
            'refused': False
        }
        if evidence_bundle:
            result['evidence_bundle'] = evidence_bundle.model_dump()
        logger.info("Query processing complete")
        return result

    def _normalize_pico_value(self, v: Any) -> Optional[str]:
        """把各种 null/空/列表 统一成 str 或 None"""
        if v is None:
            return None
        # list -> join
        if isinstance(v, list):
            s = "，".join([str(x).strip() for x in v if x is not None and str(x).strip()])
        else:
            s = str(v).strip()

        if not s:
            return None

        low = s.lower()
        if low in ("null", "none", "n/a", "na", "unknown", "undefined"):
            return None
        if s in ("无", "未知", "不确定", "不详"):
            return None
        return s

    def _clean_pico(self, pico: Any) -> Dict[str, Optional[str]]:
        """保证输出结构固定为 P/I/C/O"""
        out: Dict[str, Optional[str]] = {"P": None, "I": None, "C": None, "O": None}
        if not isinstance(pico, dict):
            return out

        # 允许模型输出小写/其它key，统一转大写
        for k, v in pico.items():
            kk = str(k).strip().upper()
            if kk in out:
                out[kk] = self._normalize_pico_value(v)

        return out

    def _route_store(self, query: str) -> Dict[str, Any]:
        """
        路由选择：
        - 如果能抽出任一 P/I/C/O（非空），则走 PICO 的 db_path / vector_db_path
        - 否则走 LLM 的 db_path / vector_db_path

        返回:
          {
            "route": "pico"|"llm",
            "pico": {"P":..,"I":..,"C":..,"O":..},
            "db": <MedicalDatabase>,
            "vector_store": <VectorStore>,
          }
        """
        # 1) 先抽 PICO（失败也不要影响主流程）
        try:
            pico_raw = self.llm.extract_pico(query)
        except Exception as e:
            logger.debug(f"PICO extraction failed, fallback to LLM store. err={e}")
            pico_raw = {}

        pico = self._clean_pico(pico_raw)
        has_pico = any(pico.get(k) for k in ("P", "I", "C", "O"))

        # 2) 选择目标库（兼容你尚未改 __init__ 的情况）
        # 优先用新字段：db_llm/db_pico, vector_store_llm/vector_store_pico
        if has_pico and hasattr(self, "db_pico") and hasattr(self, "vector_store_pico"):
            logger.info(f"[RAG route] route=pico, pico={{P:{bool(pico['P'])},I:{bool(pico['I'])},C:{bool(pico['C'])},O:{bool(pico['O'])}}}")
            return {
                "route": "pico",
                "pico": pico,
                "db": self.db_pico,
                "vector_store": self.vector_store_pico,
            }

        # 否则走 llm（新字段优先）
        if hasattr(self, "db_llm") and hasattr(self, "vector_store_llm"):
            logger.info(f"[RAG route] route=llm, pico_present={has_pico}")
            return {
                "route": "llm",
                "pico": pico,
                "db": self.db_llm,
                "vector_store": self.vector_store_llm,
            }

        # 3) 最后兜底：老代码字段 self.db/self.vector_store
        logger.info(f"[RAG route] route=llm(fallback_default), pico_present={has_pico}")
        return {
            "route": "llm",
            "pico": pico,
            "db": getattr(self, "db", None),
            "vector_store": getattr(self, "vector_store", None),
        }

    def process_query(self, query: str, max_context_chunks: int = 5, stream: bool = False,
                      extract_entities: bool = True, use_cot: bool = True,
                      session_id: Optional[str] = None) -> Union[Dict, Generator]:
        
        logger.info(f"Processing query: {query}")

        cot_analysis = None
        if use_cot:
            logger.info("Performing CoT analysis...")
            cot_analysis = self.cot_reasoner.analyze_query(query)
            self.last_cot = cot_analysis
            logger.info(f"CoT Result: is_medical={cot_analysis['is_medical']}, needs_staging={cot_analysis.get('needs_melanoma_staging')}, reasoning={cot_analysis['reasoning']}")

            if self.cot_reasoner.should_refuse(cot_analysis):
                logger.info("Query refused: non-medical content")
                return self._build_refusal_payload(query, cot_analysis)

        route = "chat"
        if self._should_use_staging_tool(query, cot_analysis):
            route = "tool"
        elif cot_analysis and cot_analysis.get("is_medical"):
            route = "rag"

        payload = {
            "query": query,
            "max_context_chunks": max_context_chunks,
            "stream": stream,
            "extract_entities": extract_entities,
            "session_id": session_id,
            "cot_analysis": cot_analysis,
            "route": route
        }

        router = RunnableBranch(
            (lambda x: x["route"] == "tool", RunnableLambda(self._run_tool_chain)),
            (lambda x: x["route"] == "rag", RunnableLambda(self._run_rag_chain)),
            RunnableLambda(self._run_chat_chain),
        )

        return router.invoke(payload)

    def add_knowledge(self, evidence: Dict, chunks: List[Dict], route: str = "llm"):
        """Add new medical knowledge to the system (default into LLM store)"""
        logger.info(f"Adding new medical knowledge... route={route}")

        if route == "pico":
            db = self.db_pico
            vector_store = self.vector_store_pico
        else:
            db = self.db_llm
            vector_store = self.vector_store_llm

        evidence_id = db.insert_evidence(evidence)

        chunk_ids = []
        texts = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{evidence_id}_{i}"
            chunk_ids.append(chunk_id)

            db.insert_chunk(
                chunk_id=chunk_id,
                content=chunk['content'],
                metadata=chunk.get('metadata', {}),
                source_doc=chunk.get('source', 'manual')
            )

            texts.append(chunk['content'])
            metadatas.append(chunk.get('metadata', {}))
            ids.append(chunk_id)

        if texts:
            vector_store.add_documents(texts, metadatas, ids)

        evidence['chunk_ids'] = chunk_ids
        logger.info(f"Added evidence {evidence_id} with {len(chunks)} chunks")
        return evidence_id
