"""
Complete RAG pipeline
"""
from src.database.schema import MedicalDatabase
from src.retrieval.vector_store import VectorStore
from src.chat.llm_handler import QwenLLM
from src.chat.cot_reasoner import CoTReasoner
import json
from typing import List, Dict, Optional, Generator, Union
from loguru import logger
from config import Config
from typing import Any, Dict, Optional

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

        # Load entity mappings
        self.entity_mappings = self._load_entity_mappings()

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

    # def search_structured(self, entities: Dict[str, Optional[str]]) -> List[Dict]:
    #     """Search structured database"""
    #     disease = entities.get('disease')
    #     drug = entities.get('drug')
    #     population = entities.get('population')
    #
    #     results = self.db.search_evidence(
    #         disease=disease,
    #         drug=drug,
    #         limit=10
    #     )
    #
    #     logger.info(f"Found {len(results)} structured results")
    #     return results
    def search_structured(self, db: MedicalDatabase, entities: Dict[str, Optional[str]]) -> List[Dict]:
        disease = entities.get('disease')
        drug = entities.get('drug')

        results = db.search_evidence(disease=disease, drug=drug, limit=10)
        logger.info(f"Found {len(results)} structured results")
        return results

    def get_chunks_from_evidence(self, evidence_list: List[Dict], max_chunks_per_result: int = 2) -> List[Dict]:
        """Get text chunks from evidence results"""
        all_chunks: List[Dict] = []

        for result in evidence_list:
            chunk_ids = json.loads(result.get('chunk_ids', '[]'))

            for chunk_id in chunk_ids[:max_chunks_per_result]:
                chunk = self.db.get_chunk(chunk_id)
                if chunk:
                    all_chunks.append({
                        'content': chunk['content'],
                        'metadata': json.loads(chunk.get('metadata', '{}')),
                        'source': chunk.get('source_doc', 'unknown')
                    })

        return all_chunks

    def search_vector(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search vector database with fallback to broader query"""
        try:
            # First attempt: search with original query
            results = self.vector_store.search(query, n_results=n_results)

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

            # If results are poor quality (high distance) or too few, try broader search
            if not chunks or (chunks and chunks[0].get('distance', 0) > 0.5):
                logger.info(f"Initial search quality low, trying broader query...")
                # Extract key medical terms and search again
                broader_query = self._extract_core_medical_terms(query)
                if broader_query != query:
                    logger.info(f"Broader query: {broader_query}")
                    broader_results = self.vector_store.search(broader_query, n_results=n_results)

                    if broader_results and broader_results.get('documents'):
                        broader_distances = broader_results.get('distances', [[]])[0]
                        for i, doc in enumerate(broader_results['documents'][0]):
                            metadata = broader_results['metadatas'][0][i] if broader_results.get('metadatas') else {}
                            distance = broader_distances[i] if i < len(broader_distances) else 999
                            # Only add if not already in chunks
                            doc_chunk = {
                                'content': doc,
                                'metadata': metadata,
                                'source': metadata.get('source', 'vector_db'),
                                'distance': distance
                            }
                            if doc not in [c['content'] for c in chunks]:
                                chunks.append(doc_chunk)

                # Sort by distance (lower is better)
                chunks.sort(key=lambda x: x.get('distance', 999))

            logger.info(f"Found {len(chunks)} vector search results")

            # Rerank by GRADE if we have enough results
            if len(chunks) > 0:
                chunks = self._rerank_by_grade(chunks, Config.VECTOR_SEARCH_TOP_K)

            return chunks

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _rerank_by_grade(self, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Rerank chunks by combining similarity and GRADE scores

        GRADE is numeric (1-10):
        - 10: Highest quality evidence (e.g., NCCN guidelines)
        - 9: High quality systematic reviews
        - 7-8: Strong consensus/guidelines
        - 5-6: Moderate quality consensus
        - 3-4: Low quality or expert opinion
        - 1-2: Very low quality
        """
        # Get GRADE from metadata (already stored during ingestion)
        for chunk in chunks:
            metadata = chunk.get('metadata', {})

            # Try to get GRADE from metadata first (most reliable)
            grade = metadata.get('grade')
            cluster = metadata.get('guideline_focus')
             
            
            # If not in metadata, try to look up from database
            if grade is None:
                chunk_id = metadata.get('chunk_id', '')
                try:
                    # Find the evidence entry for this chunk
                    evidence_row = self.db.conn.execute("""
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

            # Store in chunk for visibility
            chunk['grade'] = grade if grade is not None else 5  # Default to medium quality
            chunk['guideline_focus'] = cluster

        # Calculate combined score: relevance (60%) + GRADE (40%)
        # Increased GRADE weight from 30% to 40% to prioritize evidence quality

        for chunk in chunks:
            distance = chunk.get('distance', 1.0)
            grade = chunk.get('grade', 5)

            # Normalize GRADE (1-10) to 0-1 scale
            grade_score = (grade - 1) / 9.0  # Maps 1->0.0, 10->1.0

            # Combined score (lower is better for distance, higher is better for grade)
            # Normalize distance to 0-1 range (assuming max distance ~1.5)
            normalized_distance = min(distance / 1.5, 1.0)
            relevance_score = 1.0 - normalized_distance

            # Weighted combination: 60% relevance + 40% GRADE
            chunk['combined_score'] = 0.8 * relevance_score + 0.2 * grade_score

        # Sort by combined score (higher is better)
        chunks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        if chunks:
            logger.info(f"Reranked by GRADE. Top result: GRADE={chunks[0].get('grade')}, "
                       f"cluster={chunks[0].get('guideline_focus')}, score={chunks[0].get('combined_score', 0):.3f}")

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
                      extract_entities: bool = True, use_cot: bool = True) -> Union[Dict, Generator]:
        logger.info(f"Processing query: {query}")

        # ✅ Step 0.1：先路由（有PICO就走PICO库）
        route_info = self._route_store(query)
        db = route_info["db"]
        vector_store = route_info["vector_store"]
        pico = route_info["pico"]
        route = route_info["route"]
        logger.info(f"[RAG route] route={route}, pico_keys={list(pico.keys())}")

        # Step 0: CoT
        cot_analysis = None
        if use_cot:
            cot_analysis = self.cot_reasoner.analyze_query(query)
            if self.cot_reasoner.should_refuse(cot_analysis):
                refusal_msg = self.cot_reasoner.get_refusal_message(cot_analysis)
                return {
                    'query': query,
                    'pico': pico,
                    'route': route,
                    'entities': {},
                    'normalized_entities': {},
                    'answer': refusal_msg,
                    'sources': [],
                    'num_sources': 0,
                    'cot_analysis': cot_analysis,
                    'refused': True
                }

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

        # ✅ Step 4：向量检索用“选中的 vector_store + db”
        vector_chunks = self.search_vector(vector_store, db, query, n_results=max_context_chunks * 2)

        if extract_entities and entities and any(entities.values()):
            vector_chunks = self.filter_chunks_by_entities(vector_chunks, entities)

        all_chunks = self.combine_results(structured_chunks, vector_chunks, max_total=max_context_chunks)

        if stream:
            def generate():
                for token in self.llm.generate_answer(query, all_chunks, stream=True):
                    yield {
                        'token': token,
                        'route': route,
                        'pico': pico,
                        'entities': entities,
                        'normalized_entities': normalized_entities,
                        'sources': all_chunks,
                        'num_sources': len(all_chunks),
                        'cot_analysis': cot_analysis
                    }

            return generate()
        else:
            answer = self.llm.generate_answer(query, all_chunks, stream=False)
            return {
                'query': query,
                'route': route,
                'pico': pico,
                'entities': entities,
                'normalized_entities': normalized_entities,
                'answer': answer,
                'sources': all_chunks,
                'num_sources': len(all_chunks),
                'cot_analysis': cot_analysis,
                'refused': False
            }

    # def process_query(self, query: str, max_context_chunks: int = 5, stream: bool = False,
    #                   extract_entities: bool = True, use_cot: bool = True) -> Union[Dict, Generator]:
    #     """Main RAG pipeline with Chain-of-Thought reasoning"""
    #     logger.info(f"Processing query: {query}")

    #     # Step 0: Chain-of-Thought Analysis
    #     cot_analysis = None
    #     if use_cot:
    #         logger.info("Performing CoT analysis...")
    #         cot_analysis = self.cot_reasoner.analyze_query(query)
    #         logger.info(f"CoT Result: is_medical={cot_analysis['is_medical']}, reasoning={cot_analysis['reasoning']}")
    #         # Check if query should be refused
    #         if self.cot_reasoner.should_refuse(cot_analysis):
    #             logger.info("Query refused: non-medical content")
    #             refusal_msg = self.cot_reasoner.get_refusal_message(cot_analysis)
    #             return {
    #                 'query': query,
    #                 'entities': {},
    #                 'normalized_entities': {},
    #                 'answer': refusal_msg,
    #                 'sources': [],
    #                 'num_sources': 0,
    #                 'cot_analysis': cot_analysis,
    #                 'refused': True
    #             }

    #     # Step 1: Extract entities (enabled by default)
    #     entities = {}
    #     normalized_entities = {}

    #     if extract_entities:
    #         logger.info("Extracting entities...")
    #         entities = self.llm.extract_entities(query)
    #         logger.info(f"Extracted entities: {entities}")

    #         # Step 2: Normalize entities
    #         for key, value in entities.items():
    #             if value:
    #                 normalized_entities[key] = self.normalize_entity(value)

    #         # Step 3: Structured search with entities (only if we have entities to search for)
    #         if any(entities.values()):
    #             logger.info("Performing structured search...")
    #             structured_results = self.search_structured(entities)
    #             structured_chunks = self.get_chunks_from_evidence(structured_results)
    #         else:
    #             logger.info("Skipping structured search (no entities extracted)")
    #             structured_chunks = []
    #     else:
    #         # Skip entity extraction if explicitly disabled
    #         logger.info("Skipping entity extraction (disabled)...")
    #         structured_chunks = []

    #     # Step 4: Vector search (main retrieval method)
    #     logger.info("Performing vector search...")
    #     vector_chunks = self.search_vector(query, n_results=max_context_chunks * 2)  # Get more for filtering

    #     # Step 4.5: Filter vector results by entities (only if we actually extracted any entities)
    #     if extract_entities and entities and any(entities.values()):
    #         logger.info("Filtering vector results by extracted entities...")
    #         vector_chunks = self.filter_chunks_by_entities(vector_chunks, entities)

    #     # Step 5: Combine results
    #     logger.info("Combining results...")
    #     all_chunks = self.combine_results(
    #         structured_chunks,
    #         vector_chunks,
    #         max_total=max_context_chunks
    #     )

    #     # Step 6: Generate answer
    #     logger.info("Generating answer...")

    #     if stream:
    #         # Return generator for streaming
    #         def generate():
    #             for token in self.llm.generate_answer(query, all_chunks, stream=True):
    #                 yield {
    #                     'token': token,
    #                     'entities': entities,
    #                     'normalized_entities': normalized_entities,
    #                     'sources': all_chunks,
    #                     'num_sources': len(all_chunks),
    #                     'cot_analysis': cot_analysis
    #                 }
    #         return generate()
    #     else:
    #         # Return complete result
    #         answer = self.llm.generate_answer(query, all_chunks, stream=False)
    #         result = {
    #             'query': query,
    #             'entities': entities,
    #             'normalized_entities': normalized_entities,
    #             'answer': answer,
    #             'sources': all_chunks,  # Return all sources (up to max_context_chunks)
    #             'num_sources': len(all_chunks),
    #             'cot_analysis': cot_analysis,
    #             'refused': False
    #         }
    #         logger.info("Query processing complete")
    #         return result

    def add_knowledge(self, evidence: Dict, chunks: List[Dict]):
        """Add new medical knowledge to the system"""
        logger.info("Adding new medical knowledge...")

        # Add evidence to database
        evidence_id = self.db.insert_evidence(evidence)

        # Add chunks to database and vector store
        chunk_ids = []
        texts = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{evidence_id}_{i}"
            chunk_ids.append(chunk_id)

            # Add to SQL database
            self.db.insert_chunk(
                chunk_id=chunk_id,
                content=chunk['content'],
                metadata=chunk.get('metadata', {}),
                source_doc=chunk.get('source', 'manual')
            )

            # Prepare for vector store
            texts.append(chunk['content'])
            metadatas.append(chunk.get('metadata', {}))
            ids.append(chunk_id)

        # Add to vector store
        if texts:
            self.vector_store.add_documents(texts, metadatas, ids)

        # Update evidence with chunk IDs
        evidence['chunk_ids'] = chunk_ids

        logger.info(f"Added evidence {evidence_id} with {len(chunks)} chunks")
        return evidence_id
