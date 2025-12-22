"""
Lightweight vector store using ChromaDB + OpenAI-compatible Qwen embeddings
"""
import chromadb
from typing import List, Dict, Optional
import os
import numpy as np  # 新增：用于距离计算
from loguru import logger
from openai import OpenAI
from config import Config
from typing import Any

class RemoteEmbedder:
    """
    简单封装：提供 encode(texts) 接口，
    内部走 OpenAI Embeddings -> Qwen3-Embedding-8B。
    """

    def __init__(
        self,
        base_url: str = Config.EMBEDDING_BASE_URL,
        api_key: str = Config.OPENAI_API_KEY,
        model: str = Config.EMBEDDING_MODEL,
    ):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self.model = model
        logger.info(f"RemoteEmbedder initialized: base_url={base_url}, model={model}")

    def encode(self, texts: List[str], show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = [d.embedding for d in resp.data]
        return embeddings


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "database/chroma",
        model_path: Optional[str] = None,  # 兼容老参数，不再使用本地模型
        embedding_base_url: str = Config.EMBEDDING_BASE_URL,
        embedding_model: str = Config.EMBEDDING_MODEL,
        api_key: str = Config.OPENAI_API_KEY,
    ):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="medical_chunks",
            metadata={"hnsw:space": "cosine"},  # cosine距离：值越小越相似
        )

        self.model_path = model_path  # 仅为兼容

        self.embedder: Optional[RemoteEmbedder] = None
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.api_key = api_key

        # 新增：默认去重阈值（可根据业务调整，越小越严格）
        self.distance_threshold = getattr(Config, "DISTANCE_THRESHOLD", 0.001)

        self._load_embedder()

    def _load_embedder(self):
        try:
            logger.info(
                f"Loading remote embedding model: {self.embedding_model} "
                f"from {self.embedding_base_url}"
            )
            self.embedder = RemoteEmbedder(
                base_url=self.embedding_base_url,
                api_key=self.api_key,
                model=self.embedding_model,
            )
            logger.info("Embedding model (remote Qwen) loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        if not self.embedder:
            self._load_embedder()

        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            logger.info(f"Generating embeddings for batch {i//batch_size + 1}")
            # 修复：encode返回列表，无需tolist()
            embeddings = self.embedder.encode(batch_texts, show_progress_bar=False)

            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        logger.info(f"Added {len(texts)} documents to vector store")

    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None  # 支持自定义去重阈值
    ) -> Dict:
        """
        Search for similar documents using semantic similarity, remove extremely similar chunks

        Args:
            query: Query text to search for
            n_results: Number of unique results to return
            filter_dict: Optional metadata filters
            distance_threshold: Threshold for removing similar chunks (cosine distance, smaller=more strict)
                                Defaults to self.distance_threshold (0.05)

        Returns:
            Dictionary containing matched documents, distances, and metadata
        """
        if not self.embedder:
            self._load_embedder()

        # 使用自定义阈值或默认阈值
        threshold = distance_threshold or self.distance_threshold
        # 初始查询更多候选（目标数量的2倍），保证去重后有足够结果
        initial_candidate_num = n_results * 2

        # 生成查询向量
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        
        # 构建查询参数
        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": initial_candidate_num
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        # 初始查询（获取更多候选）
        results = self.collection.query(**kwargs)
        
        # 提取第一个查询的结果（单查询场景）
        docs = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        dists = results["distances"][0] if results.get("distances") and results["distances"][0] else []
        metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else []
        ids = results["ids"][0] if results.get("ids") and results["ids"][0] else []

        # 去重逻辑：保留与已选结果距离差大于阈值的chunk
        retained_docs = []
        retained_dists = []
        retained_metas = []
        retained_ids = []

        for doc, dist, meta, id_ in zip(docs, dists, metas, ids):
            # 跳过空文档
            if not doc:
                continue
            
            # 判断当前chunk是否与已保留的chunk极其相似
            is_duplicate = False
            for retained_dist in retained_dists:
                # 余弦距离差小于阈值 → 极其相似，视为重复
                if abs(dist - retained_dist) < threshold:
                    is_duplicate = True
                    logger.debug(f"Skipping duplicate chunk (ID: {id_}) - distance diff < {threshold}")
                    break
            
            # 非重复则保留
            if not is_duplicate:
                retained_docs.append(doc)
                retained_dists.append(dist)
                retained_metas.append(meta)
                retained_ids.append(id_)
            
            # 达到目标数量则提前终止
            if len(retained_docs) >= n_results:
                break

        # 截取最终结果（最多n_results个）
        final_n = min(len(retained_docs), n_results)
        final_results = {
            "ids": [retained_ids[:final_n]],
            "documents": [retained_docs[:final_n]],
            "metadatas": [retained_metas[:final_n]],
            "distances": [retained_dists[:final_n]],
            # 保留原结果的其他字段（如embeddings）
            **{k: v for k, v in results.items() if k not in ["ids", "documents", "metadatas", "distances"]}
        }

        logger.info(
            f"Search completed: {len(docs)} initial candidates → "
            f"{final_n} unique chunks (target: {n_results}, threshold: {threshold})"
        )
        return final_results

    def count(self) -> int:
        return self.collection.count()

    def delete_all(self) -> None:
        """Delete all documents from collection"""
        self.client.delete_collection("medical_chunks")
        self.collection = self.client.get_or_create_collection(
            name="medical_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Deleted all documents from vector store")