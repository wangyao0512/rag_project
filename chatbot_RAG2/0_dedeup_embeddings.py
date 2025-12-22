# -*- coding: utf-8 -*-
"""
Batch dedup JSON files by semantic similarity of {PICO, llm_text} using OpenAI embeddings.
Keep the newest year and higher GRADE when duplicates exist.

Outputs:
  output_dir/
    unique/      # kept jsons (same relative structure)
    duplicates/  # removed jsons
    dedup_report.jsonl  # mapping report
"""

import os
import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config import Config
import numpy as np
from tqdm import tqdm

from collections import defaultdict
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "Missing dependency: openai. Install with: pip install -U openai"
    ) from e


# ------------------------- Config defaults -------------------------
DEFAULT_MODEL = Config.EMBEDDING_MODEL   # 如果你实际模型名是 Qwen3-Embedding-8B，请运行时用 --model 覆盖
DEFAULT_SIM_THRESHOLD = 0.99       # 语义几乎相同的阈值，太严/太松都可调
DEFAULT_EMBED_BATCH = 12
MAX_TEXT_CHARS = 12000                # 防止极长文本造成请求失败（可按需调大）


# ------------------------- Helpers -------------------------
def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            if math.isnan(x):
                return default
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = -1.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, float, np.number)):
            v = float(x)
            if math.isnan(v):
                return default
            return v
        s = str(x).strip()
        if s == "":
            return default
        v = float(s)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def norm_text(s: str) -> str:
    # 用于“完全相同”快速判断（不影响语义去重）
    return " ".join(s.lower().split())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_json_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".json"


def build_pico_text(rec: Dict[str, Any]) -> str:
    """
    支持：
      - P/I/C/O 四字段
      - PICO 字段（str 或 dict）
    """
    if "PICO" in rec and rec["PICO"] is not None:
        pico = rec["PICO"]
        if isinstance(pico, str):
            return pico.strip()
        if isinstance(pico, dict):
            # 尽量按 P/I/C/O 排序拼接
            parts = []
            for k in ["P", "I", "C", "O"]:
                if k in pico and pico[k] is not None:
                    parts.append(f"{k}:{str(pico[k]).strip()}")
            # 再拼接其它键
            for k, v in pico.items():
                if k in ["P", "I", "C", "O"]:
                    continue
                if v is not None:
                    parts.append(f"{k}:{str(v).strip()}")
            return "\n".join(parts).strip()

    # fallback: P/I/C/O
    parts = []
    for k in ["P", "I", "C", "O"]:
        v = rec.get(k)
        if v is not None:
            parts.append(f"{k}:{str(v).strip()}")
    return "\n".join(parts).strip()


def build_embed_text(rec: Dict[str, Any]) -> str:
    pico = build_pico_text(rec)
    llm_text = str(rec.get("llm_text", "") or "").strip()
    text = f"{pico}\n\nllm_text:\n{llm_text}".strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


@dataclass
class Item:
    src_path: Path           # 原始文件路径
    rel_path: Path           # 相对 input_dir 的路径
    record: Dict[str, Any]   # json 内容（假设单条 dict；如果你有 list，会在 loader 里展开）
    year: int
    grade: float
    embed_text: str
    exact_hash: str
    vec: Optional[np.ndarray] = None


def load_items(input_dir: Path) -> List[Item]:
    items: List[Item] = []
    for p in input_dir.rglob("*.json"):
        if not is_json_file(p):
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 跳过坏文件
            continue

        rel = p.relative_to(input_dir)

        # 兼容：文件里是 list[dict] 的情况（展开成多条）
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            for idx, rec in enumerate(data):
                if not isinstance(rec, dict):
                    continue
                if "llm_text" not in rec:
                    continue
                embed_text = build_embed_text(rec)
                if not embed_text:
                    continue
                year = safe_int(rec.get("year", -1), -1)
                grade = safe_float(rec.get("GRADE", -1), -1.0)
                h = sha1(norm_text(embed_text))
                # 输出文件名：原文件名__idx.json
                rel2 = rel.with_name(f"{rel.stem}__{idx}{rel.suffix}")
                items.append(Item(p, rel2, rec, year, grade, embed_text, h))
        elif isinstance(data, dict):
            rec = data
            if "llm_text" not in rec:
                # 没有 llm_text 的文件就跳过
                continue
            embed_text = build_embed_text(rec)
            if not embed_text:
                continue
            year = safe_int(rec.get("year", -1), -1)
            grade = safe_float(rec.get("GRADE", -1), -1.0)
            h = sha1(norm_text(embed_text))
            items.append(Item(p, rel, rec, year, grade, embed_text, h))
        else:
            continue

    return items


# def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int) -> List[np.ndarray]:
#     vecs: List[np.ndarray] = []
#     i = 0
#     while i < len(texts):
#         chunk = texts[i:i + batch_size]
#         # OpenAI embeddings API: input can be list[str]
#         resp = client.embeddings.create(model=model, input=chunk)
#         # 保证按顺序
#         data_sorted = sorted(resp.data, key=lambda d: d.index)
#         for d in data_sorted:
#             vecs.append(np.array(d.embedding, dtype=np.float32))
#         i += batch_size
#     return vecs
def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int) -> List[np.ndarray]:
    vecs: List[np.ndarray] = []
    i = 0
    # 使用 tqdm 包装循环以展示进度条
    total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数
    with tqdm(total=total_batches, desc="Generating embeddings") as pbar:
        while i < len(texts):
            chunk = texts[i:i + batch_size]
            # OpenAI embeddings API: input can be list[str]
            resp = client.embeddings.create(model=model, input=chunk)
            # 保证按顺序
            data_sorted = sorted(resp.data, key=lambda d: d.index)
            for d in data_sorted:
                vecs.append(np.array(d.embedding, dtype=np.float32))
            i += batch_size
            pbar.update(1)  # 更新进度条
    return vecs

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return mat / denom


def main(
    input_dir: str,
    output_dir: str,
    base_url: Optional[str],
    api_key: Optional[str],
    model: str,
    sim_threshold: float,
    embed_batch: int,
):
    in_dir = Path(input_dir).resolve()
    out_dir = Path(output_dir).resolve()
    unique_dir = out_dir / "unique"
    dup_dir = out_dir / "duplicates"
    ensure_dir(unique_dir)
    ensure_dir(dup_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {in_dir}")

    # OpenAI client（支持你本地 OpenAI 兼容服务：设置 base_url 即可）
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url)

    items = load_items(in_dir)
    if not items:
        print("No valid items found (need json with llm_text).")
        return

    # 先按 (year, grade) 降序排序：这样相似项里“最优”会先入库，后来的重复直接丢弃
    items.sort(key=lambda x: (x.year, x.grade), reverse=True)

    # 先做“完全相同”快速去重（不花 embedding）
    kept: List[Item] = []
    exact_seen: Dict[str, Item] = {}
    exact_dups: List[Tuple[Item, Item]] = []  # (dropped, kept)

    for it in items:
        if it.exact_hash in exact_seen:
            exact_dups.append((it, exact_seen[it.exact_hash]))
        else:
            exact_seen[it.exact_hash] = it
            kept.append(it)

    items = kept  # 只对 exact 唯一的再做语义去重

    # Embedding
    texts = [it.embed_text for it in items]
    vecs = embed_texts(client, model=model, texts=texts, batch_size=embed_batch)
    for it, v in zip(items, vecs):
        it.vec = v

    # 语义去重（余弦相似度）
    kept_items: List[Item] = []
    kept_mat: Optional[np.ndarray] = None
    dropped: List[Tuple[Item, Item, float]] = []  # (dropped, kept, sim)

    for it in tqdm(items, desc="Processing semantic deduplication"):
        v = it.vec
        assert v is not None
        v = v.astype(np.float32)

        if not kept_items:
            kept_items.append(it)
            kept_mat = v.reshape(1, -1)
            continue

        # 计算与所有 kept 的 cosine similarity
        # 先 normalize
        if kept_mat is None:
            kept_mat = v.reshape(1, -1)
            kept_items.append(it)
            continue

        km = kept_mat
        km_n = l2_normalize(km)
        v_n = v / max(np.linalg.norm(v), 1e-12)
        sims = km_n @ v_n  # shape (k,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= sim_threshold:
            # 重复：丢弃 it（因为 items 已按 year/grade 排序，先出现的一定更优或不差）
            dropped.append((it, kept_items[best_idx], best_sim))
        else:
            kept_items.append(it)
            kept_mat = np.vstack([kept_mat, v.reshape(1, -1)])

    ensure_dir(out_dir)


    # 写出文件
    # 写出文件
    report_path = out_dir / "dedup_report.jsonl"
    with report_path.open("w", encoding="utf-8") as rf:
        # exact duplicates
        for d, k in exact_dups:
            # 保存到duplicates目录，使用扁平结构
            filename = d.rel_path.name  # 只使用文件名，不包含路径
            out_p = dup_dir / filename
            ensure_dir(out_p.parent)
            with out_p.open("w", encoding="utf-8") as f:
                json.dump(d.record, f, ensure_ascii=False, indent=2)
            rf.write(json.dumps({
                "type": "exact",
                "dropped": str(d.rel_path),
                "kept": str(k.rel_path),
                "sim": 1.0,
                "dropped_year": d.year,
                "dropped_grade": d.grade,
                "kept_year": k.year,
                "kept_grade": k.grade,
                "dropped_llm_text": d.record.get("llm_text", ""),
                "kept_llm_text": k.record.get("llm_text", "")
            }, ensure_ascii=False) + "\n")

        # semantic duplicates
        for d, k, s in dropped:
            # 保存到duplicates目录，使用扁平结构
            filename = d.rel_path.name  # 只使用文件名，不包含路径
            out_p = dup_dir / filename
            ensure_dir(out_p.parent)
            with out_p.open("w", encoding="utf-8") as f:
                json.dump(d.record, f, ensure_ascii=False, indent=2)
            rf.write(json.dumps({
                "type": "semantic",
                "dropped": str(d.rel_path),
                "kept": str(k.rel_path),
                "sim": round(s, 6),
                "dropped_year": d.year,
                "dropped_grade": d.grade,
                "kept_year": k.year,
                "kept_grade": k.grade,
                "dropped_llm_text": d.record.get("llm_text", ""),
                "kept_llm_text": k.record.get("llm_text", "")
            }, ensure_ascii=False) + "\n")

        # 按源文件路径分组保留的items
        source_files = defaultdict(list)
        
        # 将保留的items按源文件分组
        for item in kept_items:
            source_files[item.src_path].append(item)
        
        # 处理每个源文件，保存到unique目录
        for src_path, items in source_files.items():
            # 获取源文件名作为输出文件名
            filename = src_path.name
            out_p = unique_dir / filename
            ensure_dir(out_p.parent)
            
            # 按相对路径中的索引排序，确保顺序正确
            items.sort(key=lambda x: (
                int(x.rel_path.stem.split('__')[-1]) if '__' in x.rel_path.stem and 
                x.rel_path.stem.split('__')[-1].isdigit() else 0,
                x.rel_path.stem
            ))
            
            # 统一保存为数组格式
            records = [item.record for item in items]
            with out_p.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Input items (after loading): {len(load_items(in_dir))}")
    print(f"Unique kept: {len(kept_items)}")
    print(f"Dropped exact: {len(exact_dups)}")
    print(f"Dropped semantic: {len(dropped)}")
    print(f"Output unique dir: {unique_dir}")
    print(f"Output duplicates dir: {dup_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default='/data_vision9/ywang/heisesuliuMVP_PICO3rd_1210', help="Folder containing json files")
    ap.add_argument("--output_dir", default='/data_vision9/ywang/heisesuliuMVP_PICO3rd_1210_dedeup0_99', help="Output folder")
    ap.add_argument("--base_url", default=Config.EMBEDDING_BASE_URL, help="OpenAI compatible base_url")
    ap.add_argument("--api_key", default=Config.OPENAI_API_KEY, help="OpenAI API key (or set OPENAI_API_KEY)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_SIM_THRESHOLD, help="Cosine sim threshold for duplicates")
    ap.add_argument("--embed_batch", type=int, default=DEFAULT_EMBED_BATCH, help="Embedding batch size")
    args = ap.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        sim_threshold=args.sim_threshold,
        embed_batch=args.embed_batch,
    )
