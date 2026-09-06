"""
Step 4: 语义分块 — 将长文本切割成适合RAG和SFT的片段
策略：优先按段落/章节边界切分，再按字符数限制
用法：python 01_data_pipeline/04_chunk_text.py --input data/cleaned --output data/chunks
"""
import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from loguru import logger


CHAPTER_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千\d]+[章节卷篇]|[一二三四五六七八九十]+、|\d+[\.\、])",
    re.MULTILINE,
)


def split_by_paragraphs(text: str) -> List[str]:
    """按双换行拆段落"""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def merge_short_paragraphs(paragraphs: List[str], min_len: int = 100) -> List[str]:
    """将过短的段落合并到相邻段落"""
    merged = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < min_len:
            buffer += para + "\n"
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = para + "\n"
    if buffer:
        merged.append(buffer.strip())
    return merged


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source: str = "",
) -> List[Dict]:
    """
    将文本切分为带重叠的块。
    chunk_size: 每块最大字符数
    overlap: 相邻块重叠字符数（保持上下文连贯性）
    """
    paragraphs = split_by_paragraphs(text)
    paragraphs = merge_short_paragraphs(paragraphs, min_len=50)

    chunks = []
    buffer = ""
    chunk_idx = 0

    for para in paragraphs:
        # 如果单个段落已超过chunk_size，强制切割
        if len(para) > chunk_size:
            if buffer:
                chunks.append(_make_chunk(buffer, source, chunk_idx))
                chunk_idx += 1
                buffer = buffer[-overlap:] if overlap else ""
            # 逐字切割超长段落
            while len(para) > chunk_size:
                part = para[:chunk_size]
                chunks.append(_make_chunk(part, source, chunk_idx))
                chunk_idx += 1
                para = para[chunk_size - overlap:]
            buffer = para
            continue

        if len(buffer) + len(para) > chunk_size:
            if buffer:
                chunks.append(_make_chunk(buffer, source, chunk_idx))
                chunk_idx += 1
                # 保留末尾作为下一块的开头（overlap）
                buffer = buffer[-overlap:] + "\n" + para if overlap else para
            else:
                buffer = para
        else:
            buffer += ("\n" if buffer else "") + para

    if buffer.strip():
        chunks.append(_make_chunk(buffer, source, chunk_idx))

    return chunks


def _make_chunk(text: str, source: str, idx: int) -> Dict:
    text = text.strip()
    uid = hashlib.md5(f"{source}_{idx}_{text[:32]}".encode()).hexdigest()[:12]
    return {
        "id": uid,
        "source": source,
        "chunk_index": idx,
        "text": text,
        "char_count": len(text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned")
    parser.add_argument("--output", default="data/chunks")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))
    logger.info(f"分块 {len(txt_files)} 个文件，chunk_size={args.chunk_size}, overlap={args.overlap}")

    total_chunks = 0
    all_chunks_path = output_dir / "all_chunks.jsonl"

    with all_chunks_path.open("w", encoding="utf-8") as all_f:
        for txt_path in tqdm(txt_files, desc="分块"):
            text = txt_path.read_text(encoding="utf-8")
            chunks = chunk_text(
                text,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                source=txt_path.stem,
            )

            # 每本书单独一个文件
            per_book_path = output_dir / f"{txt_path.stem}.jsonl"
            with per_book_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    line = json.dumps(chunk, ensure_ascii=False)
                    f.write(line + "\n")
                    all_f.write(line + "\n")

            total_chunks += len(chunks)
            logger.info(f"  {txt_path.name}: {len(chunks)} 块")

    logger.info(f"✅ 共生成 {total_chunks} 个块 → {all_chunks_path}")


if __name__ == "__main__":
    main()
