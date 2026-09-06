"""
RAG Step 1: 构建ChromaDB向量索引
使用 BAAI/bge-m3 多语言嵌入模型（中文效果最佳）
用法：python 02_rag/build_index.py --chunks data/chunks --db-path data/chroma_db
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "ming_military"
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 64


def load_chunks(chunks_dir: Path) -> list:
    all_chunks = []
    chunks_file = chunks_dir / "all_chunks.jsonl"
    if chunks_file.exists():
        for line in chunks_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_chunks.append(json.loads(line))
    else:
        for f in chunks_dir.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_chunks.append(json.loads(line))
    return all_chunks


def build_index(chunks_dir: Path, db_path: Path, embed_model_name: str = EMBED_MODEL):
    logger.info(f"加载嵌入模型: {embed_model_name}")
    model = SentenceTransformer(embed_model_name, trust_remote_code=True)

    logger.info(f"初始化ChromaDB: {db_path}")
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # 如果已存在则删除重建
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"已删除旧集合 {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = load_chunks(chunks_dir)
    logger.info(f"共 {len(chunks)} 个文本块待索引")

    # 批量编码和入库
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="构建索引"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "source": c.get("source", ""),
                "chunk_index": c.get("chunk_index", 0),
                "char_count": c.get("char_count", len(c["text"])),
            }
            for c in batch
        ]

        # bge-m3 推荐的查询前缀
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    count = collection.count()
    logger.info(f"✅ 索引构建完成，共 {count} 条记录 → {db_path}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks")
    parser.add_argument("--db-path", default="data/chroma_db")
    parser.add_argument("--embedding-model", default=EMBED_MODEL)
    args = parser.parse_args()

    build_index(
        chunks_dir=Path(args.chunks),
        db_path=Path(args.db_path),
        embed_model_name=args.embedding_model,
    )


if __name__ == "__main__":
    main()
