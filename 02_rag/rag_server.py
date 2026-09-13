"""
RAG Step 2: FastAPI RAG服务
提供 OpenAI 兼容的 /v1/chat/completions 接口，内部执行检索增强生成
用法：python 02_rag/rag_server.py --db-path data/chroma_db --port 8080
"""
import argparse
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

COLLECTION_NAME = "ming_military"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 5

RAG_SYSTEM = """你是一位专精明代军事史的学术助手，擅长明朝军事装备、兵器制造、军事制度等领域。

请基于以下检索到的史料原文回答问题。回答时：
1. 优先引用检索到的原文内容，保持学术准确性
2. 如果原文信息不足，可以结合背景知识补充，但需明确说明
3. 使用规范的学术语言，必要时说明文献来源
4. 如果问题超出明代军事范围，礼貌说明

检索到的相关史料：
{context}
"""

app = FastAPI(title="明代军事装备RAG服务", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 全局组件（启动时初始化）
chroma_collection = None
embed_model = None
llm_client = None
llm_model_name = None


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    query_embedding = embed_model.encode(
        [query], normalize_embeddings=True
    ).tolist()
    results = chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": round(1 - dist, 4),
        })
    return items


@app.get("/health")
def health():
    return {"status": "ok", "collection_count": chroma_collection.count()}


@app.post("/v1/search")
def search(req: SearchRequest):
    """仅检索，不生成，用于调试"""
    results = retrieve(req.query, req.top_k)
    return {"results": results}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # 提取最后一个用户消息作为检索查询
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="没有用户消息")
    query = user_messages[-1].content

    # 检索
    retrieved = retrieve(query, TOP_K)
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(f"[{i}] 来源：{r['source']}（相关度：{r['score']}）\n{r['text']}")
    context = "\n\n".join(context_parts)

    # 构建增强后的消息
    system_content = RAG_SYSTEM.format(context=context)
    augmented_messages = [{"role": "system", "content": system_content}]
    for m in req.messages:
        augmented_messages.append({"role": m.role, "content": m.content})

    # 调用LLM
    response = llm_client.chat.completions.create(
        model=llm_model_name,
        messages=augmented_messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    answer = response.choices[0].message.content
    # 附加检索来源信息
    sources_info = "\n\n---\n**参考来源：**\n" + "\n".join(
        f"- [{r['source']}] 相关度 {r['score']}" for r in retrieved
    )

    return {
        "id": response.id,
        "object": "chat.completion",
        "model": req.model or llm_model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer + sources_info,
            },
            "finish_reason": "stop",
        }],
        "retrieved_contexts": retrieved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/chroma_db")
    parser.add_argument("--embedding-model", default=EMBED_MODEL)
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-api-key", default="ollama")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    global chroma_collection, embed_model, llm_client, llm_model_name

    logger.info(f"加载嵌入模型: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model, trust_remote_code=True)

    logger.info(f"连接ChromaDB: {args.db_path}")
    client = chromadb.PersistentClient(
        path=args.db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_collection = client.get_collection(COLLECTION_NAME)
    logger.info(f"集合大小: {chroma_collection.count()} 条")

    llm_client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
    llm_model_name = args.model

    logger.info(f"RAG服务启动: http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
