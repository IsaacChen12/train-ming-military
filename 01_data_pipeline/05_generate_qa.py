"""
Step 5: 用LLM自动从文本块生成Q&A对，作为SFT训练数据
用法：python 01_data_pipeline/05_generate_qa.py \
        --input data/chunks --output data/qa_pairs \
        --model qwen3.6:35b --questions-per-chunk 3
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

SYSTEM_PROMPT = """你是一位专精明代军事史的学术助手。
你的任务是根据给定的史料原文，生成高质量的问答对，用于训练专业大模型。

要求：
1. 问题必须基于原文内容，具体明确，涉及明代军事装备、战术、制度等专业知识
2. 答案必须忠实于原文，用学术语言表述，不要凭空捏造
3. 避免过于简单的是非题，优先生成需要解释、比较、分析的问题
4. 每个问答对独立输出，格式严格按照要求

输出格式（JSON数组）：
[
  {"question": "问题内容", "answer": "答案内容"},
  ...
]
"""

USER_TEMPLATE = """请根据以下明代史料原文，生成 {n} 个高质量问答对：

---原文开始---
{text}
---原文结束---

直接输出JSON数组，不要有其他内容："""


def generate_qa_for_chunk(
    client: OpenAI,
    chunk: Dict,
    model: str,
    n_questions: int,
    max_retries: int = 3,
) -> List[Dict]:
    prompt = USER_TEMPLATE.format(text=chunk["text"], n=n_questions)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()

            # 提取JSON部分
            json_match = re.search(r"\[[\s\S]*\]", content)
            if not json_match:
                logger.warning(f"无法解析JSON（尝试 {attempt+1}）: {content[:100]}")
                continue

            qa_list = json.loads(json_match.group())
            results = []
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if q and a and len(a) > 20:
                    results.append({
                        "source": chunk["source"],
                        "chunk_id": chunk["id"],
                        "question": q,
                        "answer": a,
                    })
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败（尝试 {attempt+1}）: {e}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            time.sleep(2)

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/chunks")
    parser.add_argument("--output", default="data/qa_pairs")
    parser.add_argument("--model", default="qwen3.6:35b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="OpenAI兼容API地址（Ollama或vLLM）")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="限制处理块数（测试用）")
    parser.add_argument("--skip-short", type=int, default=100,
                        help="跳过字符数少于此值的块")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 读取所有块
    chunks_file = Path(args.input) / "all_chunks.jsonl"
    if not chunks_file.exists():
        # 尝试按书名分散读取
        chunk_files = list(Path(args.input).glob("*.jsonl"))
        chunks = []
        for f in chunk_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunks.append(json.loads(line))
    else:
        chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 过滤过短的块
    chunks = [c for c in chunks if c.get("char_count", len(c.get("text", ""))) >= args.skip_short]

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    logger.info(f"共 {len(chunks)} 个块，每块生成 {args.questions_per_chunk} 问，目标 {len(chunks) * args.questions_per_chunk} 条Q&A")

    all_qa = []
    output_file = output_dir / "all_qa.jsonl"

    with output_file.open("w", encoding="utf-8") as f_out:
        for chunk in tqdm(chunks, desc="生成Q&A"):
            qa_list = generate_qa_for_chunk(
                client, chunk, args.model,
                n_questions=args.questions_per_chunk,
            )
            for qa in qa_list:
                f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
            all_qa.extend(qa_list)
            time.sleep(0.2)  # 避免API限速

    logger.info(f"✅ 共生成 {len(all_qa)} 条Q&A → {output_file}")

    # 统计每本书的Q&A数量
    from collections import Counter
    source_counts = Counter(qa["source"] for qa in all_qa)
    for src, count in sorted(source_counts.items()):
        logger.info(f"  {src}: {count} 条")


if __name__ == "__main__":
    main()
