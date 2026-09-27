# 明朝军事装备专域大模型 / Ming Dynasty Military Equipment Domain LLM

授权：陈方郡
Authorized by: Chen Fangjun

基于 **RAG + LoRA微调** 的明代军事史专业问答系统。经 [机器A：RTX 4090 Laptop 16GB 实测](benchmark/reports/rtx4090-laptop-16gb.md) 验证，微调基座与线上推理默认模型均为 **Qwen3.5:9B**（16GB显卡即可训练+部署，速度快），另有 **Qwen3.6:35B MoE** 作为质量优先的RAG部署候选（无需微调，直接挂知识库），详见 [RUNBOOK.md](RUNBOOK.md) 阶段5部署说明。项目还在 [机器B：RTX 3060 12GB](benchmark/reports/rtx3060-12gb.md) 上做了同一批基准测试——模型排名并不相同，12GB显存下 `qwen2.5:7b` 反而是速度最快的选项。

A domain-specific Q&A system for Ming dynasty military history, built on **RAG + LoRA fine-tuning**. Verified on [Machine A: RTX 4090 Laptop 16GB](benchmark/reports/rtx4090-laptop-16gb.md), the fine-tuning base model and the default online inference model are both **Qwen3.5:9B** (trainable and deployable on a 16GB GPU, fast). **Qwen3.6:35B MoE** is also available as a quality-first RAG deployment candidate (no fine-tuning needed, plugs straight into the knowledge base) — see the Stage 5 deployment notes in [RUNBOOK.md](RUNBOOK.md). The same benchmark suite was also run on [Machine B: RTX 3060 12GB](benchmark/reports/rtx3060-12gb.md) — the model ranking is different there: with only 12GB VRAM, `qwen2.5:7b` turns out to be the fastest option.

## 快速开始 / Quick Start

```bash
# 1. 环境初始化（WSL Ubuntu）/ Step 1: Environment setup (WSL Ubuntu)
bash 00_setup/wsl_setup.sh

# 2. 将书籍PDF放入 data/raw/，然后运行数据管线 / Step 2: Put book PDFs into data/raw/, then run the data pipeline
bash 01_data_pipeline/run_pipeline.sh

# 3. 构建RAG索引并启动服务 / Step 3: Build the RAG index and start the service
bash 02_rag/build_and_serve.sh

# 4. （可选）微调训练 / Step 4: (Optional) fine-tuning
bash 03_finetune/train.sh

# 5. 一键部署 / Step 5: One-click deployment
bash 05_deployment/deploy.sh
```

## 当前语料统计 / Current Corpus Statistics

截至 2026-09-27，`data/chunks/` 中已处理的素材如下：

As of 2026-09-27, the material processed into `data/chunks/` is as follows:

| 素材 / Source | 块数 / Chunks | 字符数 / Characters |
|---|---|---|
| 《营要事宜》中的明军阵列复原与同期瑞典军队的对战模拟 | 19 | 7,758 |
| 中欧火绳枪的比较——以现有实验为基础的威力推算 | 69 | 28,432 |
| 大陆两端的纯步兵详解——从编制到战术 | 178 | 73,865 |
| 梦到什么写什么——从车营到步火营 | 50 | 20,247 |
| **合计 / Total** | **316** | **130,302** |

> 字符数为 `04_chunk_text.py` 的 `char_count` 口径（`len(text)`，含标点/数字/英文/换行，非纯中文字数；纯汉字约98,571个，占比约76%）。
>
> The character count uses the `char_count` field from `04_chunk_text.py` (`len(text)` — includes punctuation, digits, English letters, and newlines, not just Chinese characters; pure Han characters are ~98,571, about 76% of the total).

> 📌 **维护规则 / Maintenance rule**：每次向 `data/raw/` 添加新素材并跑通分块流程后，必须同步更新本节与 [RUNBOOK.md](RUNBOOK.md) 阶段1的语料统计小节，保持文档与实际数据一致。
>
> Whenever new material is added to `data/raw/` and run through the chunking pipeline, this section and the corpus-statistics subsection in Stage 1 of [RUNBOOK.md](RUNBOOK.md) must be updated at the same time, so the docs stay in sync with the actual data.

## 目录结构 / Directory Structure

```
train-ming-military/
├── RUNBOOK.md                    ← 完整操作手册（主要文档）/ Full operation manual (primary doc)
├── README.md                     ← 本文件 / This file
├── 00_setup/
│   ├── wsl_setup.sh              ← WSL环境一键初始化 / One-click WSL environment setup
│   ├── requirements.txt          ← Python依赖 / Python dependencies
│   └── docker-compose.yml        ← ChromaDB + Open-WebUI
├── 01_data_pipeline/
│   ├── 01_extract_pdf.py         ← PDF文字提取 / PDF text extraction
│   ├── 02_ocr_images.py          ← OCR扫描识别（PaddleOCR）/ OCR for scans (PaddleOCR)
│   ├── 03_clean_text.py          ← 文本清洗 / Text cleaning
│   ├── 04_chunk_text.py          ← 语义分块 / Semantic chunking
│   ├── 05_generate_qa.py         ← LLM自动生成Q&A对 / LLM auto-generated Q&A pairs
│   └── run_pipeline.sh           ← 一键运行数据管线 / One-click data pipeline runner
├── 02_rag/
│   ├── build_index.py            ← 构建ChromaDB向量索引 / Build the ChromaDB vector index
│   ├── rag_server.py             ← FastAPI RAG服务 / FastAPI RAG service
│   ├── test_rag.py               ← 交互式测试 / Interactive testing
│   └── build_and_serve.sh        ← 一键构建和启动 / One-click build and serve
├── 03_finetune/
│   ├── prepare_sft.py            ← 准备SFT训练数据 / Prepare SFT training data
│   ├── qwen35_lora.yaml          ← LLaMA-Factory训练配置（Qwen3.5-9B）/ LLaMA-Factory training config (Qwen3.5-9B)
│   ├── train.sh                  ← 训练脚本 / Training script
│   └── merge_lora.sh             ← 合并LoRA权重 / Merge LoRA weights
├── 04_evaluation/
│   ├── create_testset.py         ← 生成测试集模板 / Generate a test set template
│   ├── eval_model.py             ← 评估模型（ROUGE + LLM-as-Judge）/ Evaluate the model (ROUGE + LLM-as-Judge)
│   ├── eval_rag.py               ← 评估RAG系统 / Evaluate the RAG system
│   └── compare.py                ← 三路对比报告 / Three-way comparison report
├── 05_deployment/
│   ├── Modelfile                 ← Ollama模型配置 / Ollama model config
│   └── deploy.sh                 ← 一键部署 / One-click deployment
└── benchmark/
    ├── run_benchmark.py          ← 本地Ollama模型基准测试脚本（跑完后需手动按机型归档，见下）/ Local Ollama benchmarking script (results must be manually archived per machine afterward, see below)
    ├── reports/                  ← 按机型命名的报告（rtx4090-laptop-16gb.md、rtx3060-12gb.md）/ Reports named per machine (rtx4090-laptop-16gb.md, rtx3060-12gb.md)
    └── raw/                      ← 各模型原始测试数据，按机型分子目录归档（rtx4090-laptop-16gb/、rtx3060-12gb/）/ Raw per-model test data, archived in per-machine subdirectories (rtx4090-laptop-16gb/, rtx3060-12gb/)
```

## 核心技术选型 / Core Technology Choices

| 组件 / Component | 选型 / Choice | 理由 / Rationale |
|------|------|------|
| 微调基座 / Fine-tuning base | Qwen3.5-9B-Instruct | 4090（16GB）实测QLoRA训练显存仅需10-13GB，训练与部署统一到同一硬件规格 / Measured on a 4090 (16GB): QLoRA training needs only 10-13GB VRAM, so training and deployment share the same hardware spec |
| RAG部署候选（不微调）/ RAG deployment candidate (no fine-tuning) | Qwen3.6:35B MoE | 见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)：回答最详尽，质量优先场景直接挂知识库使用 / See [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md): gives the most detailed answers, used directly with the knowledge base for quality-first scenarios |
| 嵌入模型 / Embedding model | BAAI/bge-m3 | 中文检索SOTA，支持长文本 / SOTA for Chinese retrieval, supports long text |
| 向量数据库 / Vector database | ChromaDB | 轻量易用，支持持久化 / Lightweight, easy to use, supports persistence |
| 微调框架 / Fine-tuning framework | LLaMA-Factory | 支持Qwen系列，QLoRA省显存 / Supports the Qwen family, QLoRA saves VRAM |
| 推理服务 / Inference service | Ollama | 本地部署，OpenAI兼容API / Local deployment, OpenAI-compatible API |
| OCR引擎 / OCR engine | PaddleOCR | 中文识别率高 / High Chinese recognition accuracy |

## 机器A：RTX 4090 Laptop（16GB VRAM）候选模型 / Machine A: RTX 4090 Laptop (16GB VRAM) Candidate Models

在 RTX 4090 Laptop（16GB VRAM）上对本地 Ollama 模型做了基准测试（[完整报告](benchmark/reports/rtx4090-laptop-16gb.md)），综合速度、显存占用与回答质量后确定两个部署候选：

Local Ollama models were benchmarked on an RTX 4090 Laptop (16GB VRAM) ([full report](benchmark/reports/rtx4090-laptop-16gb.md)). Weighing speed, VRAM usage, and answer quality together, two deployment candidates were selected:

| 候选模型 / Candidate | 生成速度 / Speed | VRAM占用 / VRAM | 适用场景 / Use case |
|----------|----------|----------|----------|
| `qwen3.5:9b` | 82.6 tok/s | ~6.6 GB | 微调基座 + 日常问答，速度优先，可与其他模型共存显存 / Fine-tuning base + everyday Q&A, speed-first, can share VRAM with other models |
| `qwen3.6:35b`（MoE） | 53.6 tok/s | ~23 GB | 深度分析/SFT数据生成，回答更详尽，不参与微调 / Deep analysis / SFT data generation, more detailed answers, not used for fine-tuning |

两者的 Open-WebUI 部署配置与差异说明见 [RUNBOOK.md](RUNBOOK.md) 阶段5.2/5.3。

See Stage 5.2/5.3 in [RUNBOOK.md](RUNBOOK.md) for the Open-WebUI deployment configs for both, and how they differ.

## 机器B：RTX 3060（12GB VRAM）候选模型 / Machine B: RTX 3060 (12GB VRAM) Candidate Models

在 RTX 3060（12GB VRAM）上对本地已装的5个模型做了基准测试（[完整报告](benchmark/reports/rtx3060-12gb.md)）。**排名与机器A不同**——显存更小，模型越大掉速越明显：

Five locally installed models were benchmarked on an RTX 3060 (12GB VRAM) ([full report](benchmark/reports/rtx3060-12gb.md)). **The ranking differs from Machine A** — with less VRAM, larger models slow down more noticeably:

| 候选模型 / Candidate | 生成速度 / Speed | VRAM占用 / VRAM | 适用场景 / Use case |
|----------|----------|----------|----------|
| `qwen2.5:7b` | **65.8 tok/s**（最快 / fastest） | ~4.7 GB | 速度优先场景，机器A未测试过此模型 / Speed-first scenarios; not tested on Machine A |
| `qwen3.5:9b` | 53.9 tok/s | ~6.6 GB | 与机器A一致的微调基座/RAG默认，保持训练管线统一 / Same fine-tuning base/RAG default as Machine A, keeps the training pipeline consistent |
| `qwen3.6:35b`（MoE） | 38.8 tok/s（机器A为53.6 / 53.6 on Machine A） | ~22 GB（CPU offload占比更高 / higher CPU offload share） | 质量优先，但比机器A慢约28%，仅建议离线/非交互场景 / Quality-first, but ~28% slower than Machine A; recommended only for offline/non-interactive use |
| `gemma4:12b` | 37.8 tok/s（机器A为58.8 / 58.8 on Machine A） | ~7.6 GB（完全放得进显存，无需offload / fits fully in VRAM, no offload needed） | 与`qwen3.6:35b`速度几乎打平，但体积小1/3且无offload，可作为质量优先的备选 / Nearly matches `qwen3.6:35b` in speed but is 1/3 the size with no offload — a viable quality-first alternative |

> ⚠️ **微调需注意 / Fine-tuning caveat：** `qwen3.5:9b` QLoRA训练显存估算10-13GB，在12GB显卡上偏紧（上限已超出可用显存），建议减小 `cutoff_len` 或开启梯度检查点，且需实测验证是否OOM——本次基准仅覆盖推理，未覆盖训练。
> Estimated `qwen3.5:9b` QLoRA training VRAM is 10-13GB, which is tight on a 12GB card (the upper end exceeds available VRAM). Consider reducing `cutoff_len` or enabling gradient checkpointing, and verify empirically whether OOM occurs — this benchmark only covered inference, not training.

详见 [RUNBOOK.md](RUNBOOK.md) "多机型基准与选型"一节，两机的正面对比分析见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)（结论：即使模型完全放得进显存，生成速度差距仍有约35%，根源是GPU算力/带宽差异而非显存不足）。

See the "Multi-machine Benchmarks and Model Selection" section in [RUNBOOK.md](RUNBOOK.md) for details, and [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md) for a direct comparison between the two machines (conclusion: even for models that fit entirely in VRAM, the speed gap is still ~35% — the root cause is the GPU compute/bandwidth difference, not insufficient VRAM).

## 详细说明 / Further Details

请阅读 [RUNBOOK.md](RUNBOOK.md) 获取完整的操作步骤、参数说明和故障排除指南。

Please read [RUNBOOK.md](RUNBOOK.md) for the complete operating steps, parameter reference, and troubleshooting guide.
