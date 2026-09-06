# 明朝军事装备专域大模型
授权：陈方郡

基于 **Qwen2.5:14B** + **RAG + LoRA微调** 的明代军事史专业问答系统。

## 快速开始

```bash
# 1. 环境初始化（WSL Ubuntu）
bash 00_setup/wsl_setup.sh

# 2. 将书籍PDF放入 data/raw/，然后运行数据管线
bash 01_data_pipeline/run_pipeline.sh

# 3. 构建RAG索引并启动服务
bash 02_rag/build_and_serve.sh

# 4. （可选）微调训练
bash 03_finetune/train.sh

# 5. 一键部署
bash 05_deployment/deploy.sh
```

## 目录结构

```
train-ming-military/
├── RUNBOOK.md                    ← 完整操作手册（主要文档）
├── README.md                     ← 本文件
├── 00_setup/
│   ├── wsl_setup.sh              ← WSL环境一键初始化
│   ├── requirements.txt          ← Python依赖
│   └── docker-compose.yml        ← ChromaDB + Open-WebUI
├── 01_data_pipeline/
│   ├── 01_extract_pdf.py         ← PDF文字提取
│   ├── 02_ocr_images.py          ← OCR扫描识别（PaddleOCR）
│   ├── 03_clean_text.py          ← 文本清洗
│   ├── 04_chunk_text.py          ← 语义分块
│   ├── 05_generate_qa.py         ← LLM自动生成Q&A对
│   └── run_pipeline.sh           ← 一键运行数据管线
├── 02_rag/
│   ├── build_index.py            ← 构建ChromaDB向量索引
│   ├── rag_server.py             ← FastAPI RAG服务
│   ├── test_rag.py               ← 交互式测试
│   └── build_and_serve.sh        ← 一键构建和启动
├── 03_finetune/
│   ├── prepare_sft.py            ← 准备SFT训练数据
│   ├── qwen25_lora.yaml          ← LLaMA-Factory训练配置
│   ├── train.sh                  ← 训练脚本
│   └── merge_lora.sh             ← 合并LoRA权重
├── 04_evaluation/
│   ├── create_testset.py         ← 生成测试集模板
│   ├── eval_model.py             ← 评估模型（ROUGE + LLM-as-Judge）
│   ├── eval_rag.py               ← 评估RAG系统
│   └── compare.py                ← 三路对比报告
└── 05_deployment/
    ├── Modelfile                 ← Ollama模型配置
    └── deploy.sh                 ← 一键部署
```

## 核心技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 基座模型 | Qwen2.5-14B-Instruct | 中文能力强，开源可本地运行 |
| 嵌入模型 | BAAI/bge-m3 | 中文检索SOTA，支持长文本 |
| 向量数据库 | ChromaDB | 轻量易用，支持持久化 |
| 微调框架 | LLaMA-Factory | 支持Qwen2.5，QLoRA省显存 |
| 推理服务 | Ollama | 本地部署，OpenAI兼容API |
| OCR引擎 | PaddleOCR | 中文识别率高 |

## 详细说明

请阅读 [RUNBOOK.md](RUNBOOK.md) 获取完整的操作步骤、参数说明和故障排除指南。
