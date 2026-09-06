# 明朝军事装备大模型训练 — 完整运行手册

> **目标**：基于 Qwen2.5:14B，通过 RAG + LoRA 微调，构建一个能精准回答明朝军事装备问题的专域大模型。
> **环境**：Windows 11 + WSL2 Ubuntu 22.04 + CUDA（主力训练环境）
> **日期**：2026-09

---

## 架构总览

```
原始书籍 (PDF/扫描件)
    │
    ▼
[阶段1] 数据管线
    ├── PDF文本提取
    ├── OCR扫描识别
    ├── 文本清洗分块
    └── LLM自动生成Q&A对
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[阶段2] RAG系统               [阶段3] SFT微调
  ChromaDB向量库                LLaMA-Factory
  bge-m3嵌入模型                QLoRA (4-bit)
  FastAPI服务                   Qwen2.5:14B
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        [阶段4] 评估对比
          RAG vs 微调 vs RAG+微调
               │
               ▼
        [阶段5] 部署上线
          Ollama + Open-WebUI
```

### 三种方案对比

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| 纯RAG | 无需训练，快速上线，易更新 | 依赖检索质量，推理慢 | 快速验证 |
| 纯LoRA微调 | 推理快，回答风格好 | 无法引用原文，可能幻觉 | 通用增强 |
| **RAG + LoRA** | **最准确，可溯源** | **工程复杂** | **生产推荐** |

**建议路径**：先做RAG（1-2天），验证效果后做LoRA微调（3-5天），最终合并部署。

---

## 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3090 (24GB VRAM) | RTX 4090 (24GB) / A100 |
| RAM | 32GB | 64GB |
| 磁盘 | 100GB SSD | 200GB NVMe |
| CUDA | 12.1+ | 12.4 |

> Qwen2.5:14B QLoRA训练显存估算：14B × 4bit ≈ 7GB模型 + 梯度/优化器 ≈ **总计18-22GB**

---

## 阶段0：环境搭建

### 0.1 启用 WSL2 + Ubuntu

```powershell
# Windows PowerShell (管理员)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

### 0.2 WSL Ubuntu 环境初始化

```bash
# 进入WSL
wsl

# 更新系统
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 poppler-utils tesseract-ocr tesseract-ocr-chi-sim

# 创建项目Python虚拟环境
cd /mnt/d/ollama/train-ming-military
python3.11 -m venv .venv
source .venv/bin/activate

# 安装PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r 00_setup/requirements.txt
```

### 0.3 安装 LLaMA-Factory（微调框架）

```bash
cd /mnt/d/ollama/train-ming-military
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 03_finetune/LLaMA-Factory
cd 03_finetune/LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes,qwen]"
```

### 0.4 安装 Ollama（推理服务）

```bash
# WSL 内安装
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # 后台启动

# 拉取基座模型
ollama pull qwen2.5:14b
```

### 0.5 Docker 服务（ChromaDB + Open-WebUI）

```bash
# 在 Windows PowerShell 中
cd D:\ollama\train-ming-military
docker-compose -f 00_setup/docker-compose.yml up -d
```

---

## 阶段1：数据管线

### 1.1 目录结构

```
data/
├── raw/          # 原始书籍文件 (PDF/扫描图片)
├── txt/          # 提取后的纯文本
├── cleaned/      # 清洗后文本
├── chunks/       # 分块后的JSONL
├── qa_pairs/     # 生成的Q&A对
└── sft/          # 最终SFT训练集
```

```bash
mkdir -p data/{raw,txt,cleaned,chunks,qa_pairs,sft}
```

### 1.2 将书籍放入 data/raw/

支持格式：`.pdf`、`.txt`、`.epub`、扫描图片(`.jpg`/`.png`)

### 1.3 运行数据管线

```bash
source .venv/bin/activate

# Step 1: PDF文本提取
python 01_data_pipeline/01_extract_pdf.py --input data/raw --output data/txt

# Step 2: OCR处理扫描书籍（如有）
python 01_data_pipeline/02_ocr_images.py --input data/raw --output data/txt

# Step 3: 文本清洗
python 01_data_pipeline/03_clean_text.py --input data/txt --output data/cleaned

# Step 4: 分块
python 01_data_pipeline/04_chunk_text.py \
    --input data/cleaned \
    --output data/chunks \
    --chunk-size 512 \
    --overlap 64

# Step 5: 自动生成Q&A对（需要Ollama在运行）
python 01_data_pipeline/05_generate_qa.py \
    --input data/chunks \
    --output data/qa_pairs \
    --model qwen2.5:14b \
    --questions-per-chunk 3
```

---

## 阶段2：RAG系统

### 2.1 构建向量索引

```bash
# 构建ChromaDB索引（使用bge-m3中文嵌入）
python 02_rag/build_index.py \
    --chunks data/chunks \
    --db-path data/chroma_db \
    --embedding-model BAAI/bge-m3
```

### 2.2 启动RAG服务

```bash
python 02_rag/rag_server.py \
    --db-path data/chroma_db \
    --llm-base-url http://localhost:11434/v1 \
    --model qwen2.5:14b \
    --port 8080
```

### 2.3 测试RAG效果

```bash
python 02_rag/test_rag.py \
    --server http://localhost:8080 \
    --questions "明代火绳枪与欧洲同期火枪有何差异？"
```

---

## 阶段3：SFT微调

### 3.1 准备SFT数据集

```bash
python 03_finetune/prepare_sft.py \
    --qa-dir data/qa_pairs \
    --output-dir data/sft \
    --train-ratio 0.9
```

### 3.2 下载基座模型权重（HuggingFace格式）

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载Qwen2.5-14B-Instruct
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /mnt/d/models/Qwen2.5-14B-Instruct \
    --resume-download
```

> 或使用已有的vLLM服务（192.168.1.209）做数据生成，训练仍需本地HF权重。

### 3.3 启动LoRA训练

```bash
cd 03_finetune/LLaMA-Factory

llamafactory-cli train ../qwen25_lora.yaml
```

训练过程监控（另开终端）：

```bash
# 查看GPU使用
watch -n 2 nvidia-smi

# 查看训练日志
tail -f /mnt/d/ollama/train-ming-military/outputs/training.log
```

### 3.4 合并LoRA权重

```bash
llamafactory-cli export \
    --model_name_or_path /mnt/d/models/Qwen2.5-14B-Instruct \
    --adapter_name_or_path outputs/lora_weights \
    --export_dir /mnt/d/models/ming-military-14b \
    --export_size 4 \
    --export_legacy_format false
```

---

## 阶段4：评估

### 4.1 创建测试集（人工标注20-50题）

```bash
python 04_evaluation/create_testset.py --output data/testset.jsonl
```

手动编辑 `data/testset.jsonl`，填写标准答案。

### 4.2 三路对比评估

```bash
# 评估纯基座模型
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model qwen2.5:14b \
    --testset data/testset.jsonl \
    --output results/baseline.json

# 评估RAG系统
python 04_evaluation/eval_rag.py \
    --rag-server http://localhost:8080 \
    --testset data/testset.jsonl \
    --output results/rag.json

# 评估微调后模型（需先部署）
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model ming-military \
    --testset data/testset.jsonl \
    --output results/finetuned.json

# 汇总对比报告
python 04_evaluation/compare.py \
    --results results/baseline.json results/rag.json results/finetuned.json
```

---

## 阶段5：部署

### 5.1 注册到Ollama

```bash
# 创建Modelfile
cp 05_deployment/Modelfile /mnt/d/models/ming-military-14b/Modelfile

cd /mnt/d/models/ming-military-14b
ollama create ming-military -f Modelfile

# 验证
ollama run ming-military "明朝虎蹲炮的射程和威力如何？"
```

### 5.2 Open-WebUI访问

浏览器打开 `http://localhost:3000`，选择模型 `ming-military`。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| OOM (显存不足) | 减小 `per_device_train_batch_size` 到 1，增大 `gradient_accumulation_steps` |
| 训练loss不下降 | 检查数据格式，降低 `learning_rate` 到 1e-5 |
| OCR中文识别差 | 改用 PaddleOCR 替代 Tesseract |
| ChromaDB慢 | 改用 FAISS 本地索引，或部署 Milvus |
| 中文分词差 | chunk时按句号/换行分割，而非固定字符数 |

---

## 参考资源

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 HuggingFace](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [ChromaDB Docs](https://docs.trychroma.com)
- [RAGAS 评估框架](https://docs.ragas.io)
- [BAAI/bge-m3 嵌入](https://huggingface.co/BAAI/bge-m3)
