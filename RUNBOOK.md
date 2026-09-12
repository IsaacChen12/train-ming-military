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

### 5.2 Open-WebUI 配置（qwen3.6:35b MoE + 知识库）

浏览器打开 `http://localhost:3000`

#### 5.2.1 配置 RAG 全局参数

**Admin Panel → Settings → Documents**

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding Model | `BAAI/bge-m3` | 中文检索最佳，首次自动下载 |
| Chunk Size | `1024` | 配合大上下文模型 |
| Chunk Overlap | `128` | |
| Top K | `5` | 先用小值确认稳定，再逐步增加到 20 |

#### 5.2.2 创建知识库

**Workspace → Knowledge → `+` New Knowledge**

```
Name:        明代军事装备知识库
Description: 明朝军事装备、火器、兵制专著文献
```

上传 `data/cleaned/` 中的 TXT 文件，等待向量化完成。

#### 5.2.3 创建专属模型

**Workspace → Models → `+` New Model**

**基本信息：**
```
Model ID:   ming-military-35b
Name:       明代军事助手 (qwen3.6:35b)
Base Model: qwen3.6:35b
```

**Knowledge 选项卡：** 勾选 `明代军事装备知识库`

**System Prompt：**
```
你是一位专精明代军事史的学术助手，擅长明朝军事装备、火器制造、兵制、战术等领域。

回答时请：
1. 优先基于检索到的史料原文，保持学术准确性
2. 引用原文时注明文献来源
3. 对不确定的信息如实说明，不捏造史实
4. 使用规范的历史术语和学术语言

/no_think
```

> ⚠️ **`/no_think` 必须保留**（见调整记录 2026-09-13）

**Advanced Parameters：**
```
num_ctx:         16384
temperature:     0.7
top_p:           0.9
repeat_penalty:  1.1
```

> ⚠️ **`num_ctx = 16384` 必须设置**（见调整记录 2026-09-13）

点击 **Save** 保存，新建对话选择该模型即可使用。

#### 5.2.4 验证连接

```cmd
docker exec ming-webui curl -s http://host.docker.internal:11434/api/tags
```

有 JSON 输出说明 Open-WebUI → Ollama 连接正常。

---

### 5.3 Open-WebUI 配置（qwen3.5:9b 密集模型 — 推荐日常使用）

#### 模型信息

| 参数 | 值 |
|------|-----|
| 架构 | `qwen35`（密集模型，非 MoE） |
| 参数量 | 9.7B |
| 文件大小 | 6.6 GB (Q4_K_M) |
| 最大上下文 | 262,144 tokens |
| VRAM 占用 | ~7 GB，剩余 ~9 GB 可用于 KV Cache |
| 实测速度 | **82.6 tok/s**（benchmark 2026-09-12） |

**与 qwen3.6:35b 的取舍：**

| 项目 | qwen3.5:9b | qwen3.6:35b MoE |
|------|-----------|-----------------|
| 生成速度 | **82 tok/s** | 53 tok/s |
| VRAM 占用 | **6.6 GB** | 23 GB |
| 可用 KV Cache | **~9 GB** | ~1 GB |
| 回答深度 | 一般 | **更详尽** |
| 可同时运行其他模型 | ✅ | ❌ |

> 9B 模型释放了大量显存给 KV Cache，实际可用上下文反而更长、更稳定。

---

#### 5.3.1 RAG 全局参数（与 5.2.1 相同，可直接复用）

**Admin Panel → Settings → Documents** 设置不变：

| 参数 | 值 |
|------|-----|
| Embedding Model | `BAAI/bge-m3` |
| Chunk Size | `1024` |
| Chunk Overlap | `128` |
| Top K | `15` ← 比 35b 可以设更高，VRAM 更宽裕 |

---

#### 5.3.2 创建专属模型

**Workspace → Models → `+` New Model**

**基本信息：**
```
Model ID:   ming-military-9b
Name:       明代军事助手 (qwen3.5:9b)
Base Model: qwen3.5:9b
```

**Knowledge 选项卡：** 勾选 `明代军事装备知识库`

**System Prompt：**
```
你是一位专精明代军事史的学术助手，擅长明朝军事装备、火器制造、兵制、战术等领域。

回答时请：
1. 优先基于检索到的史料原文，保持学术准确性
2. 引用原文时注明文献来源
3. 对不确定的信息如实说明，不捏造史实
4. 使用规范的历史术语和学术语言
```

**Advanced Parameters：**
```
temperature:     0.7
top_p:           0.9
repeat_penalty:  1.1
```

> ✅ qwen3.5:9b 使用默认 num_ctx 即可正常运行，无需像 qwen3.6:35b 那样强制设为 16384。如希望支持更长对话上下文可选填 `num_ctx = 8192`。

---

#### 5.3.3 与 qwen3.6:35b 的配置差异说明

qwen3.5:9b 在 Open-WebUI 挂载知识库后**开箱即用**，不需要 qwen3.6:35b 所需的两项强制修复：

| 配置项 | qwen3.5:9b | qwen3.6:35b MoE |
|--------|-----------|-----------------|
| `/no_think` | 非必须 | **必须** |
| `num_ctx = 16384` | 非必须 | **必须** |
| 知识库挂载后正常输出 | ✅ 开箱即用 | ❌ 需修复后才可用 |

根本原因：qwen3.6:35b 是 MoE 架构，thinking 模式 + RAG 上下文会撑满默认 context 窗口导致无输出；qwen3.5:9b 是密集模型，上下文占用更小，不存在此问题。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| OOM (显存不足) | 减小 `per_device_train_batch_size` 到 1，增大 `gradient_accumulation_steps` |
| 训练loss不下降 | 检查数据格式，降低 `learning_rate` 到 1e-5 |
| OCR中文识别差 | 改用 PaddleOCR 替代 Tesseract |
| ChromaDB慢 | 改用 FAISS 本地索引，或部署 Milvus |
| 中文分词差 | chunk时按句号/换行分割，而非固定字符数 |
| RAG有来源但无输出 | num_ctx 不足，改为 16384；System Prompt 末尾加 `/no_think` |
| Open-WebUI 思考后无答案 | qwen3.6:35b thinking 模式占满上下文，同上两步修复 |
| 模型第一次响应极慢 | 正常，冷启动需 30~40 秒加载；在 System Prompt 中加预热请求 |

---

## 参考资源

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 HuggingFace](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [ChromaDB Docs](https://docs.trychroma.com)
- [RAGAS 评估框架](https://docs.ragas.io)
- [BAAI/bge-m3 嵌入](https://huggingface.co/BAAI/bge-m3)

---

## 调整记录

### 2026-09-13 — Open-WebUI + qwen3.6:35b RAG 无输出问题修复

**现象：** 在 Open-WebUI 中使用 qwen3.6:35b 挂载知识库后，每次请求均显示"Thought for 10 seconds"并找到 4 个来源，但之后没有任何文字输出，仅显示 Follow up 建议。

**根本原因：** 两个因素叠加导致 context 溢出：

```
System Prompt   ≈  400 tokens
4个RAG chunks   ≈ 2000 tokens
Thinking过程    ≈  800 tokens
用户问题         ≈  100 tokens
─────────────────────────────
合计             ≈ 3300 tokens  >  默认 num_ctx 2048
```

模型完成 Thinking 后已无剩余 context 空间输出答案。

**修复措施：**

1. **关闭 Thinking 模式**：在模型 System Prompt 末尾添加 `/no_think`
2. **增大上下文窗口**：Advanced Parameters 中设置 `num_ctx = 16384`

**验证结果：** 两项同时生效后模型正常输出，速度恢复至约 53 tok/s。

**影响范围：** 所有在 Open-WebUI 中使用 Qwen3 系列 MoE 模型（qwen3.6:27b、qwen3.8:latest 等）并挂载知识库的配置，均需应用相同修复。
