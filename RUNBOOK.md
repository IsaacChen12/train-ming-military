# 明朝军事装备大模型训练 — 完整运行手册
# Ming Dynasty Military Equipment LLM Training — Full Operations Manual

> **目标**：以 **Qwen3.5:9B** 为微调基座，通过 RAG + LoRA 微调，构建一个能精准回答明朝军事装备问题的专域大模型；经机器A（4090 Laptop 16GB，[报告](benchmark/reports/rtx4090-laptop-16gb.md)）与机器B（RTX 3060 12GB，[报告](benchmark/reports/rtx3060-12gb.md)）两台机器实测，9B 模型在两台机器上均可部署，但**具体候选模型排名因机器而异**（12GB显存下 `qwen2.5:7b` 反而最快），另有 **qwen3.6:35b MoE** 作为质量优先的RAG部署候选（不参与微调，两机均可运行但速度不同）；详见"多机型基准与选型"一节。
> **Goal**: Use **Qwen3.5:9B** as the fine-tuning base and combine RAG + LoRA fine-tuning to build a domain-specific LLM that can accurately answer questions about Ming dynasty military equipment. Tested on Machine A (4090 Laptop 16GB, [report](benchmark/reports/rtx4090-laptop-16gb.md)) and Machine B (RTX 3060 12GB, [report](benchmark/reports/rtx3060-12gb.md)), the 9B model deploys on both machines, but **the exact candidate-model ranking differs by machine** (with 12GB VRAM, `qwen2.5:7b` is actually the fastest). **qwen3.6:35b MoE** is also available as a quality-first RAG deployment candidate (not used for fine-tuning, runs on both machines at different speeds); see "Multi-machine Benchmarks and Model Selection" for details.
>
> **环境**：Windows 11 + WSL2 Ubuntu 22.04 + CUDA（主力训练环境）；推理基准测试覆盖 RTX 4090 Laptop（16GB VRAM）与 RTX 3060（12GB VRAM）两台机器
> **Environment**: Windows 11 + WSL2 Ubuntu 22.04 + CUDA (primary training environment); inference benchmarks cover two machines — RTX 4090 Laptop (16GB VRAM) and RTX 3060 (12GB VRAM)
>
> **日期**：2026-09
> **Date**: 2026-09

---

## 架构总览 / Architecture Overview

```
原始书籍 (PDF/扫描件) / Source books (PDF/scans)
    │
    ▼
[阶段1] 数据管线 / [Stage 1] Data pipeline
    ├── PDF文本提取 / PDF text extraction
    ├── OCR扫描识别 / OCR for scans
    ├── 文本清洗分块 / Text cleaning & chunking
    └── LLM自动生成Q&A对 / LLM auto-generated Q&A pairs
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[阶段2] RAG系统 / [Stage 2] RAG system      [阶段3] SFT微调 / [Stage 3] SFT fine-tuning
  ChromaDB向量库 / ChromaDB vector store       LLaMA-Factory
  bge-m3嵌入模型 / bge-m3 embedding model      QLoRA (4-bit)
  FastAPI服务 / FastAPI service                Qwen3.5:9B（微调基座 / fine-tuning base）
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        [阶段4] 评估对比 / [Stage 4] Evaluation & comparison
          RAG vs 微调 vs RAG+微调 / RAG vs. fine-tuned vs. RAG+fine-tuned
               │
               ▼
        [阶段5] 部署上线 / [Stage 5] Deployment
          Ollama + Open-WebUI
          候选模型（4090实测）/ Candidates (measured on 4090)：
          qwen3.5:9b / qwen3.6:35b MoE
```

### 三种方案对比 / Comparing the Three Approaches

| 方案 / Approach | 优点 / Pros | 缺点 / Cons | 推荐场景 / Recommended for |
|------|------|------|----------|
| 纯RAG / RAG only | 无需训练，快速上线，易更新 / No training needed, fast to ship, easy to update | 依赖检索质量，推理慢 / Depends on retrieval quality, slower inference | 快速验证 / Quick validation |
| 纯LoRA微调 / LoRA fine-tuning only | 推理快，回答风格好 / Fast inference, good answer style | 无法引用原文，可能幻觉 / Can't cite source text, may hallucinate | 通用增强 / General-purpose enhancement |
| **RAG + LoRA** | **最准确，可溯源 / Most accurate, traceable** | **工程复杂 / More engineering complexity** | **生产推荐 / Recommended for production** |

**建议路径**：先做RAG（1-2天），验证效果后做LoRA微调（3-5天），最终合并部署。
**Suggested path**: Build RAG first (1-2 days), validate results, then do LoRA fine-tuning (3-5 days), and finally merge and deploy.

---

## 硬件要求 / Hardware Requirements

| 组件 / Component | 最低配置 / Minimum | 推荐配置 / Recommended |
|------|----------|----------|
| GPU | RTX 3060 (12GB VRAM) | RTX 4090 (16GB+) |
| RAM | 32GB | 64GB |
| 磁盘 / Disk | 100GB SSD | 200GB NVMe |
| CUDA | 12.1+ | 12.4 |

> 微调基座已由 Qwen2.5:14B 改为 **Qwen3.5:9B**（原14B方案QLoRA显存需求18-22GB，需24GB+显卡）。9B QLoRA训练显存估算：9B × 4bit ≈ 4.5GB模型 + 梯度/优化器 ≈ **总计10-13GB**，16GB显卡（如 RTX 4090 Laptop）即可完成训练，无需24GB+显卡；12GB显卡（如 RTX 3060）**推理已验证可行**（见下方基准数据），但**训练尚未实测**，估算区间上限（13GB）已超出可用显存，需调小 `cutoff_len` 或开启梯度检查点，见下方"多机型基准与选型"章节。
>
> The fine-tuning base was changed from Qwen2.5:14B to **Qwen3.5:9B** (the original 14B plan needed 18-22GB of QLoRA VRAM, requiring a 24GB+ GPU). Estimated 9B QLoRA training VRAM: 9B × 4-bit ≈ 4.5GB for the model + gradients/optimizer ≈ **10-13GB total**. A 16GB GPU (e.g. RTX 4090 Laptop) is enough to train, no 24GB+ GPU required. A 12GB GPU (e.g. RTX 3060) **has been verified for inference** (see benchmark data below), but **training has not yet been tested empirically** — the upper end of the estimate (13GB) exceeds available VRAM, so reduce `cutoff_len` or enable gradient checkpointing; see the "Multi-machine Benchmarks and Model Selection" section below.

### 推理候选模型显存占用（4090 Laptop 16GB 实测）/ Inference Candidate VRAM Usage (Measured on 4090 Laptop 16GB)

微调完成后的**线上推理/部署**同样不要求24GB显卡：在16GB显存的 RTX 4090 Laptop 上实测（详见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)），以下两个候选模型均可正常运行：

Post-fine-tuning **online inference/deployment** likewise does not require a 24GB GPU. Measured on a 16GB RTX 4090 Laptop (see [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md) for details), both of the following candidates run normally:

| 候选模型 / Candidate | 参数量 / Params | VRAM占用 / VRAM | 生成速度 / Speed | 说明 / Notes |
|----------|--------|----------|----------|------|
| `qwen3.5:9b` | 9B（密集，微调基座 / dense, fine-tuning base） | ~6.6 GB | 82.6 tok/s | 剩余显存多，可同时跑其他模型/服务 / Plenty of VRAM left over, can run other models/services concurrently |
| `qwen3.6:35b`（MoE，不参与微调 / not used for fine-tuning） | 35B | ~23 GB（部分CPU卸载 / partial CPU offload） | 53.6 tok/s | 16GB显卡需offload，回答更详尽 / Needs offload on a 16GB GPU, gives more detailed answers |

因此 16GB 级别显卡足以承担本项目的**训练+推理全流程**（基座已改为9B）。

So a 16GB-class GPU is enough to cover the project's **entire training + inference pipeline** (now that the base model is 9B).

## 多机型基准与选型 / Multi-machine Benchmarks and Model Selection

本项目已在两台硬件不同的机器上做过基准测试，**模型选型需按实际机器的显存/算力单独确认，不能直接照搬另一台机器的推荐**。两台机器的直接对比分析见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)。

This project has been benchmarked on two machines with different hardware. **Model selection must be confirmed separately for each machine's actual VRAM/compute — a recommendation from one machine cannot simply be copied to the other.** See [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md) for a direct comparison of the two machines.

### 机器 A — RTX 4090 Laptop（16GB VRAM）/ Machine A — RTX 4090 Laptop (16GB VRAM)

- 完整基准报告 / Full benchmark report：[benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)
- 结论 / Conclusion：微调基座 + 日常RAG 用 `qwen3.5:9b`；质量优先RAG（不微调）用 `qwen3.6:35b` MoE / Use `qwen3.5:9b` for the fine-tuning base + everyday RAG; use `qwen3.6:35b` MoE for quality-first RAG (no fine-tuning)
- 详见本文档前面各节 / See the earlier sections of this document for details

### 机器 B — RTX 3060（12GB VRAM）/ Machine B — RTX 3060 (12GB VRAM)

- 硬件 / Hardware：CPU Intel i7-12700F（12核20线程 / 12 cores, 20 threads）/ RAM 64GB / GPU RTX 3060 12GB VRAM
- 完整基准报告 / Full benchmark report：[benchmark/reports/rtx3060-12gb.md](benchmark/reports/rtx3060-12gb.md)

| 候选模型 / Candidate | 生成速度 / Speed | VRAM占用 / VRAM | 说明 / Notes |
|----------|----------|----------|------|
| `qwen2.5:7b` | **65.8 tok/s**（本机最快 / fastest on this machine） | ~4.7 GB | 机器A未测试过，速度优先场景可用 / Not tested on Machine A, usable for speed-first scenarios |
| `qwen3.5:9b` | 53.9 tok/s | ~6.6 GB | 与机器A保持一致的微调基座/RAG默认 / Same fine-tuning base/RAG default as Machine A |
| `qwen3.6:35b`（MoE） | 38.8 tok/s（较机器A的53.6 tok/s下降约28% / down ~28% from 53.6 tok/s on Machine A） | ~22 GB（`ollama ps`显示58%CPU/42%GPU offload / `ollama ps` shows 58% CPU / 42% GPU offload） | 仍可用但明显更慢，只建议离线/非交互场景 / Still usable but noticeably slower; recommended only for offline/non-interactive use |
| `gemma4:12b` | 37.8 tok/s（较机器A的58.8 tok/s下降约36% / down ~36% from 58.8 tok/s on Machine A） | ~7.6 GB（完全放得进显存，无offload / fits fully in VRAM, no offload） | 后补测试；与`qwen3.6:35b`速度几乎打平但体积小1/3，质量优先场景的备选 / Added later; nearly matches `qwen3.6:35b` in speed but 1/3 the size — an alternative for quality-first scenarios |
| `qwen2.5:14b` | 34.7 tok/s | ~9 GB | 机器A未测试过；本机上是5个模型中最慢的（14B密集模型 vs 35B MoE，MoE反而更快） / Not tested on Machine A; the slowest of the 5 models here (14B dense vs. 35B MoE — the MoE model is actually faster) |

**结论 / Conclusion：** 排名与机器A**不同**——12GB显存下 `qwen2.5:7b` 反而最快，且模型越大掉速越明显（35B MoE从53.6降到38.8 tok/s，降幅约28%）。RAG部署仍建议 `qwen3.5:9b`（与机器A一致，保持训练管线统一），速度优先可换 `qwen2.5:7b`，质量优先仍用 `qwen3.6:35b`（`gemma4:12b`速度相近且更省显存，可作平替）。

The ranking **differs** from Machine A — with 12GB VRAM, `qwen2.5:7b` is actually the fastest, and larger models slow down more noticeably (the 35B MoE drops from 53.6 to 38.8 tok/s, about -28%). `qwen3.5:9b` is still recommended for RAG deployment (consistent with Machine A, keeps the training pipeline unified); swap in `qwen2.5:7b` for speed-first use, and still use `qwen3.6:35b` for quality-first use (`gemma4:12b` has similar speed and uses less VRAM, so it can substitute).

> ⚠️ **微调注意事项 / Fine-tuning caveat：** 本机是否能顺利完成 `qwen3.5:9b` 的 QLoRA 训练**尚未实测**（本次基准只测了推理）。9B QLoRA 训练显存估算10-13GB，而本机仅有12GB，估算区间上限已超出可用显存，实际训练前建议先减小 `qwen35_lora.yaml` 中的 `cutoff_len`（如降到1024）或开启 `gradient_checkpointing: true`，并密切观察 `nvidia-smi` 显存占用，防止OOM。
>
> Whether this machine can successfully complete `qwen3.5:9b` QLoRA training **has not yet been tested** (this benchmark only covered inference). Estimated 9B QLoRA training VRAM is 10-13GB, while this machine only has 12GB — the upper end of the estimate exceeds available VRAM. Before actual training, consider reducing `cutoff_len` in `qwen35_lora.yaml` (e.g. down to 1024) or enabling `gradient_checkpointing: true`, and watch `nvidia-smi` VRAM usage closely to avoid OOM.

> 📊 **两机对比 / Cross-machine comparison：** `qwen3.5:9b`/`gemma4:12b`/`qwen3.6:35b` 三个模型在两台机器上都测过，生成速度分别下降34.8%/35.7%/27.6%。前两个完全放得进两张卡的显存，降幅却和显存不足的`qwen3.6:35b`一样大甚至更大——说明降速**并非**由显存不足导致，根源是两张GPU本身的算力/带宽差异；细节与更多发现见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)。
>
> `qwen3.5:9b`, `gemma4:12b`, and `qwen3.6:35b` were all tested on both machines, with speed dropping 34.8% / 35.7% / 27.6% respectively. The first two fit entirely in VRAM on both cards, yet their slowdown is as large as or larger than `qwen3.6:35b`'s (which is VRAM-constrained) — showing the slowdown is **not** caused by insufficient VRAM but by the raw compute/bandwidth difference between the two GPUs. See [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md) for details and further findings.

---

## 阶段0：环境搭建 / Stage 0: Environment Setup

### 0.1 启用 WSL2 + Ubuntu / Enable WSL2 + Ubuntu

```powershell
# Windows PowerShell (管理员 / Administrator)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

### 0.2 WSL Ubuntu 环境初始化 / WSL Ubuntu Environment Initialization

```bash
# 进入WSL / Enter WSL
wsl

# 更新系统 / Update the system
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 poppler-utils tesseract-ocr tesseract-ocr-chi-sim

# 创建项目Python虚拟环境 / Create the project's Python virtual environment
cd /mnt/d/ollama/train-ming-military
python3.11 -m venv .venv
source .venv/bin/activate

# 安装PyTorch (CUDA 12.1) / Install PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖 / Install project dependencies
pip install -r 00_setup/requirements.txt
```

### 0.3 安装 LLaMA-Factory（微调框架）/ Install LLaMA-Factory (Fine-tuning Framework)

```bash
cd /mnt/d/ollama/train-ming-military
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 03_finetune/LLaMA-Factory
cd 03_finetune/LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes,qwen]"
```

### 0.4 安装 Ollama（推理服务）/ Install Ollama (Inference Service)

```bash
# WSL 内安装 / Install inside WSL
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # 后台启动 / Start in the background

# 拉取基座模型（用于Q&A自动生成/RAG基线测试）/ Pull the base model (for Q&A auto-generation / RAG baseline testing)
ollama pull qwen2.5:14b

# 拉取部署候选模型（机器A 4090实测，见 benchmark/reports/rtx4090-laptop-16gb.md）/ Pull deployment candidates (measured on Machine A's 4090, see benchmark/reports/rtx4090-laptop-16gb.md)
ollama pull qwen3.5:9b
ollama pull qwen3.6:35b
```

### 0.5 Docker 服务（ChromaDB + Open-WebUI）/ Docker Services (ChromaDB + Open-WebUI)

```bash
# 在 Windows PowerShell 中 / In Windows PowerShell
cd D:\ollama\train-ming-military
docker-compose -f 00_setup/docker-compose.yml up -d
```

---

## 阶段1：数据管线 / Stage 1: Data Pipeline

### 1.1 目录结构 / Directory Structure

```
data/
├── raw/          # 原始书籍文件 (PDF/扫描图片) / Source book files (PDF/scanned images)
├── txt/          # 提取后的纯文本 / Extracted plain text
├── cleaned/      # 清洗后文本 / Cleaned text
├── chunks/       # 分块后的JSONL / Chunked JSONL
├── qa_pairs/     # 生成的Q&A对 / Generated Q&A pairs
└── sft/          # 最终SFT训练集 / Final SFT training set
```

```bash
mkdir -p data/{raw,txt,cleaned,chunks,qa_pairs,sft}
```

### 1.2 将书籍放入 data/raw/ / Put Books into data/raw/

支持格式：`.pdf`、`.txt`、`.epub`、扫描图片(`.jpg`/`.png`)

Supported formats: `.pdf`, `.txt`, `.epub`, scanned images (`.jpg`/`.png`)

### 1.3 运行数据管线 / Run the Data Pipeline

```bash
source .venv/bin/activate

# Step 1: PDF文本提取 / Step 1: PDF text extraction
python 01_data_pipeline/01_extract_pdf.py --input data/raw --output data/txt

# Step 2: OCR处理扫描书籍（如有）/ Step 2: OCR scanned books (if any)
python 01_data_pipeline/02_ocr_images.py --input data/raw --output data/txt

# Step 3: 文本清洗 / Step 3: Text cleaning
python 01_data_pipeline/03_clean_text.py --input data/txt --output data/cleaned

# Step 4: 分块 / Step 4: Chunking
python 01_data_pipeline/04_chunk_text.py \
    --input data/cleaned \
    --output data/chunks \
    --chunk-size 512 \
    --overlap 64

# Step 5: 自动生成Q&A对（需要Ollama在运行）/ Step 5: Auto-generate Q&A pairs (requires Ollama running)
python 01_data_pipeline/05_generate_qa.py \
    --input data/chunks \
    --output data/qa_pairs \
    --model qwen2.5:14b \
    --questions-per-chunk 3
```

### 1.4 当前分块策略说明 / Current Chunking Strategy

`04_chunk_text.py` 目前的实际切分逻辑（`chunk_size=512`，`overlap=64`）：

The actual splitting logic currently in `04_chunk_text.py` (`chunk_size=512`, `overlap=64`):

1. 先按空行（`\n{2,}`）把全文拆成段落 / First split the full text into paragraphs on blank lines (`\n{2,}`)
2. 把过短的段落（<50字符）合并进相邻段落，避免产生大量碎片小块 / Merge overly short paragraphs (<50 characters) into neighboring ones, to avoid a lot of fragmentary small chunks
3. 依次把段落塞进缓冲区，一旦累计超过 `chunk_size` 就切出一块，切块时保留末尾 `overlap` 个字符作为下一块开头，维持上下文连贯 / Feed paragraphs into a buffer in order; once the accumulated length exceeds `chunk_size`, cut a chunk, keeping the last `overlap` characters as the start of the next chunk to preserve context continuity
4. 单个段落本身就超过 `chunk_size` 的（如无空行的大段古籍原文），按字符数硬切 / A single paragraph that already exceeds `chunk_size` (e.g. a long block of classical-text source with no blank lines) is hard-cut by character count

> ⚠️ **已知局限 / Known limitation**：脚本里定义了按章节标题的正则 `CHAPTER_RE`（如"第X章/节"），但当前切分逻辑**并未实际调用它**——目前是纯粹的"段落+字数"切分，不具备章节/小节边界感知能力。docx来源（无PDF那样明确的分页/章节结构）尤其依赖这个正则来做更语义化的切分，目前实测（4篇军事史论文 → 316块）效果可用，但块边界有时会切在论述中间。
>
> The script defines a chapter-heading regex, `CHAPTER_RE` (matching things like "Chapter X / Section X"), but the current splitting logic **does not actually call it** — splitting today is purely "paragraph + character count," with no awareness of chapter/section boundaries. This matters especially for docx sources (which lack the clear pagination/chapter structure a PDF has) that would benefit from this regex for more semantic splitting. In practice (4 military-history papers → 316 chunks) the result is usable, but chunk boundaries sometimes fall in the middle of an argument.

**后续待办**：分块策略后续会做进一步研究和调优（例如启用/替换章节感知切分、按语义或标点边界切分、针对不同文体（论文正文 vs 古籍引文/表格）采用不同 `chunk_size`），当前参数仅作为跑通全流程的基线，不代表最终方案。

**Future work**: The chunking strategy will get further research and tuning later (e.g. enabling/replacing chapter-aware splitting, splitting on semantic or punctuation boundaries, using different `chunk_size` values for different content types such as paper prose vs. classical-text quotations/tables). The current parameters are only a baseline to get the full pipeline running end to end, not the final design.

### 1.5 当前语料统计 / Current Corpus Statistics

截至 2026-09-27，用 `--chunk-size 512 --overlap 64` 处理过的素材：

Material processed so far with `--chunk-size 512 --overlap 64`, as of 2026-09-27:

| 素材 / Source | 块数 / Chunks | 字符数 / Characters |
|---|---|---|
| 《营要事宜》中的明军阵列复原与同期瑞典军队的对战模拟 | 19 | 7,758 |
| 中欧火绳枪的比较——以现有实验为基础的威力推算 | 69 | 28,432 |
| 大陆两端的纯步兵详解——从编制到战术 | 178 | 73,865 |
| 梦到什么写什么——从车营到步火营 | 50 | 20,247 |
| **合计 / Total** | **316** | **130,302** |

> 字符数为 `04_chunk_text.py` 的 `char_count` 口径（`len(text)`，含标点/数字/英文/换行，非纯中文字数；纯汉字约98,571个，占比约76%）。来源均为 `.docx`（本项目为此在 `01_extract_pdf.py` 中新增了 `.docx` 支持）。
>
> The character count uses the `char_count` field from `04_chunk_text.py` (`len(text)` — includes punctuation, digits, English letters, and newlines, not just Chinese characters; pure Han characters are ~98,571, about 76% of the total). All sources so far are `.docx` files (`.docx` support was added to `01_extract_pdf.py` specifically for this).

> 📌 **维护规则 / Maintenance rule**：每次向 `data/raw/` 添加新素材、重新跑一遍提取→清洗→分块流程后，必须同步更新本节表格和 [README.md](README.md) 里的"当前语料统计"，保持文档与 `data/chunks/` 的实际内容一致；同时把本次改动记到项目根目录的 [WORKLOG.md](WORKLOG.md) 里。
>
> Whenever new material is added to `data/raw/` and the extract → clean → chunk pipeline is re-run, this table and the "Current Corpus Statistics" section in [README.md](README.md) must both be updated to match the actual contents of `data/chunks/`; the change should also be recorded in [WORKLOG.md](WORKLOG.md) at the project root.

---

## 阶段2：RAG系统 / Stage 2: RAG System

### 2.1 构建向量索引 / Build the Vector Index

```bash
# 构建ChromaDB索引（使用bge-m3中文嵌入）/ Build the ChromaDB index (using bge-m3 Chinese embeddings)
python 02_rag/build_index.py \
    --chunks data/chunks \
    --db-path data/chroma_db \
    --embedding-model BAAI/bge-m3
```

### 2.2 启动RAG服务 / Start the RAG Service

```bash
python 02_rag/rag_server.py \
    --db-path data/chroma_db \
    --llm-base-url http://localhost:11434/v1 \
    --model qwen2.5:14b \
    --port 8080
```

> 机器A（4090 16GB）环境下也可将 `--model` 换成实测候选模型 `qwen3.5:9b`（速度优先）或 `qwen3.6:35b`（质量优先，需搭配 `/no_think` 与更大 `num_ctx`，见阶段5.2/5.3），详见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)。
>
> On Machine A (4090 16GB), `--model` can also be swapped for the benchmarked candidates `qwen3.5:9b` (speed-first) or `qwen3.6:35b` (quality-first, needs `/no_think` and a larger `num_ctx` — see Stage 5.2/5.3); see [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md) for details.

### 2.3 测试RAG效果 / Test the RAG Output

```bash
python 02_rag/test_rag.py \
    --server http://localhost:8080 \
    --questions "明代火绳枪与欧洲同期火枪有何差异？"
```

---

## 阶段3：SFT微调 / Stage 3: SFT Fine-tuning

### 3.1 准备SFT数据集 / Prepare the SFT Dataset

```bash
python 03_finetune/prepare_sft.py \
    --qa-dir data/qa_pairs \
    --output-dir data/sft \
    --train-ratio 0.9
```

### 3.2 下载基座模型权重（HuggingFace格式）/ Download Base Model Weights (HuggingFace Format)

```bash
# 安装 huggingface-cli / Install huggingface-cli
pip install huggingface_hub

# 下载Qwen3.5-9B-Instruct / Download Qwen3.5-9B-Instruct
huggingface-cli download Qwen/Qwen3.5-9B-Instruct \
    --local-dir /mnt/d/models/Qwen3.5-9B-Instruct \
    --resume-download
```

> 或使用已有的vLLM服务（192.168.1.209）做数据生成，训练仍需本地HF权重。
>
> Alternatively, use the existing vLLM service (192.168.1.209) for data generation; training still requires local HF weights.

### 3.3 启动LoRA训练 / Start LoRA Training

```bash
cd 03_finetune/LLaMA-Factory

llamafactory-cli train ../qwen35_lora.yaml
```

训练过程监控（另开终端）：
Monitor the training run (in another terminal):

```bash
# 查看GPU使用 / Watch GPU usage
watch -n 2 nvidia-smi

# 查看训练日志 / Tail the training log
tail -f /mnt/d/ollama/train-ming-military/outputs/training.log
```

### 3.4 合并LoRA权重 / Merge LoRA Weights

```bash
llamafactory-cli export \
    --model_name_or_path /mnt/d/models/Qwen3.5-9B-Instruct \
    --adapter_name_or_path outputs/lora_weights \
    --template qwen3 \
    --export_dir /mnt/d/models/ming-military-9b \
    --export_size 4 \
    --export_legacy_format false
```

---

## 阶段4：评估 / Stage 4: Evaluation

### 4.1 创建测试集（人工标注20-50题）/ Create a Test Set (Manually Label 20-50 Questions)

```bash
python 04_evaluation/create_testset.py --output data/testset.jsonl
```

手动编辑 `data/testset.jsonl`，填写标准答案。

Manually edit `data/testset.jsonl` and fill in the reference answers.

### 4.2 三路对比评估 / Three-way Comparison Evaluation

```bash
# 评估纯基座模型 / Evaluate the plain base model
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model qwen3.5:9b \
    --testset data/testset.jsonl \
    --output results/baseline.json

# 评估RAG系统 / Evaluate the RAG system
python 04_evaluation/eval_rag.py \
    --rag-server http://localhost:8080 \
    --testset data/testset.jsonl \
    --output results/rag.json

# 评估微调后模型（需先部署）/ Evaluate the fine-tuned model (must be deployed first)
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model ming-military \
    --testset data/testset.jsonl \
    --output results/finetuned.json

# 汇总对比报告 / Generate the summary comparison report
python 04_evaluation/compare.py \
    --results results/baseline.json results/rag.json results/finetuned.json
```

---

## 阶段5：部署 / Stage 5: Deployment

> 5.1 是"合并LoRA权重"后的自定义模型部署路径；若不做微调，直接用 RAG + 基座模型上线，可跳到 5.2/5.3 — 两节分别对应机器A（4090）环境实测确定的两个候选基座：`qwen3.6:35b`（质量优先）与 `qwen3.5:9b`（速度优先，推荐日常使用），选型依据见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)。
>
> Section 5.1 is the deployment path for the custom model after "merging LoRA weights." If you're not fine-tuning and just shipping RAG + the base model, skip to 5.2/5.3 — these two sections cover the two candidate base models determined by testing on Machine A (4090): `qwen3.6:35b` (quality-first) and `qwen3.5:9b` (speed-first, recommended for everyday use). See [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md) for the rationale.

### 5.1 注册到Ollama / Register with Ollama

```bash
# 创建Modelfile / Create the Modelfile
cp 05_deployment/Modelfile /mnt/d/models/ming-military-9b/Modelfile

cd /mnt/d/models/ming-military-9b
ollama create ming-military -f Modelfile

# 验证 / Verify
ollama run ming-military "明朝虎蹲炮的射程和威力如何？"
```

### 5.2 Open-WebUI 配置（qwen3.6:35b MoE + 知识库）/ Open-WebUI Configuration (qwen3.6:35b MoE + Knowledge Base)

浏览器打开 `http://localhost:3000`

Open `http://localhost:3000` in a browser.

#### 5.2.1 配置 RAG 全局参数 / Configure Global RAG Parameters

**Admin Panel → Settings → Documents**

| 参数 / Parameter | 值 / Value | 说明 / Notes |
|------|-----|------|
| Embedding Model | `BAAI/bge-m3` | 中文检索最佳，首次自动下载 / Best for Chinese retrieval, auto-downloaded on first use |
| Chunk Size | `1024` | 配合大上下文模型 / Matches the large-context model |
| Chunk Overlap | `128` | |
| Top K | `5` | 先用小值确认稳定，再逐步增加到 20 / Start small to confirm stability, then gradually increase up to 20 |

#### 5.2.2 创建知识库 / Create the Knowledge Base

**Workspace → Knowledge → `+` New Knowledge**

```
Name:        明代军事装备知识库 (Ming Dynasty Military Equipment Knowledge Base)
Description: 明朝军事装备、火器、兵制专著文献 (Monographs on Ming military equipment, firearms, and military institutions)
```

上传 `data/cleaned/` 中的 TXT 文件，等待向量化完成。

Upload the TXT files from `data/cleaned/` and wait for vectorization to finish.

#### 5.2.3 创建专属模型 / Create the Dedicated Model

**Workspace → Models → `+` New Model**

**基本信息 / Basic info:**
```
Model ID:   ming-military-35b
Name:       明代军事助手 (qwen3.6:35b) [Ming Military Assistant]
Base Model: qwen3.6:35b
```

**Knowledge 选项卡 / Knowledge tab：** 勾选 `明代军事装备知识库` / Check `明代军事装备知识库` (Ming Dynasty Military Equipment Knowledge Base)

**System Prompt**（请按原样填入，内容为中文，因为助手本身以中文回答学术问题；大意见下方英文说明 / paste as-is — it is in Chinese because the assistant answers academic questions in Chinese; an English gloss follows below）：
```
你是一位专精明代军事史的学术助手，擅长明朝军事装备、火器制造、兵制、战术等领域。

回答时请：
1. 优先基于检索到的史料原文，保持学术准确性
2. 引用原文时注明文献来源
3. 对不确定的信息如实说明，不捏造史实
4. 使用规范的历史术语和学术语言

/no_think
```

> English gloss: "You are an academic assistant specializing in Ming dynasty military history — military equipment, firearms manufacturing, military institutions, and tactics. When answering: (1) prioritize the retrieved source text and maintain academic accuracy; (2) cite the source when quoting original text; (3) honestly flag uncertain information rather than fabricating history; (4) use standard historical terminology and academic language." The trailing `/no_think` disables the model's thinking mode.

> ⚠️ **`/no_think` 必须保留**（见调整记录 2026-09-13）
> **`/no_think` must be kept** (see the changelog entry for 2026-09-13)

**Advanced Parameters：**
```
num_ctx:         16384
temperature:     0.7
top_p:           0.9
repeat_penalty:  1.1
```

> ⚠️ **`num_ctx = 16384` 必须设置**（见调整记录 2026-09-13）
> **`num_ctx = 16384` must be set** (see the changelog entry for 2026-09-13)

点击 **Save** 保存，新建对话选择该模型即可使用。

Click **Save**, then start a new chat and select this model to use it.

#### 5.2.4 验证连接 / Verify the Connection

```cmd
docker exec ming-webui curl -s http://host.docker.internal:11434/api/tags
```

有 JSON 输出说明 Open-WebUI → Ollama 连接正常。

A JSON response means the Open-WebUI → Ollama connection is working.

---

### 5.3 Open-WebUI 配置（qwen3.5:9b 密集模型 — 推荐日常使用）/ Open-WebUI Configuration (qwen3.5:9b Dense Model — Recommended for Everyday Use)

#### 模型信息 / Model Info

| 参数 / Parameter | 值 / Value |
|------|-----|
| 架构 / Architecture | `qwen35`（密集模型，非 MoE / dense model, not MoE） |
| 参数量 / Parameters | 9.7B |
| 文件大小 / File size | 6.6 GB (Q4_K_M) |
| 最大上下文 / Max context | 262,144 tokens |
| VRAM 占用 / VRAM usage | ~7 GB，剩余 ~9 GB 可用于 KV Cache / ~9 GB left over for KV cache |
| 实测速度 / Measured speed | **82.6 tok/s**（benchmark 2026-09-12） |

**与 qwen3.6:35b 的取舍 / Trade-offs vs. qwen3.6:35b：**

| 项目 / Item | qwen3.5:9b | qwen3.6:35b MoE |
|------|-----------|-----------------|
| 生成速度 / Speed | **82 tok/s** | 53 tok/s |
| VRAM 占用 / VRAM | **6.6 GB** | 23 GB |
| 可用 KV Cache / Available KV cache | **~9 GB** | ~1 GB |
| 回答深度 / Answer depth | 一般 / average | **更详尽 / more detailed** |
| 可同时运行其他模型 / Can run other models concurrently | ✅ | ❌ |

> 9B 模型释放了大量显存给 KV Cache，实际可用上下文反而更长、更稳定。
>
> The 9B model frees up a lot of VRAM for KV cache, so the practically usable context is actually longer and more stable.

---

#### 5.3.1 RAG 全局参数（与 5.2.1 相同，可直接复用）/ Global RAG Parameters (Same as 5.2.1, Reusable As-is)

**Admin Panel → Settings → Documents** 设置不变 / settings unchanged：

| 参数 / Parameter | 值 / Value |
|------|-----|
| Embedding Model | `BAAI/bge-m3` |
| Chunk Size | `1024` |
| Chunk Overlap | `128` |
| Top K | `15` ← 比 35b 可以设更高，VRAM 更宽裕 / can be set higher than for the 35b model, since VRAM headroom is larger |

---

#### 5.3.2 创建专属模型 / Create the Dedicated Model

**Workspace → Models → `+` New Model**

**基本信息 / Basic info:**
```
Model ID:   ming-military-9b
Name:       明代军事助手 (qwen3.5:9b) [Ming Military Assistant]
Base Model: qwen3.5:9b
```

**Knowledge 选项卡 / Knowledge tab：** 勾选 `明代军事装备知识库` / Check `明代军事装备知识库` (Ming Dynasty Military Equipment Knowledge Base)

**System Prompt**（同 5.2.3，无需 `/no_think` / same as 5.2.3, `/no_think` not needed）：
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
>
> qwen3.5:9b runs fine with the default `num_ctx` — it doesn't need to be forced to 16384 like `qwen3.6:35b`. If you want longer conversation context, you can optionally set `num_ctx = 8192`.

---

#### 5.3.3 与 qwen3.6:35b 的配置差异说明 / Configuration Differences vs. qwen3.6:35b

qwen3.5:9b 在 Open-WebUI 挂载知识库后**开箱即用**，不需要 qwen3.6:35b 所需的两项强制修复：

Once the knowledge base is attached in Open-WebUI, qwen3.5:9b **works out of the box** — it doesn't need the two mandatory fixes that qwen3.6:35b requires:

| 配置项 / Config item | qwen3.5:9b | qwen3.6:35b MoE |
|--------|-----------|-----------------|
| `/no_think` | 非必须 / not required | **必须 / required** |
| `num_ctx = 16384` | 非必须 / not required | **必须 / required** |
| 知识库挂载后正常输出 / Normal output after attaching the knowledge base | ✅ 开箱即用 / out of the box | ❌ 需修复后才可用 / needs the fix first |

根本原因：qwen3.6:35b 是 MoE 架构，thinking 模式 + RAG 上下文会撑满默认 context 窗口导致无输出；qwen3.5:9b 是密集模型，上下文占用更小，不存在此问题。

Root cause: qwen3.6:35b is a MoE architecture, and thinking mode plus the RAG context fills up the default context window, resulting in no output. qwen3.5:9b is a dense model with smaller context usage, so it doesn't have this problem.

---

## 常见问题 / FAQ

| 问题 / Problem | 解决方案 / Solution |
|------|----------|
| OOM (显存不足 / out of VRAM) | 减小 `per_device_train_batch_size` 到 1，增大 `gradient_accumulation_steps` / Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` |
| 训练loss不下降 / Training loss doesn't decrease | 检查数据格式，降低 `learning_rate` 到 1e-5 / Check the data format, lower `learning_rate` to 1e-5 |
| OCR中文识别差 / Poor Chinese OCR accuracy | 改用 PaddleOCR 替代 Tesseract / Switch to PaddleOCR instead of Tesseract |
| ChromaDB慢 / ChromaDB is slow | 改用 FAISS 本地索引，或部署 Milvus / Switch to a local FAISS index, or deploy Milvus |
| 中文分词差 / Poor Chinese tokenization | chunk时按句号/换行分割，而非固定字符数 / Split chunks on periods/newlines instead of a fixed character count |
| RAG有来源但无输出 / RAG shows sources but no output | num_ctx 不足，改为 16384；System Prompt 末尾加 `/no_think` / `num_ctx` too small, set it to 16384; append `/no_think` to the end of the System Prompt |
| Open-WebUI 思考后无答案 / Open-WebUI shows thinking but no answer | qwen3.6:35b thinking 模式占满上下文，同上两步修复 / qwen3.6:35b's thinking mode fills up the context; apply the same two fixes above |
| 模型第一次响应极慢 / The model's first response is very slow | 正常，冷启动需 30~40 秒加载；在 System Prompt 中加预热请求 / Normal — cold start takes 30-40 seconds to load; add a warm-up request in the System Prompt |

---

## 参考资源 / References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3.5 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B-Instruct)
- [ChromaDB Docs](https://docs.trychroma.com)
- [RAGAS 评估框架 / RAGAS evaluation framework](https://docs.ragas.io)
- [BAAI/bge-m3 嵌入 / BAAI/bge-m3 embeddings](https://huggingface.co/BAAI/bge-m3)
- [机器A（4090 Laptop 16GB）基准测试报告 / Machine A (4090 Laptop 16GB) benchmark report](benchmark/reports/rtx4090-laptop-16gb.md) — 候选模型（qwen3.5:9b / qwen3.6:35b）选型依据 / rationale for the candidate models (qwen3.5:9b / qwen3.6:35b)
- [机器B（RTX 3060 12GB）基准测试报告 / Machine B (RTX 3060 12GB) benchmark report](benchmark/reports/rtx3060-12gb.md) — 排名因机器而异，`qwen2.5:7b`在此机器上更快 / the ranking differs by machine — `qwen2.5:7b` is faster on this one

---

## 调整记录 / Changelog

### 2026-09-13 — 基于4090基准测试确定候选部署模型 / Deployment Candidates Decided from the 4090 Benchmark

**背景 / Background：** 在 RTX 4090 Laptop（16GB VRAM，机器A）上对 7 个本地 Ollama 模型做了基准测试（`benchmark/run_benchmark.py`，完整数据见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)），覆盖生成速度、显存占用与回答质量。

Seven local Ollama models were benchmarked on an RTX 4090 Laptop (16GB VRAM, Machine A) using `benchmark/run_benchmark.py` (full data in [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)), covering generation speed, VRAM usage, and answer quality.

**结论 / Conclusion：** 本项目在 4090 环境下的部署候选确定为 / The deployment candidates for this project on the 4090 were determined to be：

| 候选模型 / Candidate | 理由 / Rationale |
|----------|----------|
| `qwen3.5:9b` | 实测 82.6 tok/s（次快），仅占 ~6.6GB 显存，剩余显存可用于更大 KV Cache，日常问答**开箱即用**（无需 `/no_think` 或调大 `num_ctx`） / Measured at 82.6 tok/s (second fastest), uses only ~6.6GB VRAM, leaving room for a larger KV cache; works **out of the box** for everyday Q&A (no `/no_think` or larger `num_ctx` needed) |
| `qwen3.6:35b`（MoE） | 实测 53.6 tok/s，回答最详尽（单次输出可达4000 tokens），适合深度分析与 SFT 数据生成，但需 `/no_think` + `num_ctx=16384` 两项修复才能在 Open-WebUI 挂载知识库后正常输出 / Measured at 53.6 tok/s, gives the most detailed answers (up to 4000 tokens in a single response), suited to deep analysis and SFT data generation, but needs both the `/no_think` and `num_ctx=16384` fixes to produce output after attaching a knowledge base in Open-WebUI |

`qwen3.8:latest`、`qwen3.6:27b` 因 thinking 模式拖慢至 12-14 tok/s、`gemma4:26b` 因 OOM 全部失败，均不作为候选。

`qwen3.8:latest` and `qwen3.6:27b` were both dragged down to 12-14 tok/s by thinking mode, and `gemma4:26b` failed entirely due to OOM — none of these are candidates.

具体部署步骤见阶段5.2（qwen3.6:35b）与阶段5.3（qwen3.5:9b）。

See Stage 5.2 (qwen3.6:35b) and Stage 5.3 (qwen3.5:9b) for the concrete deployment steps.

> **更新（同日）/ Update (same day)：** 微调基座已进一步由 `Qwen2.5-14B-Instruct` 改为 `Qwen3.5-9B-Instruct`，见下一条记录。
>
> The fine-tuning base was further changed from `Qwen2.5-14B-Instruct` to `Qwen3.5-9B-Instruct` — see the next entry.

### 2026-09-13 — 微调基座由 Qwen2.5:14B 改为 Qwen3.5:9B / Fine-tuning Base Changed from Qwen2.5:14B to Qwen3.5:9B

**原因 / Reason：** 上一条记录中，`qwen3.5:9b` 已被确定为4090环境下的推理部署候选之一，但微调基座当时仍沿用 `Qwen2.5-14B-Instruct`，导致训练（需24GB+显卡）与推理部署（16GB即可）使用两套不同规格的硬件与模型，流程不统一。改为以 `Qwen3.5-9B-Instruct` 作为微调基座后：

In the previous entry, `qwen3.5:9b` had already been determined as one of the inference deployment candidates on the 4090, but the fine-tuning base was still `Qwen2.5-14B-Instruct` at the time — meaning training (needs a 24GB+ GPU) and inference deployment (16GB is enough) used two different hardware/model specs, so the pipeline wasn't unified. After switching the fine-tuning base to `Qwen3.5-9B-Instruct`:

- 训练与部署统一到同一个9B模型，同一张16GB显卡（如 RTX 4090 Laptop）即可完成QLoRA训练+合并+部署全流程，无需额外24GB+显卡。/ Training and deployment are unified on the same 9B model — a single 16GB GPU (e.g. RTX 4090 Laptop) can complete the whole QLoRA training + merge + deployment pipeline, no extra 24GB+ GPU needed.
- 9B QLoRA训练显存估算：9B × 4bit ≈ 4.5GB模型 + 梯度/优化器 ≈ **总计10-13GB**。/ Estimated 9B QLoRA training VRAM: 9B × 4-bit ≈ 4.5GB for the model + gradients/optimizer ≈ **10-13GB total**.
- `qwen3.6:35b` 保持为不参与微调的RAG-only质量优先候选（35B MoE 微调门槛高，且当前无微调收益验证）。/ `qwen3.6:35b` remains a RAG-only, quality-first candidate that isn't fine-tuned (a 35B MoE has a high fine-tuning bar, and there's currently no verified benefit from fine-tuning it).

**受影响文件 / Affected files：** `03_finetune/qwen25_lora.yaml` 重命名为 `qwen35_lora.yaml`（model_name_or_path、template、run_name均已更新）、`train.sh`、`merge_lora.sh`（BASE_MODEL/MERGED_MODEL/template）、`05_deployment/deploy.sh`（MERGED_MODEL路径与基座回退逻辑）、`04_evaluation/eval_model.py`（--model与--judge-model默认值）、`04_evaluation/eval_rag.py`（--judge-model默认值，改为回答最详尽的`qwen3.6:35b`）、`01_data_pipeline/05_generate_qa.py`与`run_pipeline.sh`（QA生成默认模型改为`qwen3.6:35b`，取benchmark"SFT数据生成"推荐）、`02_rag/rag_server.py`与`build_and_serve.sh`（RAG默认模型改为`qwen3.5:9b`）。

`03_finetune/qwen25_lora.yaml` renamed to `qwen35_lora.yaml` (model_name_or_path, template, and run_name all updated); `train.sh`, `merge_lora.sh` (BASE_MODEL/MERGED_MODEL/template); `05_deployment/deploy.sh` (MERGED_MODEL path and base-model fallback logic); `04_evaluation/eval_model.py` (default values for --model and --judge-model); `04_evaluation/eval_rag.py` (default value for --judge-model, changed to the most detailed responder `qwen3.6:35b`); `01_data_pipeline/05_generate_qa.py` and `run_pipeline.sh` (default QA-generation model changed to `qwen3.6:35b`, per the benchmark's "SFT data generation" recommendation); `02_rag/rag_server.py` and `build_and_serve.sh` (default RAG model changed to `qwen3.5:9b`).

### 2026-09-13 — 新增机器B（RTX 3060 12GB）基准测试，模型排名因机器而异 / Added Machine B (RTX 3060 12GB) Benchmark — Ranking Differs by Machine

**背景 / Background：** 在第二台机器（CPU i7-12700F / RAM 64GB / GPU RTX 3060 12GB VRAM）上安装了 `qwen2.5:7b`，并对本机已安装的4个模型（`qwen2.5:7b`、`qwen3.5:9b`、`qwen3.6:35b`、`qwen2.5:14b`）运行了 `benchmark/run_benchmark.py`。

`qwen2.5:7b` was installed on a second machine (CPU i7-12700F / RAM 64GB / GPU RTX 3060 12GB VRAM), and `benchmark/run_benchmark.py` was run against the 4 models already installed there (`qwen2.5:7b`, `qwen3.5:9b`, `qwen3.6:35b`, `qwen2.5:14b`).

**处理方式 / Approach：** `run_benchmark.py` 内部硬编码输出路径为 `benchmark/report.md` 和 `benchmark/raw/*.json`（按模型名命名，不区分机器），运行前先把机器A已有数据移开以免被覆盖：
- 机器A的原始报告移到 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)
- 机器A的原始JSON数据移到 `benchmark/raw/rtx4090-laptop-16gb/`

`run_benchmark.py` hardcodes its output paths as `benchmark/report.md` and `benchmark/raw/*.json` (named per model, not per machine), so Machine A's existing data was moved aside first to avoid being overwritten:
- Machine A's raw report moved to [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)
- Machine A's raw JSON data moved to `benchmark/raw/rtx4090-laptop-16gb/`

脚本跑完后 `benchmark/report.md`（脚本手动补充Hardware/Key Findings/Recommendations等章节后）与 `benchmark/raw/*.json` 又手动归档为与机器A同样的命名格式：报告移到 `benchmark/reports/rtx3060-12gb.md`，原始JSON移到 `benchmark/raw/rtx3060-12gb/`，两台机器的目录结构完全对称。

After the script finished, `benchmark/report.md` (after manually adding Hardware/Key Findings/Recommendations sections) and `benchmark/raw/*.json` were manually archived using the same naming convention as Machine A: the report moved to `benchmark/reports/rtx3060-12gb.md`, and the raw JSON moved to `benchmark/raw/rtx3060-12gb/`, so the two machines' directory structures are fully symmetric.

**结果 / Result：** 12GB显存下模型排名与机器A（16GB）**不同**——`qwen2.5:7b`（65.8 tok/s）比`qwen3.5:9b`（53.9 tok/s）更快，且模型越大掉速越明显：`qwen3.6:35b` 从机器A的53.6 tok/s降到38.8 tok/s（约28%降幅，`ollama ps`显示CPU/GPU offload比例从机器A的多数GPU变为58%CPU/42%GPU）。详见 [benchmark/reports/rtx3060-12gb.md](benchmark/reports/rtx3060-12gb.md)"多机型基准与选型"及"Key Findings"章节。

With 12GB VRAM, the model ranking **differs** from Machine A (16GB) — `qwen2.5:7b` (65.8 tok/s) is faster than `qwen3.5:9b` (53.9 tok/s), and larger models slow down more noticeably: `qwen3.6:35b` drops from 53.6 tok/s on Machine A to 38.8 tok/s (about -28%; `ollama ps` shows the CPU/GPU offload split shifting from mostly-GPU on Machine A to 58% CPU / 42% GPU). See the "Multi-machine Benchmarks and Model Selection" and "Key Findings" sections of [benchmark/reports/rtx3060-12gb.md](benchmark/reports/rtx3060-12gb.md) for details.

**已知遗留问题 / Known outstanding issue：** `run_benchmark.py` 在Windows GBK控制台下打印`✅`等emoji时会抛出`UnicodeEncodeError`导致脚本在写完报告后以退出码1终止；报告和原始数据在崩溃前已正确写入，不影响结果正确性。已将崩溃的两行 `console.print` 改为纯ASCII文本修复该问题；但脚本硬编码的 `benchmark/report.md`/`benchmark/raw/` 输出路径本身仍需每次跑完后手动归档到按机器命名的子目录，尚未自动化。

`run_benchmark.py` raises a `UnicodeEncodeError` when printing emoji like `✅` on a Windows GBK console, causing the script to exit with code 1 after writing the report; the report and raw data are written correctly before the crash, so results are unaffected. The two crashing `console.print` lines were changed to plain ASCII text to fix this, but the script's hardcoded `benchmark/report.md`/`benchmark/raw/` output paths still need to be manually archived into per-machine subdirectories after every run — this has not yet been automated.

### 2026-09-13 — 新增两机对比报告，加测 gemma4:12b / Added a Two-machine Comparison Report, Additionally Tested gemma4:12b

**对比报告 / Comparison report：** 新增 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)，把机器A、机器B两份报告里都测过的模型放在一起算涨跌幅。核心发现：完全放得进显存的模型（不需要CPU offload）反而降速比例更大更一致（约35%），比需要offload的`qwen3.6:35b`（降28%）降得更多——说明降速根源是两张GPU本身的算力/带宽差异，不是显存不够。

Added [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md), putting the models tested on both Machine A's and Machine B's reports side by side to compute the change. Core finding: models that fit entirely in VRAM (no CPU offload needed) actually slow down by a larger and more consistent margin (~35%) than the offloaded `qwen3.6:35b` (-28%) — showing the root cause of the slowdown is the raw compute/bandwidth difference between the two GPUs, not insufficient VRAM.

**加测 gemma4:12b / Additionally tested gemma4:12b：** 用户在机器B上另外装了 `gemma4:12b`（机器A之前测过，58.80 tok/s）。为避免重跑其余4个模型引入的随机波动污染已经写好的结论和百分比，没有重新跑整个 `run_benchmark.py`，而是写了个小脚本单独调用其 `benchmark_model()` 函数只测这一个模型，结果单独存档、再合并进各文档：

The user separately installed `gemma4:12b` on Machine B (previously tested on Machine A at 58.80 tok/s). To avoid random variance from re-running the other 4 models polluting the conclusions and percentages already written up, the full `run_benchmark.py` wasn't re-run; instead, a small script called its `benchmark_model()` function directly to test just this one model, with the result archived separately and then merged into the documents:

- 机器B实测 37.83 tok/s（较机器A降35.7%），几乎与`qwen3.6:35b`（38.79 tok/s）打平，但体积只有其1/3且无需offload / Measured at 37.83 tok/s on Machine B (down 35.7% from Machine A), nearly matching `qwen3.6:35b` (38.79 tok/s) but at 1/3 the size and with no offload needed
- 原始数据：`benchmark/raw/rtx3060-12gb/gemma4_12b.json`，同时补入该目录下的 `_all_results.json` / Raw data: `benchmark/raw/rtx3060-12gb/gemma4_12b.json`, also merged into `_all_results.json` in that directory
- 已更新：`benchmark/reports/rtx3060-12gb.md`（新增第5个模型的排名/明细/发现/推荐）、`benchmark/reports/rtx4090-vs-rtx3060.md`（新增第三个"两机都测过"的模型，强化"~35%算力差距"这一发现）、本文档与 README.md 的机器B候选模型表 / Updated: `benchmark/reports/rtx3060-12gb.md` (added the 5th model's ranking/details/findings/recommendations), `benchmark/reports/rtx4090-vs-rtx3060.md` (added a third "tested on both machines" model, reinforcing the "~35% compute gap" finding), and the Machine B candidate-model tables in this document and README.md

### 2026-09-13 — Open-WebUI + qwen3.6:35b RAG 无输出问题修复 / Fixed Open-WebUI + qwen3.6:35b RAG No-output Issue

**现象 / Symptom：** 在 Open-WebUI 中使用 qwen3.6:35b 挂载知识库后，每次请求均显示"Thought for 10 seconds"并找到 4 个来源，但之后没有任何文字输出，仅显示 Follow up 建议。

After attaching a knowledge base to qwen3.6:35b in Open-WebUI, every request showed "Thought for 10 seconds" and found 4 sources, but produced no text output afterward — only follow-up suggestions were shown.

**根本原因 / Root cause：** 两个因素叠加导致 context 溢出 / Two factors compounded to overflow the context：

```
System Prompt   ≈  400 tokens
4个RAG chunks / 4 RAG chunks   ≈ 2000 tokens
Thinking过程 / Thinking process    ≈  800 tokens
用户问题 / User question         ≈  100 tokens
─────────────────────────────
合计 / Total             ≈ 3300 tokens  >  默认 num_ctx / default num_ctx 2048
```

模型完成 Thinking 后已无剩余 context 空间输出答案。

After finishing thinking, the model had no context space left to output an answer.

**修复措施 / Fix：**

1. **关闭 Thinking 模式 / Disable thinking mode**：在模型 System Prompt 末尾添加 `/no_think` / Append `/no_think` to the end of the model's System Prompt
2. **增大上下文窗口 / Increase the context window**：Advanced Parameters 中设置 `num_ctx = 16384` / Set `num_ctx = 16384` in Advanced Parameters

**验证结果 / Verification：** 两项同时生效后模型正常输出，速度恢复至约 53 tok/s。

With both fixes applied together, the model outputs normally again, with speed back to about 53 tok/s.

**影响范围 / Scope：** 所有在 Open-WebUI 中使用 Qwen3 系列 MoE 模型（qwen3.6:27b、qwen3.8:latest 等）并挂载知识库的配置，均需应用相同修复。

Any Open-WebUI configuration using a Qwen3-family MoE model (qwen3.6:27b, qwen3.8:latest, etc.) with a knowledge base attached needs the same fix applied.
