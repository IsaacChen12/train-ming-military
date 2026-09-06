#!/bin/bash
# WSL Ubuntu 22.04 一键环境初始化脚本
# 用法: bash 00_setup/wsl_setup.sh

set -e
PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
MODEL_DIR="/mnt/d/models"

echo "=== [1/6] 系统依赖 ==="
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    poppler-utils \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

echo "=== [2/6] Python 虚拟环境 ==="
cd "$PROJECT_ROOT"
python3.11 -m venv .venv
source .venv/bin/activate

echo "=== [3/6] PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/6] 项目依赖 ==="
pip install -r 00_setup/requirements.txt

echo "=== [5/6] LLaMA-Factory ==="
if [ ! -d "03_finetune/LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git \
        03_finetune/LLaMA-Factory
fi
cd 03_finetune/LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes,qwen]"
cd "$PROJECT_ROOT"

echo "=== [6/6] Ollama ==="
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== 创建数据目录 ==="
mkdir -p data/{raw,txt,cleaned,chunks,qa_pairs,sft}
mkdir -p outputs results
mkdir -p "$MODEL_DIR"

echo ""
echo "✅ 环境初始化完成！"
echo ""
echo "下一步："
echo "  1. 将书籍PDF放入 data/raw/"
echo "  2. 启动Ollama: ollama serve &"
echo "  3. 拉取基座模型: ollama pull qwen2.5:14b"
echo "  4. 运行数据管线: bash 01_data_pipeline/run_pipeline.sh"
