#!/bin/bash
# LoRA微调训练脚本
# 用法：bash 03_finetune/train.sh
set -e

PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
LLAMAFACTORY_DIR="$PROJECT_ROOT/03_finetune/LLaMA-Factory"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=========================================="
echo " Qwen2.5-14B LoRA 微调"
echo "=========================================="

# Step 1: 准备数据集
echo ""
echo ">>> 准备SFT数据集..."
python 03_finetune/prepare_sft.py \
    --qa-dir data/qa_pairs \
    --output-dir data/sft \
    --format alpaca \
    --train-ratio 0.9

# Step 2: 软链接数据到LLaMA-Factory
echo ""
echo ">>> 链接数据集..."
mkdir -p "$LLAMAFACTORY_DIR/data"

# 将训练数据复制到LLaMA-Factory的data目录
cp data/sft/train.json "$LLAMAFACTORY_DIR/data/ming_military_train.json"
cp data/sft/val.json "$LLAMAFACTORY_DIR/data/ming_military_val.json"

# 注册数据集
python -c "
import json
from pathlib import Path

p = Path('$LLAMAFACTORY_DIR/data/dataset_info.json')
info = json.loads(p.read_text()) if p.exists() else {}
info['ming_military_train'] = {
    'file_name': 'ming_military_train.json',
    'formatting': 'alpaca',
    'columns': {'prompt': 'instruction', 'response': 'output'}
}
info['ming_military_val'] = {
    'file_name': 'ming_military_val.json',
    'formatting': 'alpaca',
    'columns': {'prompt': 'instruction', 'response': 'output'}
}
p.write_text(json.dumps(info, ensure_ascii=False, indent=2))
print('数据集注册完成')
"

# Step 3: 开始训练
echo ""
echo ">>> 开始LoRA训练..."
echo "    GPU信息："
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

mkdir -p outputs/logs

cd "$LLAMAFACTORY_DIR"
llamafactory-cli train "$PROJECT_ROOT/03_finetune/qwen25_lora.yaml" \
    2>&1 | tee "$PROJECT_ROOT/outputs/training.log"

echo ""
echo "✅ 训练完成！LoRA权重保存在 outputs/lora_weights/"
echo ""
echo ">>> 下一步：合并权重"
echo "    bash 03_finetune/merge_lora.sh"
