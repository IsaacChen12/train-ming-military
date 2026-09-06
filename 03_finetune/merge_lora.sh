#!/bin/bash
# 合并LoRA权重到完整模型，并转换为Ollama可用格式
set -e

PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
LLAMAFACTORY_DIR="$PROJECT_ROOT/03_finetune/LLaMA-Factory"
BASE_MODEL="/mnt/d/models/Qwen2.5-14B-Instruct"
LORA_WEIGHTS="$PROJECT_ROOT/outputs/lora_weights"
MERGED_MODEL="/mnt/d/models/ming-military-14b"

cd "$PROJECT_ROOT"
source .venv/bin/activate
cd "$LLAMAFACTORY_DIR"

echo ">>> 合并LoRA权重..."
llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$LORA_WEIGHTS" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "$MERGED_MODEL" \
    --export_size 4 \
    --export_device cpu \
    --export_legacy_format false

echo ""
echo "✅ 模型已合并到 $MERGED_MODEL"

echo ""
echo ">>> 创建Ollama Modelfile..."
cp "$PROJECT_ROOT/05_deployment/Modelfile" "$MERGED_MODEL/Modelfile"

echo ""
echo ">>> 注册到Ollama..."
cd "$MERGED_MODEL"
ollama create ming-military -f Modelfile

echo ""
echo ">>> 验证部署..."
ollama run ming-military "明朝神机营装备了哪些主要火器？请简述其性能。"

echo ""
echo "✅ 部署完成！使用 'ollama run ming-military' 开始对话"
