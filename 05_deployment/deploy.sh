#!/bin/bash
# 一键部署：注册模型到Ollama + 启动RAG服务 + 验证
set -e

PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
MERGED_MODEL="/mnt/d/models/ming-military-14b"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=========================================="
echo " 明代军事装备大模型 — 部署"
echo "=========================================="

# 确保Ollama在运行
if ! pgrep -x ollama > /dev/null; then
    echo ">>> 启动Ollama..."
    ollama serve &
    sleep 3
fi

# 判断是否有微调模型，否则使用基座
if [ -d "$MERGED_MODEL" ]; then
    echo ">>> 注册微调模型 ming-military..."
    cp "$PROJECT_ROOT/05_deployment/Modelfile" "$MERGED_MODEL/Modelfile"
    cd "$MERGED_MODEL"
    ollama create ming-military -f Modelfile
    cd "$PROJECT_ROOT"
    ACTIVE_MODEL="ming-military"
else
    echo ">>> 微调模型未找到，使用基座模型 qwen2.5:14b"
    ollama pull qwen2.5:14b
    ACTIVE_MODEL="qwen2.5:14b"
fi

# 启动RAG服务（后台）
echo ">>> 启动RAG服务..."
nohup python 02_rag/rag_server.py \
    --db-path data/chroma_db \
    --llm-base-url http://localhost:11434/v1 \
    --model "$ACTIVE_MODEL" \
    --port 8080 \
    > outputs/rag_server.log 2>&1 &
RAG_PID=$!
echo "    RAG PID: $RAG_PID"
sleep 5

# 健康检查
echo ">>> 健康检查..."
if curl -sf http://localhost:8080/health > /dev/null; then
    echo "    ✅ RAG服务正常"
else
    echo "    ⚠️  RAG服务未响应，检查 outputs/rag_server.log"
fi

# 启动Docker（Open-WebUI）
echo ">>> 启动Open-WebUI..."
docker-compose -f 00_setup/docker-compose.yml up -d 2>/dev/null || \
    echo "    ⚠️  Docker未运行，跳过WebUI"

echo ""
echo "=========================================="
echo " 部署完成！"
echo "=========================================="
echo ""
echo "  命令行对话:  ollama run $ACTIVE_MODEL"
echo "  RAG API:     http://localhost:8080"
echo "  Web界面:     http://localhost:3000"
echo "  搜索测试:    python 02_rag/test_rag.py"
echo ""
echo "  停止RAG服务: kill $RAG_PID"
