#!/bin/bash
set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found at venv/bin/activate"
fi

# Configuration
BASE_MODEL="unsloth/qwen3-14b-unsloth-bnb-4bit"
LORA_PATH="models/qwen3-14b-4bit-run-90-10-0/final_model"

MAX_LORA_RANK=16
GPU_MEMORY=0.85
MAX_MODEL_LEN=8192
PORT=8000

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  LoRA Path: $LORA_PATH"
echo "  Max LoRA Rank: $MAX_LORA_RANK"
echo "  GPU Memory Utilization: $GPU_MEMORY"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Port: $PORT"
echo ""

# Start vLLM server
echo "Loading model with LoRA adapter..."
python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --tokenizer "$LORA_PATH" \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --enable-lora \
  --lora-modules qwen3_lora="$LORA_PATH" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --host 0.0.0.0 \