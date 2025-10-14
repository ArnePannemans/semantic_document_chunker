# Semantic Chunker

A semantic document chunking system optimized for Retrieval-Augmented Generation (RAG). Fine-tunes Qwen3-14B to intelligently split documents at semantic boundaries.

## Setup

Create virtual environment:
```bash
python -m venv venv  # Python 3.11+ recommended
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file with your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

Install pre-commit hooks:
```bash
pre-commit install
```

## Usage

### Data Pipeline

#### 1. Label Documents

Use Gemini to identify semantic boundaries in documents:

```bash
python -m src.data generate_labels \
    --input data/dummy/documents \
    --output data/dummy/labeled
```

Configuration is managed via `DataPipelineConfig` in `src/config.py`.

#### 2. Prepare Training Pairs

Convert labeled documents into training pairs with location tags:

```bash
python -m src.data prepare_training_pairs \
    --input data/dummy/labeled \
    --output data/dummy/training_pairs
```

### Training

Train the Qwen3-14B model with LoRA:

```bash
python -m src.training \
    --input data/dummy/training_pairs \
    --output models/qwen3-run-01
```

Configuration is managed via `TrainingConfig` in `src/config.py`. The training script:
- Automatically splits data into train/val/test sets (80/10/10)
- Logs metrics to Weights & Biases
- Saves checkpoints and final model
- Stores test samples for evaluation

### Inference

Two modes for testing and demonstration:

#### Mode 1: Test with Sample (Ground Truth)

Compare model predictions against labeled data:

```bash
# Test base model
python -m src.inference \
    --sample data/dummy/training_pairs/dummy_doc.json

# Test with LoRA adapter
python -m src.inference \
    --sample data/dummy/training_pairs/dummy_doc.json \
    --adapter models/qwen3-run-01/checkpoint-140
```

#### Mode 2: Chunk Raw Text

Chunk any document for demos or production use:

```bash
# Test base model
python -m src.inference \
    --text "Your document text here..."

# Test with LoRA adapter
python -m src.inference \
    --text "Your document text here..." \
    --adapter models/qwen3-run-01/checkpoint-140
```


Configuration (temperature, max tokens, etc.) is managed via `InferenceConfig` in `src/config.py`.

### Python API

Use the chunker programmatically in your application:

```python
from src.inference import SemanticChunker

# Initialize chunker
chunker = SemanticChunker(adapter_path="models/qwen3-run-01/checkpoint-140")

# Chunk a document
text = "Your document text here..."
chunks = chunker.chunk_document(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

## Example Workflow

**Original document:**
```
Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data. Deep learning is a type of machine learning. It uses neural networks with multiple layers. These models can recognize patterns in images and text.
```

**After inserting split markers (Step 1):**
```
Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data.
<SPLIT>
Deep learning is a type of machine learning. It uses neural networks with multiple layers. These models can recognize patterns in images and text.
```

**Training pair output (Step 2):**
```json
{
  "input": "<|loc_0|>Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data.<|loc_1|>Deep learning is a type of machine learning. It uses neural networks with multiple layers.<|loc_2|>These models can recognize patterns in images and text.",
  "output": "[1]"
}
```

The model learns to predict split positions by seeing where `<SPLIT>` markers were placed (indices `[1, 2]`).

## Development

Code formatting and linting with Ruff:
```bash
ruff check . --fix  # Lint and fix issues
ruff format .       # Format code
```

Pre-commit hooks run automatically on each commit.
