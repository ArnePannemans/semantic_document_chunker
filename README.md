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
python -m src.data_pipeline generate_labels \
    --input data/dummy/documents \
    --output data/dummy/labeled
```

Configuration is managed via `DataPipelineConfig` in `src/config.py`.

#### 2. Prepare Training Pairs

Convert labeled documents into training pairs with location tags:

```bash
python -m src.data_pipeline prepare_training_pairs \
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

### API Server

Start the FastAPI server to serve the semantic chunking model:

```bash
./scripts/start_server.sh
```

The server will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

#### Endpoints

- `GET /health` - Health check and model status
- `POST /v1/chunk` - Chunk a document into semantic sections

#### Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/v1/chunk" \
  -H "Content-Type: application/json" \
  -d '{"document": "Your document text here..."}'
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chunk",
    json={"document": "Your document text here..."}
)
result = response.json()
print(f"Created {result['num_chunks']} chunks")
for i, chunk in enumerate(result['chunks'], 1):
    print(f"Chunk {i}: {chunk}")
```

Test the API:
```bash
python scripts/test_api.py
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
