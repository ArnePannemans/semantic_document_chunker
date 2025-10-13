# Semantic Chunker

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

### 1. Insert Split Markers

Use Gemini to identify semantic boundaries in documents:

```bash
python -m src.scripts.insert_split_markers \
    --input data/dummy/documents \
    --output data/dummy/labeled
```

### 2. Prepare Training Pairs

Convert labeled documents into training pairs with location tags:

```bash
python -m src.scripts.prepare_training_pairs \
    --input data/dummy/labeled \
    --output data/dummy/training_pairs
```

## Example

**Original document:**
```
Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data. Deep learning is a type of machine learning. It uses neural networks with multiple layers. These models can recognize patterns in images and text.
```

**After inserting split markers (Step 1):**
```
Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data.
<SPLIT>
Deep learning is a type of machine learning. It uses neural networks with multiple layers. These models can recognize patterns in images and text. 
<SPLIT>
```

**Training pair output (Step 2):**
```json
{
  "input": "<|loc_0|> Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data. <|loc_1|> Deep learning is a type of machine learning. It uses neural networks with multiple layers. <|loc_2|> These models can recognize patterns in images and text. <|loc_3|>",
  "output": "[1, 3]"
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
