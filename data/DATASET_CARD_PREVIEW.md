# Semantic Document Chunker Dataset

## Dataset Description

This dataset contains 639 documents with semantic boundary annotations for training text chunking models. Each document includes the raw text, text with semantic split markers, and training pairs with location tags.

## Dataset Structure

Each example contains:
- `id`: Unique document identifier
- `raw_text`: Original document text
- `labeled_text`: Document with `<SPLIT>` markers indicating semantic boundaries
- `tagged_input`: Text with `<|loc_N|>` location tags for each sentence chunk
- `split_indices`: List of integers indicating where original splits occurred

## Data Collection

Documents were labeled using **Gemini 2.5 Pro** with the following semantic boundary detection prompt:

```jinja2
You are an expert document segmenter for Retrieval-Augmented Generation (RAG).
Your task: Insert {{ split_marker }} markers at natural semantic boundaries to divide documents into coherent, self-contained chunks optimized for downstream retrieval and generation.

Aim for an average chunk size of {{ target_chunk_size }} words, with an acceptable range between {{ min_chunk_size }} and {{ max_chunk_size }} words per chunk.

Characteristics of great chunks:
- Cohesive
- Atomic (answerable independently)
- LLM-friendly (avoids cutting mid-structure)

Tips for good split locations:
- a shift in topic or sub-topic
- a natural conclusion to a section
- A change in content type
- The introduction of a new section or heading

Guidelines:
- Prioritize semantic coherence over exact word counts
- Never insert a marker after the document's final content
- Preserve all original formatting: line breaks, spacing, punctuation, and structure

Output:
Return ONLY the original document text with {{ split_marker }} markers inserted. Include no explanations, metadata, or additional text.
```

For more details on the data preparation pipeline, see the [GitHub repository](https://github.com/ArnePannemans/semantic_document_chunker).

## Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("ArnePannemans/semantic-document-chunker")

# Access training pairs
for example in ds:
    print(f"ID: {example['id']}")
    print(f"Tagged input: {example['tagged_input'][:100]}...")
    print(f"Split indices: {example['split_indices']}")
```

## Citation

If you use this dataset, please cite the repository:

```bibtex
@misc{pannemans2025semantic,
  author = {Pannemans, Arne},
  title = {Semantic Document Chunker},
  year = {2025},
  url = {https://github.com/ArnePannemans/semantic_document_chunker}
}
```
