# Data

Data is organized by version, with each version containing three subdirectiories for each stage of the pipeline:

## 1. `documents/`
Raw input documents (`.txt` files) before processing.
- One document per file
- Plain text format

## 2. `labeled/`
Documents with `<SPLIT>` markers inserted at semantic boundaries by LLM.
- Output from `python -m src.data generate_labels`
- Same filename as source document

## 3. `training_pairs/`
Training data generated from labeled documents.
- Output from `python -m src.data prepare_training_pairs`
- Format: `.json` files with 'input' and 'output' keys


## Versions

- **dummy/**: Small test dataset for validating scripts
- **v1/**: First production dataset (639 documents)