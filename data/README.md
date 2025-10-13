# Data

Data is organized by version, with each version containing three subdirectiories for each stage of the pipeline:

## 1. `documents/`
Raw input documents (`.txt` files) before processing.
- One document per file
- Plain text format

## 2. `labeled/`
Documents with `<SPLIT>` markers inserted at semantic boundaries by LLM.
- Output from `insert_split_markers` script
- Same filename as source document

## 3. `training_pairs/`
Training data generated from labeled documents.
- Output from `prepare_training_data` script
- Format: `.json` files


## Versions

- **dummy/**: Small test dataset for validating scripts
- **v1/**: First production dataset (639 documents)