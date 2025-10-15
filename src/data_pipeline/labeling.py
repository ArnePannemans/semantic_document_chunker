"""Document labeling using Gemini to insert semantic split markers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config import DataPipelineConfig
from src.core.chunking import get_chunk_bounds, validate_document
from src.core.prompts import render_labeling_prompts
from src.data_pipeline.utils.gemini import call_gemini


class DocumentLabelingError(Exception):
    """Exception raised when document labeling fails."""

    pass


def label_document(text: str, config: DataPipelineConfig) -> str:
    """
    Label document by inserting semantic split markers using Gemini.

    Args:
        text: Document text to process
        config: Data pipeline configuration

    Returns:
        Labeled text with split markers inserted

    Raises:
        DocumentLabelingError: If validation fails, LLM call fails, or response is empty
    """
    if not validate_document(text, config.min_words, config.max_words):
        word_count = len(text.split())
        raise DocumentLabelingError(
            f"Word count {word_count} outside acceptable range "
            f"({config.min_words}-{config.max_words})"
        )

    word_count = len(text.split())
    min_chunks, max_chunks = get_chunk_bounds(word_count, config.chunking)

    system_prompt, user_prompt = render_labeling_prompts(
        document=text,
        word_count=word_count,
        min_chunks=min_chunks,
        max_chunks=max_chunks,
        config=config.chunking,
    )

    labeled_text = call_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=config.gemini_model,
        temperature=config.gemini_temperature,
        max_output_tokens=config.gemini_max_tokens,
        thinking_budget=config.gemini_thinking_budget,
    )

    if not labeled_text or not labeled_text.strip():
        raise DocumentLabelingError("Empty response from LLM")

    return labeled_text.strip()


def generate_labels(
    input_dir: str,
    output_dir: str,
    config: DataPipelineConfig | None = None,
) -> None:
    """
    Label all documents in input directory and save to output directory.

    Args:
        input_dir: Directory containing .txt files
        output_dir: Directory to write processed files
        config: Data pipeline configuration (uses default if None)
    """
    if config is None:
        config = DataPipelineConfig()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    document_paths = list(input_path.rglob("*.txt"))
    if not document_paths:
        print(f"No .txt files found in {input_dir}")
        return

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        # Submit all documents for processing
        processing_tasks = {
            executor.submit(
                label_document,
                file_path.read_text(encoding="utf-8"),
                config,
            ): file_path
            for file_path in document_paths
        }

        # Process results as they complete
        for completed_task in tqdm(
            as_completed(processing_tasks),
            total=len(document_paths),
            desc="Creating labeled documents",
        ):
            document_path = processing_tasks[completed_task]
            try:
                labeled_content = completed_task.result()
                output_file = output_path / document_path.name
                output_file.write_text(labeled_content, encoding="utf-8")
            except Exception as error:
                print(f"Failed: {document_path.name} - {error}")
