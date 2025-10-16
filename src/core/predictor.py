"""Core prediction logic for semantic chunking."""

import re
import logging

from src.config import InferenceConfig
from src.core.chunking import split_into_sentences, tag_sentences
from src.core.model_loader import load_model
from src.core.prompts import format_as_chat_messages, render_prediction_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Semantic chunking model for document splitting."""

    def __init__(
        self,
        adapter_path: str | None = None,
        config: InferenceConfig | None = None,
    ):
        """
        Initialize semantic chunker.

        Args:
            adapter_path: Path to LoRA adapter directory (None for base model)
            config: Inference configuration (uses default if None)
        """
        self.config = config or InferenceConfig()
        self.model, self.tokenizer = load_model(adapter_path, self.config)

    def predict_split_locations(
        self,
        tagged_text: str,
    ) -> str:
        """
        Predict split locations for tagged document.

        Args:
            tagged_text: Document with location markers

        Returns:
            Raw model output string (e.g., "[3, 7, 12]")
        """

        # Render prompts
        system_prompt, user_prompt = render_prediction_prompts(
            document=tagged_text,
            config=self.config.chunking,
        )

        # Format as chat messages
        messages = format_as_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated part (skip input prompt)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True)

        return prediction.strip()

    def parse_indices(self, output: str) -> list[int]:
        """
        Parse model output to extract split indices.

        Args:
            output: Raw model output string

        Returns:
            List of integer indices
        """
        # Try to extract a list from the output
        match = re.search(r"\[([\d,\s]+)\]", output)
        if match:
            indices_str = match.group(1)
            return [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        return []

    def split_by_indices(self, sentences: list[str], indices: list[int]) -> list[str]:
        """
        Split sentences into chunks based on split indices.

        Args:
            sentences: List of sentences
            indices: Indices where splits should occur

        Returns:
            List of text chunks
        """
        if not indices:
            return [" ".join(sentences)]

        chunks = []
        start = 0

        for idx in sorted(indices):
            if 0 < idx <= len(sentences):
                chunk = " ".join(sentences[start:idx])
                if chunk:
                    chunks.append(chunk)
                start = idx

        # Add remaining sentences
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_document(
        self,
        text: str,
    ) -> list[str]:
        """
        Full pipeline: split into sentences → tag → predict → chunk.

        Args:
            text: Raw document text

        Returns:
            List of semantic chunks
        """
        # Split text into sentences
        sentences = split_into_sentences(
            text,
            min_words=self.config.chunking.min_words_per_sentence,
        )
        if not sentences:
            return []

        # Tag sentences with location markers
        tagged_text = tag_sentences(sentences, self.config.chunking)

        # Predict split indices
        output = self.predict_split_locations(tagged_text)
        logger.info(f"Output: {output}")
        indices = self.parse_indices(output)

        # Split into chunks
        chunks = self.split_by_indices(sentences, indices)

        return chunks
