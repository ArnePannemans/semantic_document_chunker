"""Data pipeline for document labeling and training pair preparation."""

from src.data.labeling import generate_labels
from src.data.prepare_training_data import create_training_pairs

__all__ = ["generate_labels", "create_training_pairs"]
