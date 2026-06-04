"""Data utilities for V3 self-evolution pipeline."""

from verl_inf_evolve.data.document_dataset import DocumentDataset

__all__ = ["tokenize_and_pad_to_dataproto", "DocumentDataset"]


def __getattr__(name):
    if name == "tokenize_and_pad_to_dataproto":
        from verl_inf_evolve.data.batch_utils import tokenize_and_pad_to_dataproto
        return tokenize_and_pad_to_dataproto
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
