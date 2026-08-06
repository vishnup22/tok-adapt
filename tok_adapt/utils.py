"""Model and tokenizer loading/saving helpers shared across tok_adapt modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


def load_tokenizer(path_or_id: str) -> PreTrainedTokenizerFast:
    """Loads a fast Hugging Face tokenizer from a local path or hub id.

    Args:
        path_or_id: Local directory path or Hugging Face Hub model id.

    Returns:
        A ``PreTrainedTokenizerFast`` instance.

    Raises:
        TypeError: If the resolved tokenizer is not backed by the Rust
            ``tokenizers`` library (i.e. not "fast"), which tok_adapt
            requires for vocabulary manipulation.
    """
    tokenizer = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"'{path_or_id}' did not resolve to a fast tokenizer. "
            "tok_adapt requires a PreTrainedTokenizerFast backend."
        )
    return tokenizer


def load_model(
    path_or_id: str, dtype: Optional[torch.dtype] = None
) -> PreTrainedModel:
    """Loads a causal language model from a local path or hub id.

    Args:
        path_or_id: Local directory path or Hugging Face Hub model id.
        dtype: Optional torch dtype to load the model weights in.

    Returns:
        A ``PreTrainedModel`` instance in eval-ready state.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(path_or_id, **kwargs)
    return model


def load_model_and_tokenizer(
    path_or_id: str, dtype: Optional[torch.dtype] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Convenience loader returning ``(model, tokenizer)`` for the same source."""
    return load_model(path_or_id, dtype=dtype), load_tokenizer(path_or_id)


def save_model_and_tokenizer(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, output_dir: str
) -> Path:
    """Saves a model and tokenizer pair to disk, creating directories as needed.

    Args:
        model: The model to persist.
        tokenizer: The tokenizer to persist alongside the model.
        output_dir: Destination directory.

    Returns:
        The resolved output directory as a ``Path``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
