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


def ensure_pad_token(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    """Ensures a tokenizer has a pad token, falling back to its EOS token.

    Many causal LM tokenizers (GPT-2's included) ship without a dedicated
    pad token since padding is never needed at single-sequence inference
    time. Batched training does need one; reusing ``eos_token`` as
    ``pad_token`` is the standard fix used by most HF fine-tuning scripts.

    Args:
        tokenizer: The tokenizer to check/mutate in place.

    Returns:
        The same tokenizer instance, for chaining.

    Raises:
        ValueError: If the tokenizer has neither a pad token nor an EOS
            token to fall back to.
    """
    if tokenizer.pad_token is not None:
        return tokenizer
    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer has no pad_token and no eos_token to fall back to; "
            "set tokenizer.pad_token explicitly before training."
        )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
