"""Shared pytest fixtures for tok_adapt tests."""

from __future__ import annotations

import torch
import torch.nn as nn


def untie_output_embeddings(model) -> None:
    """Forces a model's lm_head to be a distinct tensor from its input embeddings.

    Many small causal LM configs (GPT-2 included) tie ``lm_head`` to
    ``embed_tokens`` by default, which means the "untied" branch of
    embedding-slicing/mirroring code is never exercised unless a test
    deliberately breaks the tie. This gives the output embeddings their own
    weight matrix, seeded from the input embeddings plus noise so that
    per-row values are distinguishable (useful for asserting that slicing
    picks up the *correct* row, not just a matching shape).

    Args:
        model: A causal LM with tied input/output embeddings.

    Mutates ``model`` in place: sets ``config.tie_word_embeddings = False``
    and replaces its output embedding layer with an independent
    ``nn.Linear``.
    """
    model.config.tie_word_embeddings = False
    input_embeddings = model.get_input_embeddings()
    vocab_size, hidden_size = input_embeddings.weight.shape

    new_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    with torch.no_grad():
        new_lm_head.weight.copy_(input_embeddings.weight)
        new_lm_head.weight.add_(torch.randn_like(new_lm_head.weight) * 0.1)

    model.set_output_embeddings(new_lm_head)
