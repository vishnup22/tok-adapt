"""Perplexity evaluation for base vs. vocabulary-adapted checkpoints.

Implements the "Perplexity & Loss" item of Phase 5: confirms a model's
validation loss/perplexity on held-out target-language text, and compares
it directly against an unadapted base checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


@dataclass
class PerplexityResult:
    """Result of :func:`compute_perplexity`."""

    loss: float
    perplexity: float
    num_tokens: int


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    lines: Sequence[str],
    block_size: int = 512,
    device: Optional[str] = None,
) -> PerplexityResult:
    """Computes token-level causal LM perplexity over ``lines``.

    Concatenates ``lines`` (separated by EOS), chunks into ``block_size``
    windows, and averages cross-entropy loss over all tokens -- the
    standard whole-corpus perplexity definition (as opposed to the more
    expensive sliding-window scoring, which isn't needed to compare two
    checkpoints on the same held-out set).

    Args:
        model: The causal LM to evaluate.
        tokenizer: Its matching tokenizer.
        lines: Held-out text lines.
        block_size: Context window length per forward pass.
        device: Device to run on; defaults to the model's own device.

    Returns:
        A :class:`PerplexityResult` with average loss, perplexity
        (``exp(loss)``), and the token count the average was computed over.

    Raises:
        ValueError: If ``lines`` tokenizes to fewer than 2 tokens total.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    eos_id = tokenizer.eos_token_id or 0
    ids: List[int] = []
    for line in lines:
        ids.extend(tokenizer(line, add_special_tokens=False)["input_ids"])
        ids.append(eos_id)

    if len(ids) < 2:
        raise ValueError("Not enough tokens to compute perplexity.")

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, block_size):
        chunk = ids[start : start + block_size + 1]
        if len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk], device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        n_tokens = input_ids.shape[1] - 1  # transformers shifts labels internally
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    return PerplexityResult(loss=avg_loss, perplexity=perplexity, num_tokens=total_tokens)


def compare_perplexity(
    base_model: PreTrainedModel,
    base_tokenizer: PreTrainedTokenizerFast,
    adapted_model: PreTrainedModel,
    adapted_tokenizer: PreTrainedTokenizerFast,
    lines: Sequence[str],
    block_size: int = 512,
    device: Optional[str] = None,
) -> dict:
    """Computes and compares perplexity for a base and adapted checkpoint.

    Args:
        base_model: The unadapted base model.
        base_tokenizer: The base model's original tokenizer.
        adapted_model: The vocabulary-adapted / trained model.
        adapted_tokenizer: The adapted model's (possibly expanded) tokenizer.
        lines: Held-out text lines, evaluated identically for both models.
        block_size: Context window length per forward pass.
        device: Device to run both models on.

    Returns:
        A dict with ``base`` and ``adapted`` :class:`PerplexityResult`
        values, plus ``perplexity_delta`` (adapted minus base, so negative
        means the adapted model is better) and an ``improved`` bool.
    """
    base = compute_perplexity(base_model, base_tokenizer, lines, block_size=block_size, device=device)
    adapted = compute_perplexity(adapted_model, adapted_tokenizer, lines, block_size=block_size, device=device)
    return {
        "base": base,
        "adapted": adapted,
        "perplexity_delta": adapted.perplexity - base.perplexity,
        "improved": adapted.perplexity < base.perplexity,
    }
