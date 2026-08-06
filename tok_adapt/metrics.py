"""Fertility ratio and sequence compression metrics for tokenizer benchmarking."""

from __future__ import annotations

from typing import Dict, List

from transformers import PreTrainedTokenizerFast


class FertilityEvaluator:
    """Computes tokenizer efficiency statistics over a corpus of text.

    Args:
        tokenizer: The tokenizer to evaluate.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer

    def compute_fertility(self, text_list: List[str]) -> Dict[str, float]:
        """Computes token-to-word and token-to-byte ratios over a text corpus.

        Args:
            text_list: List of raw text strings to evaluate.

        Returns:
            A dict with ``total_tokens``, ``total_words``, ``total_bytes``,
            ``token_to_word_ratio`` (fertility w.r.t. whitespace words), and
            ``token_to_byte_ratio`` (fertility w.r.t. UTF-8 bytes).
        """
        total_tokens = 0
        total_words = 0
        total_bytes = 0

        for text in text_list:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(ids)
            total_words += len(text.split())
            total_bytes += len(text.encode("utf-8"))

        return {
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_bytes": total_bytes,
            "token_to_word_ratio": (total_tokens / total_words) if total_words else 0.0,
            "token_to_byte_ratio": (total_tokens / total_bytes) if total_bytes else 0.0,
        }

    def compare_sequence_compression(
        self, other_tokenizer: PreTrainedTokenizerFast, text_list: List[str]
    ) -> Dict[str, float]:
        """Compares sequence length produced by this tokenizer vs. another.

        Args:
            other_tokenizer: The tokenizer to compare against (e.g. an
                adapted/extended tokenizer).
            text_list: List of raw text strings, tokenized identically by
                both tokenizers for a fair comparison.

        Returns:
            A dict with ``base_tokens`` (this tokenizer's total token count),
            ``adapted_tokens`` (the other tokenizer's total token count),
            ``compression_ratio`` (``adapted_tokens / base_tokens``, so <1.0
            means the other tokenizer produces shorter sequences), and
            ``token_reduction_pct`` (percentage reduction from base to
            adapted).
        """
        base_tokens = sum(
            len(self.tokenizer.encode(t, add_special_tokens=False)) for t in text_list
        )
        adapted_tokens = sum(
            len(other_tokenizer.encode(t, add_special_tokens=False)) for t in text_list
        )
        reduction_pct = (
            ((base_tokens - adapted_tokens) / base_tokens * 100.0) if base_tokens else 0.0
        )

        return {
            "base_tokens": base_tokens,
            "adapted_tokens": adapted_tokens,
            "compression_ratio": (adapted_tokens / base_tokens) if base_tokens else 0.0,
            "token_reduction_pct": reduction_pct,
        }
