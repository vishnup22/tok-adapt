"""Phase 5: perplexity and downstream evaluation."""

from __future__ import annotations

from tok_adapt.evaluation.downstream import (
    MultipleChoiceResult,
    TranslationScore,
    evaluate_multiple_choice,
    score_translations,
)
from tok_adapt.evaluation.perplexity import PerplexityResult, compare_perplexity, compute_perplexity

__all__ = [
    "PerplexityResult",
    "compute_perplexity",
    "compare_perplexity",
    "TranslationScore",
    "score_translations",
    "MultipleChoiceResult",
    "evaluate_multiple_choice",
]
