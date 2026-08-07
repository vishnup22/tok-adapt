"""Downstream benchmarking: translation quality and multiple-choice accuracy.

Implements the "Downstream Accuracy" item of Phase 5 -- BLEU/chrF++ for
translation-style generation, and a lightweight multiple-choice accuracy
harness in the shape of MMLU/domain-QA benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import sacrebleu
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


@dataclass
class TranslationScore:
    """Corpus-level translation quality scores (0-100 scale)."""

    bleu: float
    chrf: float


def score_translations(hypotheses: Sequence[str], references: Sequence[str]) -> TranslationScore:
    """Scores generated translations against references with BLEU and chrF++.

    Args:
        hypotheses: Model-generated translations.
        references: Gold reference translations, one per hypothesis.

    Returns:
        A :class:`TranslationScore` with corpus-level BLEU and chrF++
        (word order 2, i.e. "chrF++") scores.

    Raises:
        ValueError: If ``hypotheses`` and ``references`` differ in length.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    bleu = sacrebleu.corpus_bleu(list(hypotheses), [list(references)])
    chrf = sacrebleu.corpus_chrf(list(hypotheses), [list(references)], word_order=2)
    return TranslationScore(bleu=bleu.score, chrf=chrf.score)


@dataclass
class MultipleChoiceResult:
    """Result of :func:`evaluate_multiple_choice`."""

    accuracy: float
    num_correct: int
    num_total: int


@torch.no_grad()
def evaluate_multiple_choice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    questions: Sequence[dict],
    device: Optional[str] = None,
) -> MultipleChoiceResult:
    """Scores a multiple-choice question set (MMLU/domain-QA style) by log-likelihood.

    Each item in ``questions`` must be a dict with:
      - ``"question"``: str
      - ``"choices"``: list[str]
      - ``"answer_index"``: int, index into ``choices`` of the correct answer

    Follows the standard MMLU-style protocol: for each choice, the average
    per-token log-likelihood of the choice text conditioned on the question
    is computed by directly concatenating token ids (never re-tokenizing
    the merged string, which would let BPE merge across the question/choice
    boundary and silently shift token counts) and the highest-scoring
    choice is taken as the model's answer.

    Args:
        model: The causal LM to evaluate.
        tokenizer: Its matching tokenizer.
        questions: The question set, as described above.
        device: Device to run on; defaults to the model's own device.

    Returns:
        A :class:`MultipleChoiceResult` with overall accuracy.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    num_correct = 0
    for q in questions:
        prefix_ids = tokenizer(f"{q['question']}\n", return_tensors="pt").input_ids.to(device)
        prefix_len = prefix_ids.shape[1]

        scores = []
        for choice in q["choices"]:
            choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            input_ids = torch.cat([prefix_ids, choice_ids], dim=1)

            outputs = model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]

            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

            # Score only the tokens belonging to the choice, not the shared question prefix.
            choice_log_probs = token_log_probs[:, prefix_len - 1 :]
            scores.append(choice_log_probs.mean().item() if choice_log_probs.numel() else float("-inf"))

        predicted = max(range(len(scores)), key=lambda i: scores[i])
        if predicted == q["answer_index"]:
            num_correct += 1

    num_total = len(questions)
    return MultipleChoiceResult(
        accuracy=num_correct / num_total if num_total else 0.0,
        num_correct=num_correct,
        num_total=num_total,
    )
