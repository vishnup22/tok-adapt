"""Language-Adaptive Continued Pre-Training (LAPT / CPT).

Implements the CPT half of Phase 4 of the end-to-end pipeline: causal
language-model training on Corpus B, with the embedding matrices
(``embed_tokens`` / ``lm_head``) unfrozen alongside LoRA/QLoRA adapters
applied to the transformer backbone -- exactly the strategy needed once
:class:`tok_adapt.expansion.VocabularyExpander` has grown the vocabulary
and :class:`tok_adapt.initialization.EmbeddingInitializer` has seeded the
new rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from tok_adapt.utils import ensure_pad_token


@dataclass
class CPTResult:
    """Outcome of a :meth:`ContinuedPretrainer.train` run."""

    output_dir: Path
    train_loss: float
    trainable_params: int
    total_params: int


class ContinuedPretrainer:
    """Runs LoRA/QLoRA continued pre-training with unfrozen embeddings.

    Args:
        model: A causal LM, typically already vocabulary-expanded via
            :class:`tok_adapt.expansion.VocabularyExpander` and embedding-
            initialized via :class:`tok_adapt.initialization.EmbeddingInitializer`.
            May be loaded in 4-bit/8-bit (via ``BitsAndBytesConfig``) for
            QLoRA; this is auto-detected.
        tokenizer: The matching (expanded) tokenizer.
        use_lora: If True, wraps the backbone in LoRA adapters and freezes
            everything except the adapters plus the input/output
            embeddings. If False, the entire model is trained (full
            fine-tuning) -- only recommended for very small models.
        lora_r, lora_alpha, lora_dropout: Standard LoRA hyperparameters.
        block_size: Sequence length used when chunking the training corpus
            for causal LM training.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        block_size: int = 512,
    ) -> None:
        self.tokenizer = ensure_pad_token(tokenizer)
        self.block_size = block_size
        self.use_lora = use_lora

        is_kbit = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
        if is_kbit:
            model = prepare_model_for_kbit_training(model)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=None,  # architecture-specific auto-detection
            )
            model = get_peft_model(model, lora_config)
            self._unfreeze_embeddings(model)
        self.model = model

    @staticmethod
    def _unfreeze_embeddings(model) -> None:
        """Unfreezes embed_tokens/lm_head on top of frozen LoRA base weights.

        ``get_peft_model`` freezes every base-model parameter and leaves
        only the injected LoRA matrices trainable. The pipeline spec calls
        for the embedding matrices to *also* stay trainable, since that's
        where the newly added vocabulary rows live.
        """
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        input_embeddings = base_model.get_input_embeddings()
        output_embeddings = base_model.get_output_embeddings()
        if input_embeddings is not None:
            input_embeddings.weight.requires_grad_(True)
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != (
            input_embeddings.weight.data_ptr() if input_embeddings is not None else None
        ):
            output_embeddings.weight.requires_grad_(True)

    def _build_dataset(self, corpus_path: Union[str, Path]) -> Dataset:
        """Tokenizes and chunks a plain-text corpus into fixed-length CLM blocks."""
        text = Path(corpus_path).read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"No non-empty lines found in {corpus_path}")

        eos_id = self.tokenizer.eos_token_id or 0
        tokenized = self.tokenizer(lines, add_special_tokens=False)["input_ids"]
        concatenated: List[int] = []
        for ids in tokenized:
            concatenated.extend(ids)
            concatenated.append(eos_id)

        block_size = self.block_size
        n_blocks = len(concatenated) // block_size
        blocks = [concatenated[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
        if not blocks:
            # Corpus shorter than one block: use it as a single short block.
            blocks = [concatenated]

        return Dataset.from_dict({"input_ids": blocks})

    def train(
        self,
        corpus_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_train_epochs: float = 1.0,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 2e-4,
        max_steps: int = -1,
        logging_steps: int = 1,
        fp16: Optional[bool] = None,
    ) -> CPTResult:
        """Trains on ``corpus_path`` (Corpus B) and saves the result.

        Args:
            corpus_path: Path to the cleaned CPT corpus (Corpus B), one
                document/line per line.
            output_dir: Directory to save the trained model + tokenizer to.
            num_train_epochs: Number of passes over the corpus.
            per_device_train_batch_size: Batch size per device.
            learning_rate: Optimizer learning rate.
            max_steps: If >0, caps total optimizer steps regardless of
                ``num_train_epochs`` (useful for smoke tests).
            logging_steps: How often to log training loss.
            fp16: Whether to train in fp16. Defaults to True if CUDA is
                available, else False.

        Returns:
            A :class:`CPTResult` with the saved path, final train loss, and
            a trainable-vs-total parameter count (useful for confirming
            the freeze/unfreeze wiring took effect).
        """
        dataset = self._build_dataset(corpus_path)
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        if fp16 is None:
            fp16 = torch.cuda.is_available()

        out_dir = Path(output_dir)
        args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=logging_steps,
            fp16=fp16,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=dataset, data_collator=collator)
        train_output = trainer.train()

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_to_save = self.model.merge_and_unload() if hasattr(self.model, "merge_and_unload") else self.model
        except ValueError:
            # e.g. merging isn't supported for the current quantization state.
            model_to_save = self.model
        model_to_save.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        return CPTResult(
            output_dir=out_dir,
            train_loss=train_output.training_loss,
            trainable_params=trainable,
            total_params=total,
        )
