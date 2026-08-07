"""Preference alignment via Direct Preference Optimization (DPO/ORPO).

Implements the alignment half of Phase 4: trains on ``(prompt, chosen,
rejected)`` preference triples to align output style/safety without a
separate reward model, using TRL's ``DPOTrainer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer

from tok_adapt.utils import ensure_pad_token


@dataclass
class DPOResult:
    """Outcome of a :meth:`PreferenceAligner.train` run."""

    output_dir: Path
    train_loss: float


class PreferenceAligner:
    """Wraps TRL's ``DPOTrainer`` for preference alignment.

    Args:
        model: The SFT checkpoint to align.
        tokenizer: The matching tokenizer.
        beta: DPO temperature; lower values allow larger deviation from
            the reference (pre-DPO) policy.
        use_lora: Whether to attach fresh LoRA adapters for this stage.
            When True and no explicit reference model is supplied to
            :meth:`train`, TRL derives the reference policy by disabling
            the adapters rather than holding a second full copy of the
            model in memory -- important for staying within a 6-8GB GPU.
        lora_r, lora_alpha, lora_dropout: Standard LoRA hyperparameters,
            used only when ``use_lora`` is True.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        beta: float = 0.1,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        self.model = model
        self.tokenizer = ensure_pad_token(tokenizer)
        self.beta = beta
        self.peft_config: Optional[LoraConfig] = (
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=None,
            )
            if use_lora
            else None
        )

    @staticmethod
    def _load_preferences(data_path: Union[str, Path]) -> Dataset:
        """Loads preference triples from a JSONL file.

        Each line must be a JSON object with ``prompt``, ``chosen``, and
        ``rejected`` string fields.

        Raises:
            ValueError: If the file has no usable records, or a record is
                missing a required field.
        """
        records = []
        for line in Path(data_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = {"prompt", "chosen", "rejected"} - obj.keys()
            if missing:
                raise ValueError(f"Preference record missing fields {missing}: {obj}")
            records.append({"prompt": obj["prompt"], "chosen": obj["chosen"], "rejected": obj["rejected"]})
        if not records:
            raise ValueError(f"No usable preference records found in {data_path}")
        return Dataset.from_list(records)

    def train(
        self,
        data_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_train_epochs: float = 1.0,
        per_device_train_batch_size: int = 1,
        learning_rate: float = 5e-7,
        max_steps: int = -1,
        max_length: int = 512,
        fp16: Optional[bool] = None,
    ) -> DPOResult:
        """Aligns the model on preference triples in ``data_path``.

        Args:
            data_path: JSONL file of ``{prompt, chosen, rejected}`` records.
            output_dir: Directory to save the aligned model + tokenizer to.
            num_train_epochs: Number of passes over the data.
            per_device_train_batch_size: Batch size per device (kept small
                by default since DPO holds two forward passes per example).
            learning_rate: Optimizer learning rate. DPO typically needs a
                much smaller LR than SFT/CPT.
            max_steps: If >0, caps total optimizer steps (useful for
                smoke tests) regardless of ``num_train_epochs``.
            max_length: Maximum tokenized sequence length (prompt+response).
            fp16: Whether to train in fp16. Defaults to True if CUDA is
                available, else False. ``bf16`` is always left off: TRL's
                ``DPOConfig`` defaults it to True unconditionally, which
                raises on CPU-only/non-Ampere hardware ("Your setup
                doesn't support bf16/gpu") -- explicit fp16 is the more
                broadly portable choice.

        Returns:
            A :class:`DPOResult` with the saved path and final train loss.
        """
        dataset = self._load_preferences(data_path)
        out_dir = Path(output_dir)

        if fp16 is None:
            fp16 = torch.cuda.is_available()

        config = DPOConfig(
            output_dir=str(out_dir),
            beta=self.beta,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            max_length=max_length,
            save_strategy="no",
            report_to=[],
            bf16=False,
            fp16=fp16,
        )
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # derived from the base policy w/ adapters disabled
            args=config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
        )
        train_output = trainer.train()

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_to_save = (
                trainer.model.merge_and_unload() if hasattr(trainer.model, "merge_and_unload") else trainer.model
            )
        except ValueError:
            model_to_save = trainer.model
        model_to_save.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        return DPOResult(output_dir=out_dir, train_loss=train_output.training_loss)
