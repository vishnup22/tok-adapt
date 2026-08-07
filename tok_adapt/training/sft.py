"""Supervised fine-tuning (SFT) on instruction / parallel-translation data.

Implements the SFT half of Phase 4: takes an already continued-pretrained
checkpoint and fine-tunes it on instruction-formatted or parallel
translation-pair examples using TRL's ``SFTTrainer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from tok_adapt.utils import ensure_pad_token


@dataclass
class SFTResult:
    """Outcome of a :meth:`SupervisedFineTuner.train` run."""

    output_dir: Path
    train_loss: float


class SupervisedFineTuner:
    """Wraps TRL's ``SFTTrainer`` for instruction / translation-pair fine-tuning.

    Args:
        model: The (continued-pretrained) causal LM to fine-tune.
        tokenizer: The matching tokenizer.
        use_lora: Whether to attach fresh LoRA adapters for the SFT stage.
        lora_r, lora_alpha, lora_dropout: Standard LoRA hyperparameters,
            used only when ``use_lora`` is True.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        self.model = model
        self.tokenizer = ensure_pad_token(tokenizer)
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
    def _load_examples(data_path: Union[str, Path]) -> Dataset:
        """Loads instruction examples from a JSONL file.

        Each line must be a JSON object matching one of:
          - ``{"text": "..."}`` -- used verbatim.
          - ``{"prompt": "...", "response": "..."}`` -- concatenated as
            ``prompt + response`` (e.g. instruction pairs or parallel
            translation pairs formatted as a single training string).

        Raises:
            ValueError: If the file has no usable records, or a record
                matches neither supported schema.
        """
        records = []
        for line in Path(data_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                records.append({"text": obj["text"]})
            elif "prompt" in obj and "response" in obj:
                records.append({"text": obj["prompt"] + obj["response"]})
            else:
                raise ValueError(f"Unsupported SFT record schema: {sorted(obj.keys())}")
        if not records:
            raise ValueError(f"No usable records found in {data_path}")
        return Dataset.from_list(records)

    def train(
        self,
        data_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_train_epochs: float = 1.0,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 1e-5,
        max_steps: int = -1,
        max_length: int = 512,
    ) -> SFTResult:
        """Fine-tunes on instruction/translation-pair data in ``data_path``.

        Args:
            data_path: JSONL file of instruction/translation examples (see
                :meth:`_load_examples` for accepted schemas).
            output_dir: Directory to save the fine-tuned model + tokenizer to.
            num_train_epochs: Number of passes over the data.
            per_device_train_batch_size: Batch size per device.
            learning_rate: Optimizer learning rate.
            max_steps: If >0, caps total optimizer steps (useful for
                smoke tests) regardless of ``num_train_epochs``.
            max_length: Maximum tokenized sequence length.

        Returns:
            An :class:`SFTResult` with the saved path and final train loss.
        """
        dataset = self._load_examples(data_path)
        out_dir = Path(output_dir)

        config = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            max_length=max_length,
            dataset_text_field="text",
            packing=False,
            save_strategy="no",
            report_to=[],
        )
        trainer = SFTTrainer(
            model=self.model,
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

        return SFTResult(output_dir=out_dir, train_loss=train_output.training_loss)
