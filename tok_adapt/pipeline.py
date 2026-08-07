"""End-to-end pipeline orchestration: Phases 1-6 driven from one YAML config.

Chains :mod:`tok_adapt.dedup` (Phase 1), :mod:`tok_adapt.expansion` +
:mod:`tok_adapt.initialization` (Phases 2-3), :mod:`tok_adapt.training`
(Phase 4), :mod:`tok_adapt.evaluation` (Phase 5), and
:mod:`tok_adapt.export` (Phase 6) into a single run driven by a config
file, threading each enabled stage's output into the next stage's input.
Any stage can be disabled to start mid-pipeline from an existing
checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from tok_adapt.dedup import CorpusPreprocessor
from tok_adapt.evaluation.perplexity import compute_perplexity
from tok_adapt.expansion import VocabularyExpander
from tok_adapt.export.gguf_export import export_to_gguf
from tok_adapt.export.onnx_export import export_to_onnx
from tok_adapt.initialization import EmbeddingInitializer
from tok_adapt.training.cpt import ContinuedPretrainer
from tok_adapt.training.dpo import PreferenceAligner
from tok_adapt.training.sft import SupervisedFineTuner
from tok_adapt.utils import load_model, load_model_and_tokenizer, load_tokenizer, save_model_and_tokenizer


class DedupStageConfig(BaseModel):
    enabled: bool = False
    input_paths: List[str] = Field(default_factory=list)
    languages: Optional[List[str]] = None
    corpus_a_max_bytes: int = 30_000_000


class ExpandStageConfig(BaseModel):
    enabled: bool = False
    corpus_path: Optional[str] = None  # defaults to dedup's corpus_a.txt if omitted
    add_vocab_size: int = 8000
    algorithm: str = "bpe"


class CPTStageConfig(BaseModel):
    enabled: bool = False
    corpus_path: Optional[str] = None  # defaults to dedup's corpus_b.txt if omitted
    use_lora: bool = True
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4


class SFTStageConfig(BaseModel):
    enabled: bool = False
    data_path: Optional[str] = None
    use_lora: bool = True
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-5


class DPOStageConfig(BaseModel):
    enabled: bool = False
    data_path: Optional[str] = None
    beta: float = 0.1
    use_lora: bool = True
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    learning_rate: float = 5e-7


class EvaluateStageConfig(BaseModel):
    enabled: bool = False
    perplexity_text_file: Optional[str] = None


class ExportStageConfig(BaseModel):
    enabled: bool = False
    gguf: bool = True
    gguf_outtype: str = "f16"
    # See tok_adapt.export.gguf_export.export_to_gguf: required for
    # checkpoints that went through the expand stage, since llama.cpp
    # can't auto-detect an expanded tokenizer's pre-tokenizer type.
    gguf_pre_tokenizer_hint: Optional[str] = None
    onnx: bool = True
    onnx_task: str = "text-generation-with-past"


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration, as loaded from a YAML file."""

    model: str
    output_root: str
    dedup: DedupStageConfig = Field(default_factory=DedupStageConfig)
    expand: ExpandStageConfig = Field(default_factory=ExpandStageConfig)
    cpt: CPTStageConfig = Field(default_factory=CPTStageConfig)
    sft: SFTStageConfig = Field(default_factory=SFTStageConfig)
    dpo: DPOStageConfig = Field(default_factory=DPOStageConfig)
    evaluate: EvaluateStageConfig = Field(default_factory=EvaluateStageConfig)
    export: ExportStageConfig = Field(default_factory=ExportStageConfig)


def load_pipeline_config(path: Union[str, Path]) -> PipelineConfig:
    """Loads and validates a pipeline YAML config file.

    Args:
        path: Path to the YAML config.

    Returns:
        A validated :class:`PipelineConfig`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig(**raw)


def run_pipeline(config: PipelineConfig) -> dict:
    """Runs every enabled stage of ``config`` in order, writing a summary.

    Each stage's output directory becomes the next enabled stage's input
    checkpoint, so any prefix of stages can be disabled to resume from an
    existing checkpoint (point ``model`` at that checkpoint and disable
    the stages that already produced it).

    Args:
        config: A validated :class:`PipelineConfig`.

    Returns:
        A summary dict of what ran and where its output landed, also
        written to ``<output_root>/summary.json``.
    """
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {"stages": {}}

    current_model_path = config.model
    corpus_a_path: Optional[str] = None
    corpus_b_path: Optional[str] = None

    # --- Phase 1: dedup ---------------------------------------------------
    if config.dedup.enabled:
        stage_dir = output_root / "01_dedup"
        preprocessor = CorpusPreprocessor(languages=config.dedup.languages)
        stats = preprocessor.process(
            config.dedup.input_paths, stage_dir, corpus_a_max_bytes=config.dedup.corpus_a_max_bytes
        )
        corpus_a_path = str(stage_dir / "corpus_a.txt")
        corpus_b_path = str(stage_dir / "corpus_b.txt")
        summary["stages"]["dedup"] = {
            "output_dir": str(stage_dir),
            "kept_lines": stats.kept_lines,
            "dropped_duplicate": stats.dropped_duplicate,
            "dropped_language": stats.dropped_language,
        }

    # --- Phases 2-3: expand + align embeddings -----------------------------
    if config.expand.enabled:
        stage_dir = output_root / "02_expanded"
        corpus_for_expand = config.expand.corpus_path or corpus_a_path
        if corpus_for_expand is None:
            raise ValueError("expand stage needs corpus_path, or an enabled dedup stage to supply corpus_a.txt")

        base_model, base_tokenizer = load_model_and_tokenizer(current_model_path)
        expander = VocabularyExpander(base_tokenizer)
        sub_tok_path = expander.train_sub_tokenizer(
            corpus_for_expand, config.expand.add_vocab_size, algorithm=config.expand.algorithm
        )
        extended_tokenizer = expander.merge_vocabularies(str(sub_tok_path))

        initializer = EmbeddingInitializer(base_model, base_tokenizer)
        expanded_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

        save_model_and_tokenizer(expanded_model, extended_tokenizer, stage_dir)
        current_model_path = str(stage_dir)
        summary["stages"]["expand"] = {
            "output_dir": str(stage_dir),
            "base_vocab_size": len(base_tokenizer),
            "expanded_vocab_size": len(extended_tokenizer),
        }

    # --- Phase 4a: continued pre-training -----------------------------------
    if config.cpt.enabled:
        stage_dir = output_root / "03_cpt"
        corpus_for_cpt = config.cpt.corpus_path or corpus_b_path
        if corpus_for_cpt is None:
            raise ValueError("cpt stage needs corpus_path, or an enabled dedup stage to supply corpus_b.txt")

        model, tokenizer = load_model_and_tokenizer(current_model_path)
        pretrainer = ContinuedPretrainer(model, tokenizer, use_lora=config.cpt.use_lora)
        result = pretrainer.train(
            corpus_for_cpt,
            stage_dir,
            num_train_epochs=config.cpt.num_train_epochs,
            max_steps=config.cpt.max_steps,
            per_device_train_batch_size=config.cpt.per_device_train_batch_size,
            learning_rate=config.cpt.learning_rate,
        )
        current_model_path = str(result.output_dir)
        summary["stages"]["cpt"] = {
            "output_dir": str(result.output_dir),
            "train_loss": result.train_loss,
            "trainable_params": result.trainable_params,
            "total_params": result.total_params,
        }

    # --- Phase 4b: supervised fine-tuning -----------------------------------
    if config.sft.enabled:
        if not config.sft.data_path:
            raise ValueError("sft stage requires data_path")
        stage_dir = output_root / "04_sft"
        model, tokenizer = load_model_and_tokenizer(current_model_path)
        tuner = SupervisedFineTuner(model, tokenizer, use_lora=config.sft.use_lora)
        result = tuner.train(
            config.sft.data_path,
            stage_dir,
            num_train_epochs=config.sft.num_train_epochs,
            max_steps=config.sft.max_steps,
            per_device_train_batch_size=config.sft.per_device_train_batch_size,
            learning_rate=config.sft.learning_rate,
        )
        current_model_path = str(result.output_dir)
        summary["stages"]["sft"] = {"output_dir": str(result.output_dir), "train_loss": result.train_loss}

    # --- Phase 4c: DPO alignment --------------------------------------------
    if config.dpo.enabled:
        if not config.dpo.data_path:
            raise ValueError("dpo stage requires data_path")
        stage_dir = output_root / "05_dpo"
        model, tokenizer = load_model_and_tokenizer(current_model_path)
        aligner = PreferenceAligner(model, tokenizer, beta=config.dpo.beta, use_lora=config.dpo.use_lora)
        result = aligner.train(
            config.dpo.data_path,
            stage_dir,
            num_train_epochs=config.dpo.num_train_epochs,
            max_steps=config.dpo.max_steps,
            per_device_train_batch_size=config.dpo.per_device_train_batch_size,
            learning_rate=config.dpo.learning_rate,
        )
        current_model_path = str(result.output_dir)
        summary["stages"]["dpo"] = {"output_dir": str(result.output_dir), "train_loss": result.train_loss}

    # --- Phase 5: evaluation -------------------------------------------------
    if config.evaluate.enabled:
        stage_dir = output_root / "06_eval"
        stage_dir.mkdir(parents=True, exist_ok=True)
        eval_summary: dict = {}
        if config.evaluate.perplexity_text_file:
            lines = [
                ln.strip()
                for ln in Path(config.evaluate.perplexity_text_file).read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            model, tokenizer = load_model_and_tokenizer(current_model_path)
            result = compute_perplexity(model, tokenizer, lines)
            eval_summary["perplexity"] = {
                "loss": result.loss,
                "perplexity": result.perplexity,
                "num_tokens": result.num_tokens,
            }
        (stage_dir / "report.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        summary["stages"]["evaluate"] = {"output_dir": str(stage_dir), **eval_summary}

    # --- Phase 6: export -------------------------------------------------------
    if config.export.enabled:
        stage_dir = output_root / "07_export"
        stage_dir.mkdir(parents=True, exist_ok=True)
        export_summary: dict = {}
        if config.export.gguf:
            gguf_path = export_to_gguf(
                current_model_path,
                stage_dir / "model.gguf",
                outtype=config.export.gguf_outtype,
                pre_tokenizer_hint=config.export.gguf_pre_tokenizer_hint,
            )
            export_summary["gguf"] = str(gguf_path)
        if config.export.onnx:
            onnx_dir = export_to_onnx(current_model_path, stage_dir / "onnx", task=config.export.onnx_task)
            export_summary["onnx"] = str(onnx_dir)
        summary["stages"]["export"] = {"output_dir": str(stage_dir), **export_summary}

    summary["final_model_path"] = current_model_path
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
