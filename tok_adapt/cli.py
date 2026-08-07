"""Unified command line interface for tok_adapt."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from tok_adapt.dedup import CorpusPreprocessor
from tok_adapt.evaluation.downstream import evaluate_multiple_choice, score_translations
from tok_adapt.evaluation.perplexity import compute_perplexity
from tok_adapt.expansion import VocabularyExpander
from tok_adapt.export.gguf_export import export_to_gguf
from tok_adapt.export.onnx_export import export_to_onnx
from tok_adapt.export.vllm_serve import serve_with_vllm
from tok_adapt.initialization import EmbeddingInitializer
from tok_adapt.metrics import FertilityEvaluator
from tok_adapt.pipeline import load_pipeline_config, run_pipeline
from tok_adapt.pruning import VocabularyPruner
from tok_adapt.training.cpt import ContinuedPretrainer
from tok_adapt.training.dpo import PreferenceAligner
from tok_adapt.training.sft import SupervisedFineTuner
from tok_adapt.utils import (
    load_model,
    load_model_and_tokenizer,
    load_tokenizer,
    save_model_and_tokenizer,
)

console = Console()


@click.group()
@click.version_option(package_name="tok-adapt")
def cli() -> None:
    """tok-adapt: adapt, extend, prune, and initialize HF tokenizers & embeddings."""


@cli.command()
@click.option("--model", required=True, help="Path or Hugging Face Hub id of the base model.")
@click.option("--corpus", required=True, help="Path to the target-domain/language corpus (text file or directory).")
@click.option("--add-vocab-size", required=True, type=int, help="Vocabulary size of the sub-tokenizer to train.")
@click.option("--algorithm", default="bpe", type=click.Choice(["bpe", "unigram"]), show_default=True)
@click.option("--output", required=True, help="Output directory for the expanded model + tokenizer.")
def expand(model: str, corpus: str, add_vocab_size: int, algorithm: str, output: str) -> None:
    """Train a sub-tokenizer on CORPUS and merge new tokens into MODEL's tokenizer."""
    with console.status("[bold green]Loading base model and tokenizer..."):
        base_model, base_tokenizer = load_model_and_tokenizer(model)
    console.log(f"Base vocab size: {len(base_tokenizer)}")

    expander = VocabularyExpander(base_tokenizer)
    with console.status(f"[bold green]Training {algorithm} sub-tokenizer (vocab_size={add_vocab_size})..."):
        sub_tok_path = expander.train_sub_tokenizer(corpus, add_vocab_size, algorithm=algorithm)
    console.log(f"Sub-tokenizer saved to {sub_tok_path}")

    with console.status("[bold green]Merging vocabularies..."):
        extended_tokenizer = expander.merge_vocabularies(str(sub_tok_path))
    num_added = len(extended_tokenizer) - len(base_tokenizer)
    console.log(f"Added {num_added} new tokens (new vocab size: {len(extended_tokenizer)})")

    with console.status("[bold green]Aligning embeddings (subword_mean)..."):
        initializer = EmbeddingInitializer(base_model, base_tokenizer)
        expanded_model = initializer.smart_resize_embeddings(extended_tokenizer, strategy="subword_mean")

    out_dir = save_model_and_tokenizer(expanded_model, extended_tokenizer, output)
    console.print(f"[bold cyan]Done.[/bold cyan] Expanded model + tokenizer saved to {out_dir}")


@cli.command(name="align-embeddings")
@click.option("--model", required=True, help="Path or Hugging Face Hub id of the base model (its original tokenizer is loaded alongside it).")
@click.option("--tokenizer", "tokenizer_path", required=True, help="Path to the already-expanded tokenizer.")
@click.option("--strategy", default="subword_mean", type=click.Choice(["subword_mean"]), show_default=True)
@click.option("--output", required=True, help="Output directory for the resized model + tokenizer.")
def align_embeddings(model: str, tokenizer_path: str, strategy: str, output: str) -> None:
    """Resize MODEL's embeddings to match an already-expanded TOKENIZER."""
    with console.status("[bold green]Loading model and original tokenizer..."):
        base_model = load_model(model)
        original_tokenizer = load_tokenizer(model)
    with console.status("[bold green]Loading expanded tokenizer..."):
        new_tokenizer = load_tokenizer(tokenizer_path)

    console.log(f"Original vocab size: {len(original_tokenizer)} -> New vocab size: {len(new_tokenizer)}")

    initializer = EmbeddingInitializer(base_model, original_tokenizer)
    with console.status(f"[bold green]Resizing + initializing embeddings ({strategy})..."):
        resized_model = initializer.smart_resize_embeddings(new_tokenizer, strategy=strategy)

    out_dir = save_model_and_tokenizer(resized_model, new_tokenizer, output)
    console.print(f"[bold cyan]Done.[/bold cyan] Resized model + tokenizer saved to {out_dir}")


@cli.command()
@click.option("--model", required=True, help="Path or Hugging Face Hub id of the model to prune.")
@click.option("--corpus", required=True, help="Reference corpus used to determine token usage frequency.")
@click.option("--min-frequency", default=1, type=int, show_default=True)
@click.option("--keep-special-tokens/--no-keep-special-tokens", default=True, show_default=True)
@click.option("--output", required=True, help="Output directory for the pruned model + tokenizer.")
def prune(model: str, corpus: str, min_frequency: int, keep_special_tokens: bool, output: str) -> None:
    """Prune unused vocabulary from MODEL based on token frequency in CORPUS."""
    with console.status("[bold green]Loading model and tokenizer..."):
        base_model, base_tokenizer = load_model_and_tokenizer(model)
    console.log(f"Original vocab size: {len(base_tokenizer)}")

    pruner = VocabularyPruner(base_model, base_tokenizer)
    with console.status("[bold green]Computing token frequencies and pruning..."):
        pruned_model, pruned_tokenizer = pruner.prune_unused_tokens(
            corpus, min_frequency=min_frequency, keep_special_tokens=keep_special_tokens
        )
    console.log(f"Pruned vocab size: {len(pruned_tokenizer)}")

    out_dir = save_model_and_tokenizer(pruned_model, pruned_tokenizer, output)
    console.print(f"[bold cyan]Done.[/bold cyan] Pruned model + tokenizer saved to {out_dir}")


@cli.command()
@click.option("--base-tokenizer", required=True, help="Path or Hugging Face Hub id of the base tokenizer.")
@click.option("--adapted-tokenizer", required=True, help="Path or Hugging Face Hub id of the adapted/extended tokenizer.")
@click.option("--text-file", required=True, help="Path to a text file, evaluated line-by-line.")
def benchmark(base_tokenizer: str, adapted_tokenizer: str, text_file: str) -> None:
    """Compare fertility and sequence compression between two tokenizers."""
    with console.status("[bold green]Loading tokenizers..."):
        base_tok = load_tokenizer(base_tokenizer)
        adapted_tok = load_tokenizer(adapted_tokenizer)

    lines = [
        line.strip()
        for line in Path(text_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise click.ClickException(f"No non-empty lines found in {text_file}")

    base_eval = FertilityEvaluator(base_tok)
    adapted_eval = FertilityEvaluator(adapted_tok)

    base_stats = base_eval.compute_fertility(lines)
    adapted_stats = adapted_eval.compute_fertility(lines)
    compression = base_eval.compare_sequence_compression(adapted_tok, lines)

    table = Table(title="tok-adapt benchmark")
    table.add_column("Metric")
    table.add_column("Base", justify="right")
    table.add_column("Adapted", justify="right")

    table.add_row("Vocab size", str(len(base_tok)), str(len(adapted_tok)))
    table.add_row("Total tokens", str(base_stats["total_tokens"]), str(adapted_stats["total_tokens"]))
    table.add_row(
        "Token/word ratio",
        f"{base_stats['token_to_word_ratio']:.4f}",
        f"{adapted_stats['token_to_word_ratio']:.4f}",
    )
    table.add_row(
        "Token/byte ratio",
        f"{base_stats['token_to_byte_ratio']:.4f}",
        f"{adapted_stats['token_to_byte_ratio']:.4f}",
    )
    table.add_row("Compression ratio (adapted/base)", "-", f"{compression['compression_ratio']:.4f}")
    table.add_row("Token reduction", "-", f"{compression['token_reduction_pct']:.2f}%")

    console.print(table)


@cli.command()
@click.option(
    "--input",
    "input_paths",
    required=True,
    multiple=True,
    help="Raw text file(s)/directory(ies) to ingest. Repeatable.",
)
@click.option("--output", required=True, help="Output directory for corpus_a.txt / corpus_b.txt.")
@click.option(
    "--languages",
    default=None,
    help="Comma-separated ISO 639-1 codes to keep (e.g. hi,te). Omit to keep all languages.",
)
@click.option("--corpus-a-max-bytes", default=30_000_000, type=int, show_default=True)
def dedup(input_paths: tuple, output: str, languages: str, corpus_a_max_bytes: int) -> None:
    """Phase 1: deduplicate, language-filter, and split raw corpora into Corpus A / Corpus B."""
    langs = languages.split(",") if languages else None
    preprocessor = CorpusPreprocessor(languages=langs)
    with console.status("[bold green]Deduplicating and filtering corpus..."):
        stats = preprocessor.process(list(input_paths), output, corpus_a_max_bytes=corpus_a_max_bytes)

    table = Table(title="tok-adapt dedup")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Raw lines", str(stats.raw_lines))
    table.add_row("Dropped (too short)", str(stats.dropped_empty))
    table.add_row("Dropped (language)", str(stats.dropped_language))
    table.add_row("Dropped (duplicate)", str(stats.dropped_duplicate))
    table.add_row("Kept lines (Corpus B)", str(stats.kept_lines))
    table.add_row("Corpus A lines", str(stats.corpus_a_lines))
    table.add_row("Corpus A bytes", str(stats.corpus_a_bytes))
    console.print(table)
    console.print(f"[bold cyan]Done.[/bold cyan] corpus_a.txt / corpus_b.txt written to {output}")


@cli.command(name="train-cpt")
@click.option(
    "--model", required=True, help="Path or Hub id of the (typically vocabulary-expanded) model to continue pretraining."
)
@click.option("--corpus", required=True, help="Path to Corpus B (cleaned CPT corpus).")
@click.option("--output", required=True, help="Output directory for the trained model + tokenizer.")
@click.option("--use-lora/--no-use-lora", default=True, show_default=True)
@click.option("--num-train-epochs", default=1.0, type=float, show_default=True)
@click.option("--max-steps", default=-1, type=int, show_default=True, help="Caps optimizer steps; -1 runs full epochs.")
@click.option("--per-device-train-batch-size", default=2, type=int, show_default=True)
@click.option("--learning-rate", default=2e-4, type=float, show_default=True)
def train_cpt(
    model: str,
    corpus: str,
    output: str,
    use_lora: bool,
    num_train_epochs: float,
    max_steps: int,
    per_device_train_batch_size: int,
    learning_rate: float,
) -> None:
    """Phase 4a: continued pre-training (CPT) with LoRA/QLoRA + unfrozen embeddings."""
    with console.status("[bold green]Loading model and tokenizer..."):
        base_model, tokenizer = load_model_and_tokenizer(model)
    pretrainer = ContinuedPretrainer(base_model, tokenizer, use_lora=use_lora)
    with console.status("[bold green]Running continued pre-training..."):
        result = pretrainer.train(
            corpus,
            output,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
        )
    console.log(f"Train loss: {result.train_loss:.4f}")
    console.log(f"Trainable params: {result.trainable_params:,} / {result.total_params:,}")
    console.print(f"[bold cyan]Done.[/bold cyan] CPT model saved to {result.output_dir}")


@cli.command(name="train-sft")
@click.option("--model", required=True, help="Path or Hub id of the (typically CPT'd) model to fine-tune.")
@click.option("--data", "data_path", required=True, help="JSONL file of {text} or {prompt, response} records.")
@click.option("--output", required=True, help="Output directory for the fine-tuned model + tokenizer.")
@click.option("--use-lora/--no-use-lora", default=True, show_default=True)
@click.option("--num-train-epochs", default=1.0, type=float, show_default=True)
@click.option("--max-steps", default=-1, type=int, show_default=True)
@click.option("--per-device-train-batch-size", default=2, type=int, show_default=True)
@click.option("--learning-rate", default=1e-5, type=float, show_default=True)
def train_sft(
    model: str,
    data_path: str,
    output: str,
    use_lora: bool,
    num_train_epochs: float,
    max_steps: int,
    per_device_train_batch_size: int,
    learning_rate: float,
) -> None:
    """Phase 4b: supervised fine-tuning (SFT) on instruction/translation-pair data."""
    with console.status("[bold green]Loading model and tokenizer..."):
        base_model, tokenizer = load_model_and_tokenizer(model)
    tuner = SupervisedFineTuner(base_model, tokenizer, use_lora=use_lora)
    with console.status("[bold green]Running supervised fine-tuning..."):
        result = tuner.train(
            data_path,
            output,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
        )
    console.log(f"Train loss: {result.train_loss:.4f}")
    console.print(f"[bold cyan]Done.[/bold cyan] SFT model saved to {result.output_dir}")


@cli.command(name="train-dpo")
@click.option("--model", required=True, help="Path or Hub id of the (typically SFT'd) model to align.")
@click.option("--data", "data_path", required=True, help="JSONL file of {prompt, chosen, rejected} records.")
@click.option("--output", required=True, help="Output directory for the aligned model + tokenizer.")
@click.option("--beta", default=0.1, type=float, show_default=True)
@click.option("--use-lora/--no-use-lora", default=True, show_default=True)
@click.option("--num-train-epochs", default=1.0, type=float, show_default=True)
@click.option("--max-steps", default=-1, type=int, show_default=True)
@click.option("--per-device-train-batch-size", default=1, type=int, show_default=True)
@click.option("--learning-rate", default=5e-7, type=float, show_default=True)
def train_dpo(
    model: str,
    data_path: str,
    output: str,
    beta: float,
    use_lora: bool,
    num_train_epochs: float,
    max_steps: int,
    per_device_train_batch_size: int,
    learning_rate: float,
) -> None:
    """Phase 4c: DPO preference alignment."""
    with console.status("[bold green]Loading model and tokenizer..."):
        base_model, tokenizer = load_model_and_tokenizer(model)
    aligner = PreferenceAligner(base_model, tokenizer, beta=beta, use_lora=use_lora)
    with console.status("[bold green]Running DPO alignment..."):
        result = aligner.train(
            data_path,
            output,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
        )
    console.log(f"Train loss: {result.train_loss:.4f}")
    console.print(f"[bold cyan]Done.[/bold cyan] DPO-aligned model saved to {result.output_dir}")


@cli.command(name="evaluate-perplexity")
@click.option("--model", required=True, help="Path or Hub id of the model to evaluate.")
@click.option("--text-file", required=True, help="Held-out text file, evaluated line-by-line.")
@click.option("--base-model", default=None, help="Optional base model to compare against.")
def evaluate_perplexity(model: str, text_file: str, base_model: str) -> None:
    """Phase 5: held-out perplexity, optionally compared against a base model."""
    lines = [ln.strip() for ln in Path(text_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise click.ClickException(f"No non-empty lines found in {text_file}")

    with console.status("[bold green]Loading model..."):
        eval_model, eval_tokenizer = load_model_and_tokenizer(model)
    with console.status("[bold green]Computing perplexity..."):
        result = compute_perplexity(eval_model, eval_tokenizer, lines)

    table = Table(title="tok-adapt evaluate-perplexity")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Loss", f"{result.loss:.4f}")
    table.add_row("Perplexity", f"{result.perplexity:.4f}")
    table.add_row("Tokens evaluated", str(result.num_tokens))

    if base_model:
        with console.status("[bold green]Loading base model for comparison..."):
            base_m, base_t = load_model_and_tokenizer(base_model)
        with console.status("[bold green]Computing base perplexity..."):
            base_result = compute_perplexity(base_m, base_t, lines)
        table.add_row("Base perplexity", f"{base_result.perplexity:.4f}")
        table.add_row("Delta (adapted - base)", f"{result.perplexity - base_result.perplexity:.4f}")

    console.print(table)


@cli.command(name="evaluate-downstream")
@click.option("--model", required=True, help="Path or Hub id of the model to evaluate.")
@click.option("--translations", default=None, help="JSONL file of {hypothesis, reference} pairs for BLEU/chrF++.")
@click.option("--mcq", default=None, help="JSONL file of {question, choices, answer_index} for multiple-choice accuracy.")
def evaluate_downstream(model: str, translations: str, mcq: str) -> None:
    """Phase 5: downstream accuracy -- BLEU/chrF++ and/or multiple-choice accuracy."""
    if not translations and not mcq:
        raise click.ClickException("Provide at least one of --translations or --mcq.")

    table = Table(title="tok-adapt evaluate-downstream")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    if translations:
        records = [
            json.loads(line) for line in Path(translations).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        score = score_translations([r["hypothesis"] for r in records], [r["reference"] for r in records])
        table.add_row("BLEU", f"{score.bleu:.2f}")
        table.add_row("chrF++", f"{score.chrf:.2f}")

    if mcq:
        questions = [json.loads(line) for line in Path(mcq).read_text(encoding="utf-8").splitlines() if line.strip()]
        with console.status("[bold green]Loading model..."):
            eval_model, eval_tokenizer = load_model_and_tokenizer(model)
        with console.status("[bold green]Scoring multiple-choice questions..."):
            result = evaluate_multiple_choice(eval_model, eval_tokenizer, questions)
        table.add_row("MCQ accuracy", f"{result.accuracy:.2%} ({result.num_correct}/{result.num_total})")

    console.print(table)


@cli.command(name="export-gguf")
@click.option("--model", required=True, help="Path to a save_pretrained-style checkpoint directory.")
@click.option("--output", required=True, help="Destination .gguf file path.")
@click.option("--outtype", default="f16", type=click.Choice(["f32", "f16", "bf16", "q8_0"]), show_default=True)
@click.option("--llama-cpp-dir", default=None, help="Path to an existing llama.cpp checkout (skips auto-fetch).")
@click.option(
    "--pre-tokenizer-hint",
    default=None,
    help=(
        "Base tokenizer pre-tokenizer id (e.g. 'gpt2', 'llama-bpe') to force. "
        "Required for tokenizers produced by `tok-adapt expand`: llama.cpp "
        "auto-detects pre-tokenizers by hashing the vocabulary, which an "
        "expanded vocabulary always fails to match."
    ),
)
def export_gguf(model: str, output: str, outtype: str, llama_cpp_dir: str, pre_tokenizer_hint: str) -> None:
    """Phase 6: convert a checkpoint to GGUF for llama.cpp-based edge deployment."""
    with console.status("[bold green]Converting to GGUF (fetching llama.cpp tooling on first use)..."):
        result = export_to_gguf(
            model, output, outtype=outtype, llama_cpp_dir=llama_cpp_dir, pre_tokenizer_hint=pre_tokenizer_hint
        )
    console.print(f"[bold cyan]Done.[/bold cyan] GGUF model written to {result}")


@cli.command(name="export-onnx")
@click.option("--model", required=True, help="Path or Hub id of the checkpoint to export.")
@click.option("--output", required=True, help="Output directory for the ONNX graph.")
@click.option("--task", default="text-generation-with-past", show_default=True)
def export_onnx(model: str, output: str, task: str) -> None:
    """Phase 6: export a checkpoint to ONNX for portable/accelerated inference."""
    with console.status("[bold green]Exporting to ONNX..."):
        result = export_to_onnx(model, output, task=task)
    console.print(f"[bold cyan]Done.[/bold cyan] ONNX graph written to {result}")


@cli.command(name="serve-vllm")
@click.option("--model", required=True, help="Path or Hub id of the checkpoint to serve.")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def serve_vllm_cmd(model: str, host: str, port: int) -> None:
    """Phase 6: launch an OpenAI-compatible vLLM server (Linux/WSL only -- see tok_adapt.export.vllm_serve)."""
    process = serve_with_vllm(model, host=host, port=port)
    console.print(f"[bold cyan]vLLM server started[/bold cyan] (pid {process.pid}) on {host}:{port}. Press Ctrl+C to stop.")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()


@cli.command()
@click.option("--config", "config_path", required=True, help="Path to a pipeline YAML config.")
def pipeline(config_path: str) -> None:
    """Run the full pipeline (dedup -> expand -> CPT -> SFT -> DPO -> eval -> export) from a YAML config."""
    cfg = load_pipeline_config(config_path)
    with console.status("[bold green]Running pipeline..."):
        summary = run_pipeline(cfg)
    console.print_json(json.dumps(summary))
    console.print(f"[bold cyan]Done.[/bold cyan] Final model at {summary['final_model_path']}")


if __name__ == "__main__":
    cli()
