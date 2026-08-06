"""Unified command line interface for tok_adapt."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from tok_adapt.expansion import VocabularyExpander
from tok_adapt.initialization import EmbeddingInitializer
from tok_adapt.metrics import FertilityEvaluator
from tok_adapt.pruning import VocabularyPruner
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


if __name__ == "__main__":
    cli()
