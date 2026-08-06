"""Vocabulary trimming and model embedding-matrix shrinking."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tokenizers import Tokenizer as BackendTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerFast


class VocabularyPruner:
    """Prunes unused vocabulary and shrinks a model's embedding matrices to match.

    Args:
        model: The model whose ``embed_tokens`` / ``lm_head`` matrices will
            be sliced down to the retained vocabulary.
        tokenizer: The tokenizer to prune. Must be a fast tokenizer backed
            by the ``tokenizers`` Rust library.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def prune_unused_tokens(
        self,
        corpus_path: str,
        min_frequency: int = 1,
        keep_special_tokens: bool = True,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        """Prunes tokens unseen (or rare) in a reference corpus.

        Builds a frequency map of token ids over ``corpus_path``, retains ids
        that appear at least ``min_frequency`` times (plus all special
        tokens if requested), and shrinks both the tokenizer and the model's
        embedding matrices to that dense retained set.

        Args:
            corpus_path: Path to a text file, tokenized line-by-line to
                build the frequency map.
            min_frequency: Minimum occurrence count for a token id to be
                retained.
            keep_special_tokens: If True, always retain special tokens
                (bos, eos, pad, unk, mask, and other control tokens)
                regardless of their observed frequency.

        Returns:
            A tuple of ``(pruned_model, pruned_tokenizer)``.

        Raises:
            FileNotFoundError: If ``corpus_path`` does not exist.
            ValueError: If no tokens would be retained.
        """
        corpus = Path(corpus_path)
        if not corpus.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        freq: Counter = Counter()
        with corpus.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = self.tokenizer.encode(line, add_special_tokens=False)
                freq.update(ids)

        special_ids = set(self.tokenizer.all_special_ids) if keep_special_tokens else set()
        retained = {tid for tid, count in freq.items() if count >= min_frequency}
        retained |= special_ids

        if not retained:
            raise ValueError(
                "No tokens would be retained after pruning; check corpus_path "
                "and min_frequency."
            )

        retained_ids = sorted(retained)
        old_id_to_new_id: Dict[int, int] = {
            old_id: new_id for new_id, old_id in enumerate(retained_ids)
        }
        self.old_id_to_new_id = old_id_to_new_id  # exposed for inspection/debugging

        pruned_tokenizer = self._build_pruned_tokenizer(retained_ids, old_id_to_new_id)
        pruned_model = self._slice_model_weights(retained_ids)

        return pruned_model, pruned_tokenizer

    def _slice_model_weights(self, retained_ids: List[int]) -> PreTrainedModel:
        """Slices ``embed_tokens`` and ``lm_head`` weights to the retained ids."""
        model = self.model
        idx = torch.tensor(retained_ids, dtype=torch.long)
        new_vocab_size = len(retained_ids)

        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        tied = (
            output_embeddings is not None
            and output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
        )

        with torch.no_grad():
            new_input_weight = input_embeddings.weight.data.index_select(0, idx).clone()

            new_output_weight = None
            new_output_bias = None
            if output_embeddings is not None and not tied:
                new_output_weight = output_embeddings.weight.data.index_select(0, idx).clone()
                if getattr(output_embeddings, "bias", None) is not None:
                    new_output_bias = output_embeddings.bias.data.index_select(0, idx).clone()

        # resize_token_embeddings correctly handles tied weights, then we
        # overwrite every retained row with its original (not re-initialized)
        # embedding so no information is lost for kept tokens.
        model.resize_token_embeddings(new_vocab_size)

        with torch.no_grad():
            model.get_input_embeddings().weight.data.copy_(new_input_weight)
            new_output_embeddings = model.get_output_embeddings()
            if new_output_weight is not None and new_output_embeddings is not None:
                new_output_embeddings.weight.data.copy_(new_output_weight)
                if new_output_bias is not None and getattr(new_output_embeddings, "bias", None) is not None:
                    new_output_embeddings.bias.data.copy_(new_output_bias)

        model.config.vocab_size = new_vocab_size
        return model

    def _build_pruned_tokenizer(
        self, retained_ids: List[int], old_id_to_new_id: Dict[int, int]
    ) -> PreTrainedTokenizerFast:
        """Rebuilds a dense tokenizer JSON containing only the retained ids.

        Handles BPE-style ``vocab``/``merges`` and Unigram-style list vocabs.
        Merge rules are filtered to only those whose left/right parts and
        resulting merged token all survive pruning; this is a best-effort
        transform, since the pruned tokens are intentionally being removed.
        """
        old_tokenizer = self.tokenizer
        backend = old_tokenizer.backend_tokenizer
        data = json.loads(backend.to_str())

        model_data: Dict[str, Any] = data.get("model", {})
        old_vocab = model_data.get("vocab", {})

        if isinstance(old_vocab, dict):
            new_vocab: Dict[str, int] = {}
            for tok, old_id in old_vocab.items():
                if old_id in old_id_to_new_id:
                    new_vocab[tok] = old_id_to_new_id[old_id]
            model_data["vocab"] = new_vocab
            kept_tokens = set(new_vocab.keys())
        elif isinstance(old_vocab, list):
            # Unigram-style vocab: list of [token, score] pairs.
            new_vocab_list = []
            kept_tokens = set()
            for entry in old_vocab:
                tok = entry[0]
                old_id = old_tokenizer.convert_tokens_to_ids(tok)
                if old_id in old_id_to_new_id:
                    new_vocab_list.append(entry)
                    kept_tokens.add(tok)
            model_data["vocab"] = new_vocab_list
        else:
            kept_tokens = {old_tokenizer.convert_ids_to_tokens(i) for i in retained_ids}

        if model_data.get("merges"):
            new_merges = []
            for merge in model_data["merges"]:
                parts = merge.split(" ") if isinstance(merge, str) else list(merge)
                if len(parts) != 2:
                    continue
                left, right = parts
                merged = left + right
                if left in kept_tokens and right in kept_tokens and merged in kept_tokens:
                    new_merges.append(merge)
            model_data["merges"] = new_merges

        data["model"] = model_data

        if data.get("added_tokens"):
            new_added_tokens = []
            for entry in data["added_tokens"]:
                old_id = entry.get("id")
                if old_id in old_id_to_new_id:
                    entry = dict(entry)
                    entry["id"] = old_id_to_new_id[old_id]
                    new_added_tokens.append(entry)
            data["added_tokens"] = new_added_tokens

        post_processor = data.get("post_processor")
        if post_processor:
            self._remap_ids_in_place(post_processor, old_id_to_new_id)

        new_backend = BackendTokenizer.from_str(json.dumps(data))

        pruned_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_backend)
        pruned_tokenizer.add_special_tokens(dict(old_tokenizer.special_tokens_map))
        for attr in ("model_max_length", "padding_side", "truncation_side"):
            if hasattr(old_tokenizer, attr):
                setattr(pruned_tokenizer, attr, getattr(old_tokenizer, attr))

        return pruned_tokenizer

    @staticmethod
    def _remap_ids_in_place(node: Any, old_id_to_new_id: Dict[int, int]) -> None:
        """Recursively remaps any ``"ids"`` integer lists found in a JSON node."""
        if isinstance(node, dict):
            if isinstance(node.get("ids"), list):
                node["ids"] = [old_id_to_new_id.get(i, i) for i in node["ids"]]
            for value in node.values():
                VocabularyPruner._remap_ids_in_place(value, old_id_to_new_id)
        elif isinstance(node, list):
            for item in node:
                VocabularyPruner._remap_ids_in_place(item, old_id_to_new_id)
