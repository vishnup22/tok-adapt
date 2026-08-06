"""Smart initialization of newly added token embeddings."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

_SUPPORTED_STRATEGIES = ("subword_mean",)


class EmbeddingInitializer:
    """Initializes new token embeddings from decomposed subword statistics.

    Args:
        model: The model whose embedding matrices will be resized in place.
        tokenizer: The *original* (pre-expansion) tokenizer, used to
            decompose newly added token strings into known subwords.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def smart_resize_embeddings(
        self,
        new_tokenizer: PreTrainedTokenizerFast,
        strategy: str = "subword_mean",
    ) -> PreTrainedModel:
        """Resizes the model's embeddings and initializes new rows intelligently.

        For every token id at or beyond the original vocabulary size, the new
        token's string is re-tokenized with the *original* tokenizer. If it
        decomposes into known subwords, the new embedding is the mean of
        those subwords' embeddings; otherwise it falls back to the mean of
        all original embeddings.

        Args:
            new_tokenizer: The expanded tokenizer (e.g. returned by
                :meth:`tok_adapt.expansion.VocabularyExpander.merge_vocabularies`).
                If it carries an ``original_vocab_size`` attribute, that value
                is used as the boundary between old and new ids; otherwise
                ``len(self.tokenizer)`` is used.
            strategy: Initialization strategy. Currently only
                ``"subword_mean"`` is supported.

        Returns:
            The model, resized and modified in place.

        Raises:
            ValueError: If ``strategy`` is not supported.
        """
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Choose one of {_SUPPORTED_STRATEGIES}."
            )

        original_vocab_size = getattr(new_tokenizer, "original_vocab_size", len(self.tokenizer))
        new_vocab_size = len(new_tokenizer)

        self.model.resize_token_embeddings(new_vocab_size)

        input_embeddings = self.model.get_input_embeddings()
        output_embeddings = self.model.get_output_embeddings()
        tied = (
            output_embeddings is not None
            and output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
        )

        if new_vocab_size <= original_vocab_size:
            # Nothing new to initialize (tokenizer did not grow).
            self.model.config.vocab_size = new_vocab_size
            return self.model

        mean_embedding = input_embeddings.weight.data[:original_vocab_size].mean(dim=0)

        with torch.no_grad():
            for new_id in range(original_vocab_size, new_vocab_size):
                token_str = new_tokenizer.convert_ids_to_tokens(new_id)
                clean_str = new_tokenizer.convert_tokens_to_string([token_str]).strip()
                if not clean_str:
                    clean_str = token_str

                old_ids = self.tokenizer.encode(clean_str, add_special_tokens=False)

                if old_ids:
                    subword_vecs = input_embeddings.weight.data[old_ids]
                    new_vec = subword_vecs.mean(dim=0)
                else:
                    new_vec = mean_embedding

                input_embeddings.weight.data[new_id] = new_vec
                if output_embeddings is not None and not tied:
                    output_embeddings.weight.data[new_id] = new_vec

        self.model.config.vocab_size = new_vocab_size
        return self.model
