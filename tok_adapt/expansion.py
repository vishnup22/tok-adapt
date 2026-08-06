"""Sub-tokenizer training and safe vocabulary merging into a base tokenizer."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from transformers import PreTrainedTokenizerFast

_SUPPORTED_ALGORITHMS = ("bpe", "unigram")


class VocabularyExpander:
    """Trains a domain/language-specific sub-tokenizer and merges it into a base tokenizer.

    Args:
        base_tokenizer: The Hugging Face fast tokenizer to extend.
    """

    def __init__(self, base_tokenizer: PreTrainedTokenizerFast) -> None:
        self.base_tokenizer = base_tokenizer
        self.original_vocab_size = len(base_tokenizer)
        self._trained_tokenizer_path: Path | None = None

    def train_sub_tokenizer(
        self,
        corpus_path: str,
        vocab_size: int,
        algorithm: str = "bpe",
        output_dir: Union[str, None] = None,
    ) -> Path:
        """Trains a new subword tokenizer on a target-domain corpus.

        Args:
            corpus_path: Path to a text file, or a directory of text files,
                containing the target corpus (one training document per line).
            vocab_size: Target vocabulary size for the sub-tokenizer.
            algorithm: Either ``"bpe"`` or ``"unigram"``.
            output_dir: Optional directory to save the trained tokenizer JSON.
                Defaults to a fresh temp directory.

        Returns:
            Path to the saved ``tokenizers.json`` file for the trained
            sub-tokenizer.

        Raises:
            ValueError: If ``algorithm`` is not supported.
            FileNotFoundError: If ``corpus_path`` does not exist.
        """
        algorithm = algorithm.lower()
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose one of {_SUPPORTED_ALGORITHMS}."
            )

        corpus = Path(corpus_path)
        if not corpus.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")
        files = [str(p) for p in sorted(corpus.glob("*")) if p.is_file()] if corpus.is_dir() else [str(corpus)]
        if not files:
            raise FileNotFoundError(f"No corpus files found under: {corpus_path}")

        if algorithm == "bpe":
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(
                vocab_size=vocab_size, special_tokens=["[UNK]"], show_progress=True
            )
        else:
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=["[UNK]"],
                unk_token="[UNK]",
                show_progress=True,
            )

        # Deliberately NOT ByteLevel: a ByteLevel pre-tokenizer stores vocab
        # entries in a remapped byte-alphabet (e.g. a literal space becomes
        # 'Ġ' / U+0120), so tokens read back via get_vocab() never literally
        # occur in normal Unicode text. Since merge_vocabularies() adds these
        # tokens to the base tokenizer as literal strings (via add_tokens,
        # which matches raw text), a ByteLevel-trained vocab would silently
        # fail to match anything at inference time except by coincidental
        # pure-ASCII overlap. Metaspace keeps entries as literal Unicode
        # substrings (SentencePiece-style, with '▁' marking word starts),
        # which are safe to add as literal tokens regardless of what
        # pre-tokenization scheme the base tokenizer itself uses internally.
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        tokenizer.train(files, trainer)

        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tok_adapt_subtok_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sub_tokenizer.json"
        tokenizer.save(str(out_path))

        self._trained_tokenizer_path = out_path
        return out_path

    def merge_vocabularies(self, new_tokens_file: str) -> PreTrainedTokenizerFast:
        """Merges tokens from a trained sub-tokenizer into the base tokenizer.

        Reads the vocabulary of the sub-tokenizer saved at ``new_tokens_file``,
        filters out any token already present in the base tokenizer (or
        placeholder special tokens used only during sub-tokenizer training),
        and injects the remaining unique tokens via ``add_tokens`` so the
        base tokenizer's existing merge rules are left untouched.

        Args:
            new_tokens_file: Path to a ``tokenizers`` JSON file (as produced
                by :meth:`train_sub_tokenizer`) or any file loadable via
                ``tokenizers.Tokenizer.from_file``.

        Returns:
            A deep copy of the base tokenizer with new tokens added. The
            returned tokenizer carries an ``original_vocab_size`` attribute
            recording the vocab size before the merge, for use by
            :mod:`tok_adapt.initialization`.

        Raises:
            FileNotFoundError: If ``new_tokens_file`` does not exist.
        """
        path = Path(new_tokens_file)
        if not path.exists():
            raise FileNotFoundError(f"New tokens file not found: {new_tokens_file}")

        sub_tokenizer = Tokenizer.from_file(str(path))
        new_vocab = sub_tokenizer.get_vocab()  # token -> id
        sorted_ids = [tid for _, tid in sorted(new_vocab.items(), key=lambda kv: kv[1])]

        # Decode each id back to its literal Unicode surface form (undoes any
        # word-boundary marker such as Metaspace's leading '▁') rather than
        # using the raw vocab key, so what we hand to add_tokens is exactly
        # the substring that will occur in real input text.
        placeholder_tokens = {"[UNK]", "<unk>", "<UNK>"}
        seen: set[str] = set()
        candidate_tokens: List[str] = []
        for tid in sorted_ids:
            decoded = sub_tokenizer.decode([tid]).strip()
            if not decoded or decoded in placeholder_tokens or decoded in seen:
                continue
            seen.add(decoded)
            # Skip anything the base tokenizer already encodes as a single
            # token -- comparing decoded literal text this way is robust
            # regardless of whether the base tokenizer's own internal vocab
            # keys are byte-remapped (GPT-2/Llama/Qwen-style) or literal
            # (SentencePiece/WordPiece-style).
            if len(self.base_tokenizer.encode(decoded, add_special_tokens=False)) == 1:
                continue
            candidate_tokens.append(decoded)

        extended_tokenizer = copy.deepcopy(self.base_tokenizer)
        original_size = len(extended_tokenizer)

        if candidate_tokens:
            extended_tokenizer.add_tokens(candidate_tokens)

        # Stash the pre-merge vocab size so downstream modules (e.g.
        # EmbeddingInitializer) can identify which ids are newly added
        # without recomputing a diff against the base tokenizer.
        extended_tokenizer.original_vocab_size = original_size
        return extended_tokenizer
