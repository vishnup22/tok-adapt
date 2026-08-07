"""Corpus ingestion: near-duplicate removal, language-ID filtering, and splitting.

Implements Phase 1 of the end-to-end pipeline (see ``docs/pipeline.md``):
collect raw text, deduplicate it, keep only the target language(s), then
split the cleaned result into a small "Corpus A" sample for sub-tokenizer
training and the full "Corpus B" for continued pre-training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from datasketch import MinHash, MinHashLSH


@dataclass
class PreprocessStats:
    """Summary counters returned by :meth:`CorpusPreprocessor.process`."""

    raw_lines: int = 0
    dropped_empty: int = 0
    dropped_language: int = 0
    dropped_duplicate: int = 0
    kept_lines: int = 0
    corpus_a_lines: int = 0
    corpus_b_lines: int = 0
    corpus_a_bytes: int = 0
    corpus_b_bytes: int = 0
    language_counts: dict = field(default_factory=dict)


def _read_lines(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[str]:
    """Reads and concatenates non-empty stripped lines from one or more files/dirs."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file()))
        else:
            files.append(p)

    lines: List[str] = []
    for f in files:
        lines.extend(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return lines


def _shingles(text: str, n: int) -> Iterable[str]:
    """Yields character n-gram shingles used as MinHash features."""
    if len(text) < n:
        yield text
        return
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


class CorpusPreprocessor:
    """Deduplicates, language-filters, and splits raw text corpora.

    Args:
        languages: ISO 639-1 codes to keep (e.g. ``["hi", "te"]``). If
            ``None``, no language filtering is applied.
        jaccard_threshold: Minimum estimated Jaccard similarity for two
            lines to be considered near-duplicates.
        num_perm: Number of MinHash permutations (higher = more accurate,
            slower). 128 is the datasketch default and a reasonable
            trade-off for corpus-scale dedup.
        ngram_size: Character shingle size used to build each line's
            MinHash signature.
        min_chars: Lines shorter than this (after stripping) are dropped
            as too short to reliably language-ID or usefully train on.
        seed: RNG seed used when sampling Corpus A.
    """

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        jaccard_threshold: float = 0.85,
        num_perm: int = 128,
        ngram_size: int = 5,
        min_chars: int = 20,
        seed: int = 0,
    ) -> None:
        self.languages = set(languages) if languages else None
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.min_chars = min_chars
        self.seed = seed

    def _minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for shingle in _shingles(text, self.ngram_size):
            mh.update(shingle.encode("utf-8"))
        return mh

    def deduplicate(self, lines: Sequence[str]) -> List[str]:
        """Removes near-duplicate lines using MinHash + LSH.

        Keeps the first occurrence of each near-duplicate cluster in
        input order. Uses Locality-Sensitive Hashing so this scales
        roughly linearly instead of the O(n^2) pairwise comparisons a
        naive Jaccard sweep would require.

        Args:
            lines: Candidate lines, already stripped of surrounding
                whitespace.

        Returns:
            The de-duplicated subset of ``lines``, in original order.
        """
        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=self.num_perm)
        kept: List[str] = []
        for i, line in enumerate(lines):
            mh = self._minhash(line)
            key = str(i)
            if lsh.query(mh):
                continue
            lsh.insert(key, mh)
            kept.append(line)
        return kept

    def filter_by_language(self, lines: Sequence[str]) -> tuple[List[str], dict]:
        """Keeps only lines detected as one of ``self.languages``.

        Args:
            lines: Candidate lines.

        Returns:
            A tuple of ``(kept_lines, language_counts)`` where
            ``language_counts`` maps detected language code -> line count
            across *all* input lines (kept or not), useful for auditing
            corpus composition.
        """
        if self.languages is None:
            return list(lines), {}

        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = self.seed  # deterministic detection

        kept: List[str] = []
        counts: dict = {}
        for line in lines:
            try:
                lang = detect(line)
            except LangDetectException:
                lang = "unknown"
            counts[lang] = counts.get(lang, 0) + 1
            if lang in self.languages:
                kept.append(line)
        return kept, counts

    def split_corpus(
        self,
        lines: Sequence[str],
        corpus_a_max_bytes: int = 30_000_000,
    ) -> tuple[List[str], List[str]]:
        """Splits cleaned lines into Corpus A (tokenizer sample) and Corpus B (CPT).

        Corpus A is a random, order-shuffled sample of up to
        ``corpus_a_max_bytes`` drawn from the full cleaned corpus, sized
        for sub-tokenizer training (~10-50MB per the pipeline spec).
        Corpus B is the *entire* cleaned corpus (Corpus A is a subset of
        it), since continued pre-training should see all cleaned data.

        Args:
            lines: Cleaned (deduplicated + language-filtered) lines.
            corpus_a_max_bytes: Byte budget for Corpus A.

        Returns:
            ``(corpus_a_lines, corpus_b_lines)``.
        """
        rng = random.Random(self.seed)
        shuffled = list(lines)
        rng.shuffle(shuffled)

        corpus_a: List[str] = []
        running_bytes = 0
        for line in shuffled:
            line_bytes = len(line.encode("utf-8")) + 1
            if running_bytes + line_bytes > corpus_a_max_bytes:
                break
            corpus_a.append(line)
            running_bytes += line_bytes

        return corpus_a, list(lines)

    def process(
        self,
        input_paths: Union[str, Path, Sequence[Union[str, Path]]],
        output_dir: Union[str, Path],
        corpus_a_max_bytes: int = 30_000_000,
    ) -> PreprocessStats:
        """Runs the full Phase 1 pipeline and writes ``corpus_a.txt`` / ``corpus_b.txt``.

        Args:
            input_paths: Raw text file(s) or directory/directories to ingest.
            output_dir: Directory to write ``corpus_a.txt`` and
                ``corpus_b.txt`` into (created if missing).
            corpus_a_max_bytes: Byte budget for Corpus A (sub-tokenizer
                training sample).

        Returns:
            A :class:`PreprocessStats` with counts at each stage.
        """
        stats = PreprocessStats()

        raw_lines = _read_lines(input_paths)
        stats.raw_lines = len(raw_lines)

        stripped = [ln.strip() for ln in raw_lines]
        non_empty = [ln for ln in stripped if len(ln) >= self.min_chars]
        stats.dropped_empty = stats.raw_lines - len(non_empty)

        language_filtered, lang_counts = self.filter_by_language(non_empty)
        stats.dropped_language = len(non_empty) - len(language_filtered)
        stats.language_counts = lang_counts

        deduped = self.deduplicate(language_filtered)
        stats.dropped_duplicate = len(language_filtered) - len(deduped)
        stats.kept_lines = len(deduped)

        corpus_a, corpus_b = self.split_corpus(deduped, corpus_a_max_bytes=corpus_a_max_bytes)
        stats.corpus_a_lines = len(corpus_a)
        stats.corpus_b_lines = len(corpus_b)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        a_path = out_dir / "corpus_a.txt"
        b_path = out_dir / "corpus_b.txt"
        a_text = "\n".join(corpus_a) + ("\n" if corpus_a else "")
        b_text = "\n".join(corpus_b) + ("\n" if corpus_b else "")
        a_path.write_text(a_text, encoding="utf-8")
        b_path.write_text(b_text, encoding="utf-8")
        stats.corpus_a_bytes = len(a_text.encode("utf-8"))
        stats.corpus_b_bytes = len(b_text.encode("utf-8"))

        return stats
