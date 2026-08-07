"""Tests for tok_adapt.dedup.CorpusPreprocessor."""

from __future__ import annotations

import pytest

from tok_adapt.dedup import CorpusPreprocessor


@pytest.fixture()
def raw_corpus_dir(tmp_path):
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()
    lines = [
        "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिखा गया है।",
        "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिखा गया है।",  # exact duplicate
        "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिख गया है।",  # near-duplicate (1 char diff)
        "यह पूरी तरह से एक अलग और असंबंधित हिंदी वाक्य है जिसमें कोई दोहराव नहीं है।",
        "This is an English sentence that should be filtered out by language-ID.",
        "short",  # below min_chars, dropped regardless of language
    ]
    (corpus_dir / "a.txt").write_text("\n".join(lines), encoding="utf-8")
    return corpus_dir


def test_deduplicate_removes_exact_and_near_duplicates():
    pre = CorpusPreprocessor(jaccard_threshold=0.85)
    lines = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog today",
        "completely unrelated sentence about something else entirely",
    ]
    deduped = pre.deduplicate(lines)
    # first two are exact duplicates -> one survives; third is near-duplicate
    # of the first at high similarity -> also collapses; last is distinct.
    assert lines[0] in deduped
    assert lines[-1] in deduped
    assert len(deduped) < len(lines)


def test_split_corpus_respects_byte_budget():
    pre = CorpusPreprocessor(seed=0)
    lines = ["x" * 100 for _ in range(50)]  # ~5KB total
    corpus_a, corpus_b = pre.split_corpus(lines, corpus_a_max_bytes=1000)
    assert sum(len(l.encode("utf-8")) + 1 for l in corpus_a) <= 1000
    assert corpus_b == lines
    assert all(line in lines for line in corpus_a)


def test_process_writes_corpus_a_and_b(raw_corpus_dir, tmp_path):
    out_dir = tmp_path / "out"
    pre = CorpusPreprocessor(languages=["hi"], min_chars=10, jaccard_threshold=0.85, seed=0)
    stats = pre.process(raw_corpus_dir, out_dir, corpus_a_max_bytes=10_000)

    assert (out_dir / "corpus_a.txt").exists()
    assert (out_dir / "corpus_b.txt").exists()

    assert stats.raw_lines == 6
    assert stats.dropped_empty >= 1  # the "short" line
    assert stats.dropped_language >= 1  # the English line
    assert stats.dropped_duplicate >= 1  # exact + near duplicate Hindi lines
    assert stats.kept_lines == stats.corpus_b_lines
    assert stats.corpus_a_lines <= stats.corpus_b_lines

    corpus_b_text = (out_dir / "corpus_b.txt").read_text(encoding="utf-8")
    assert "English sentence" not in corpus_b_text


def test_process_without_language_filter_keeps_all_languages(raw_corpus_dir, tmp_path):
    out_dir = tmp_path / "out_nofilter"
    pre = CorpusPreprocessor(languages=None, min_chars=10, seed=0)
    stats = pre.process(raw_corpus_dir, out_dir)
    corpus_b_text = (out_dir / "corpus_b.txt").read_text(encoding="utf-8")
    assert "English sentence" in corpus_b_text
    assert stats.dropped_language == 0
