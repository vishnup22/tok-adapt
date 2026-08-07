"""GGUF export for llama.cpp-based edge deployment.

Implements the GGUF portion of Phase 6. GGUF's HF-to-GGUF tensor mapping
logic lives in llama.cpp's ``convert_hf_to_gguf.py`` (there is no pip
package that reimplements this correctly and completely for every
architecture), so this module fetches that script -- and its small
``gguf-py`` / ``conversion`` support packages -- via a shallow, sparse
git checkout, and shells out to it. The sparse checkout pulls ~2MB rather
than the full llama.cpp C++ source tree.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tok_adapt" / "llama.cpp"

# llama.cpp identifies a tokenizer's BPE pre-tokenizer by hashing the
# encoded output of a fixed probe string and matching it against a table
# of known reference models (see conversion/base.py:get_vocab_base_pre).
# That hash is a function of the *entire learned vocabulary*, not just the
# pre-tokenizer's regex -- so any tok_adapt-expanded tokenizer (new merges
# on top of a stock base) produces a hash absent from that table and
# fails with "BPE pre-tokenizer was not recognized", even though the
# underlying pre-tokenizer regex is unchanged from its base model. This
# wrapper script template lets a caller who knows the base architecture
# (e.g. "gpt2") declare it directly, bypassing the hash lookup.
_OVERRIDE_WRAPPER_TEMPLATE = """\
import runpy
import sys

sys.path.insert(0, {repo_dir!r})
sys.path.insert(0, {gguf_py_dir!r})

import conversion.base as _conversion_base  # noqa: E402

_conversion_base.TextModel.get_vocab_base_pre = lambda self, tokenizer: {hint!r}

sys.argv = [{convert_script!r}, *sys.argv[1:]]
runpy.run_path({convert_script!r}, run_name="__main__")
"""

# Only the files convert_hf_to_gguf.py actually imports at runtime.
_SPARSE_PATHS = ["convert_hf_to_gguf.py", "gguf-py", "conversion"]


def _ensure_llama_cpp_checkout(cache_dir: Path) -> Path:
    """Ensures a sparse llama.cpp checkout with convert_hf_to_gguf.py exists locally.

    Args:
        cache_dir: Directory to clone/checkout llama.cpp into.

    Returns:
        Path to the checked-out repo root.

    Raises:
        RuntimeError: If git is unavailable, or the clone/checkout fails,
            or the checkout unexpectedly lacks the conversion script.
    """
    convert_script = cache_dir / "convert_hf_to_gguf.py"
    if convert_script.exists():
        return cache_dir

    if shutil.which("git") is None:
        raise RuntimeError(
            "git is required to fetch llama.cpp's GGUF conversion script but was not found on PATH. "
            "Install git, or pass llama_cpp_dir= to an existing local llama.cpp checkout."
        )

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", LLAMA_CPP_REPO, str(cache_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "--skip-checks", *_SPARSE_PATHS],
            check=True,
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to fetch llama.cpp conversion tooling: {exc.stderr}") from exc

    if not convert_script.exists():
        raise RuntimeError(f"llama.cpp checkout at {cache_dir} did not produce convert_hf_to_gguf.py.")
    return cache_dir


def export_to_gguf(
    model_dir: Union[str, Path],
    output_path: Union[str, Path],
    outtype: str = "f16",
    llama_cpp_dir: Optional[Union[str, Path]] = None,
    pre_tokenizer_hint: Optional[str] = None,
) -> Path:
    """Converts a Hugging Face checkpoint directory to GGUF format.

    Args:
        model_dir: Directory containing a ``save_pretrained``-style
            checkpoint (config.json, tokenizer files, safetensors weights).
        output_path: Destination ``.gguf`` file path.
        outtype: Output tensor type, forwarded to
            ``convert_hf_to_gguf.py --outtype`` (e.g. ``"f16"``, ``"f32"``,
            ``"bf16"``, ``"q8_0"``). Further k-quant types (``q4_k_m``,
            etc.) require building llama.cpp's ``llama-quantize`` binary
            separately and running it on the GGUF this function produces.
        llama_cpp_dir: Path to an existing llama.cpp checkout. If omitted,
            a minimal sparse checkout is fetched to
            ``~/.cache/tok_adapt/llama.cpp`` on first use and reused after.
        pre_tokenizer_hint: Declares the base tokenizer's pre-tokenizer
            identifier (e.g. ``"gpt2"``, ``"llama-bpe"``, ``"qwen2"`` --
            see llama.cpp's ``conversion/base.py`` for the full list),
            bypassing llama.cpp's hash-based auto-detection. Needed for
            checkpoints produced by :class:`tok_adapt.expansion.VocabularyExpander`:
            their learned merges differ from any stock model's, so the
            probe-string hash llama.cpp fingerprints tokenizers with
            won't match its reference table even though the underlying
            pre-tokenizer regex is unchanged from the base model. Omit
            for unmodified stock checkpoints, where auto-detection works.

    Returns:
        The resolved output ``.gguf`` path.

    Raises:
        RuntimeError: If the conversion tooling cannot be fetched, or the
            conversion subprocess exits non-zero.
    """
    repo_dir = Path(llama_cpp_dir) if llama_cpp_dir else _ensure_llama_cpp_checkout(DEFAULT_CACHE_DIR)
    convert_script = repo_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py not found under {repo_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cli_args = [str(model_dir), "--outfile", str(output_path), "--outtype", outtype]

    if pre_tokenizer_hint is None:
        script_args = [sys.executable, str(convert_script), *cli_args]
        cleanup: Optional[Path] = None
    else:
        wrapper_src = _OVERRIDE_WRAPPER_TEMPLATE.format(
            repo_dir=str(repo_dir),
            gguf_py_dir=str(repo_dir / "gguf-py"),
            hint=pre_tokenizer_hint,
            convert_script=str(convert_script),
        )
        wrapper_fd, wrapper_path_str = tempfile.mkstemp(suffix="_gguf_convert_wrapper.py")
        with open(wrapper_fd, "w", encoding="utf-8") as f:
            f.write(wrapper_src)
        cleanup = Path(wrapper_path_str)
        script_args = [sys.executable, wrapper_path_str, *cli_args]

    try:
        result = subprocess.run(script_args, capture_output=True, text=True)
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stdout}\n{result.stderr}")

    return output_path
