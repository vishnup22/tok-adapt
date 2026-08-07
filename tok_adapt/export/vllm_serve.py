"""vLLM-based serving for high-throughput inference.

Implements the vLLM portion of Phase 6. vLLM has no native Windows build
(https://github.com/vllm-project/vllm is Linux/WSL only), so this module
is a thin, defensively-checked wrapper: it raises a clear, actionable
error on platforms where vLLM cannot be imported, and otherwise exposes a
minimal offline-batch-inference helper plus an OpenAI-compatible server
launcher.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import List, Optional, Sequence


class VLLMUnavailableError(RuntimeError):
    """Raised when vLLM is not importable in the current environment."""


def _require_vllm() -> None:
    if importlib.util.find_spec("vllm") is None:
        raise VLLMUnavailableError(
            "vllm is not installed/importable. vLLM has no native Windows build; "
            "install it under WSL2 or a Linux host with `pip install vllm` "
            "(see the tok-adapt[serve] extra in pyproject.toml, which only resolves on Linux)."
        )


def generate_with_vllm(
    model_dir: str,
    prompts: Sequence[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    **engine_kwargs,
) -> List[str]:
    """Runs offline batch generation with vLLM.

    Args:
        model_dir: Path or Hub id of the checkpoint to serve.
        prompts: Prompts to generate completions for.
        max_tokens: Maximum new tokens per completion.
        temperature: Sampling temperature (``0.0`` = greedy).
        **engine_kwargs: Extra kwargs forwarded to ``vllm.LLM(...)`` (e.g.
            ``gpu_memory_utilization``, ``quantization``).

    Returns:
        Generated completion text, one per prompt, in the same order.

    Raises:
        VLLMUnavailableError: If vLLM cannot be imported on this platform.
    """
    _require_vllm()
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_dir, **engine_kwargs)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(list(prompts), sampling_params)
    return [output.outputs[0].text for output in outputs]


def serve_with_vllm(
    model_dir: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra_args: Optional[Sequence[str]] = None,
) -> subprocess.Popen:
    """Launches an OpenAI-compatible vLLM server as a background subprocess.

    Args:
        model_dir: Path or Hub id of the checkpoint to serve.
        host: Bind host.
        port: Bind port.
        extra_args: Additional CLI args forwarded to
            ``vllm.entrypoints.openai.api_server`` (e.g.
            ``["--quantization", "awq", "--max-model-len", "4096"]``).

    Returns:
        The running ``subprocess.Popen`` handle; callers are responsible
        for terminating it.

    Raises:
        VLLMUnavailableError: If vLLM cannot be imported on this platform.
    """
    _require_vllm()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--host",
        host,
        "--port",
        str(port),
        *(extra_args or []),
    ]
    return subprocess.Popen(cmd)
