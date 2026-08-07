"""ONNX export for portable/accelerated inference.

Implements the ONNX portion of Phase 6 using Hugging Face's ``optimum``
exporter, which handles operator-graph tracing and per-architecture opset
selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# optimum-onnx 0.1.0 registers its per-architecture OnnxConfig classes (GPT2,
# Llama, etc.) via decorators in this submodule, but nothing else in the
# import chain triggers that module to load -- so TasksManager's registry is
# silently empty ("model_type is not supported ... transformers") unless it's
# imported explicitly first. This is a known lazy-import ordering bug in the
# freshly split-out optimum-onnx package; forcing the import here is the fix.
import optimum.exporters.onnx.model_configs  # noqa: F401
from optimum.exporters.onnx import main_export


def _patch_optimum_partial_descriptor_bug() -> None:
    """Works around a Python 3.14 / optimum compatibility bug.

    Python 3.14 made ``functools.partial`` implement the descriptor
    protocol (``__get__``), so a partial stored as a plain class attribute
    now auto-binds ``self`` as an extra leading positional argument, the
    same way a plain function would. optimum's ``ExporterConfig`` stores
    each architecture's ``NORMALIZED_CONFIG_CLASS`` as exactly such a
    partial and calls it via ``self.NORMALIZED_CONFIG_CLASS(self._config)``,
    which worked under the pre-3.14 (non-descriptor) semantics it was
    written against. Under 3.14 this silently becomes a call with two
    positional arguments, colliding with the partial's own keyword-bound
    ``allow_new`` and raising
    ``TypeError: NormalizedConfig.__init__() got multiple values for
    argument 'allow_new'`` for every architecture, not just GPT-2.

    Fixes it by re-implementing ``ExporterConfig.__init__`` to fetch
    ``NORMALIZED_CONFIG_CLASS`` straight from the defining class's
    ``__dict__`` (bypassing instance-attribute lookup, and therefore the
    descriptor binding) before calling it. No-op if already applied, and
    silently skipped if optimum's internals have moved on and no longer
    match the shape this patches.
    """
    from optimum.exporters.base import ExporterConfig

    if getattr(ExporterConfig, "_tok_adapt_unbound_normalized_config", False):
        return

    def patched_init(self, config, task, int_dtype="int64", float_dtype="fp32"):
        self.task = task
        self._config = config
        normalized_config_cls = None
        for klass in type(self).__mro__:
            if "NORMALIZED_CONFIG_CLASS" in klass.__dict__:
                normalized_config_cls = klass.__dict__["NORMALIZED_CONFIG_CLASS"]
                break
        if normalized_config_cls is None:
            raise RuntimeError(f"No NORMALIZED_CONFIG_CLASS found on {type(self).__mro__}")
        self._normalized_config = normalized_config_cls(self._config)
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    ExporterConfig.__init__ = patched_init
    ExporterConfig._tok_adapt_unbound_normalized_config = True


_patch_optimum_partial_descriptor_bug()


def export_to_onnx(
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    task: str = "text-generation-with-past",
    opset: Optional[int] = None,
) -> Path:
    """Exports a Hugging Face causal LM checkpoint to ONNX.

    Args:
        model_dir: Directory or Hub id of the checkpoint to export.
        output_dir: Directory to write the ONNX graph + external weights to.
        task: The optimum export task. ``"text-generation-with-past"``
            exports a decoder graph with KV-cache inputs, matching how an
            ONNX Runtime-based server would actually serve generation.
        opset: ONNX opset version. Defaults to optimum's per-architecture
            recommended opset when omitted.

    Returns:
        The resolved output directory.

    Raises:
        RuntimeError: If the underlying export fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        main_export(
            model_name_or_path=str(model_dir),
            output=str(output_dir),
            task=task,
            opset=opset,
        )
    except Exception as exc:  # optimum raises a variety of exporter-specific error types
        raise RuntimeError(f"ONNX export failed for {model_dir}: {exc}") from exc
    return output_dir
