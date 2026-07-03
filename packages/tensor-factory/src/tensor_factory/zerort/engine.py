"""Graph executor for the zero-dependency runtime.

``Runtime`` loads an ``.onnx`` file via :mod:`tensor_factory.zerort.pb` and executes its
nodes in graph order (ONNX guarantees a topological sort). The API mirrors the small
slice of ``onnxruntime.InferenceSession`` the :class:`tensor_factory.inference.Detector`
uses, so the two engines are drop-in swappable behind it.
"""

from __future__ import annotations

from pathlib import Path

from . import ops, tfm
from .pb import parse_model
from .tensor import Tensor


class Runtime:
    """Execute an ONNX or .tfm graph with pure-Python kernels."""

    def __init__(self, model: str | Path | bytes) -> None:
        data = model if isinstance(model, bytes) else Path(model).read_bytes()
        self.model = tfm.load(data) if data[:4] == tfm.MAGIC else parse_model(data)
        graph = self.model.graph
        self._initializers = {t.name: ops.tensor_from_def(t) for t in graph.initializers if t.name}
        self.input_names = [n for n in graph.input_names if n not in self._initializers]
        self.output_names = list(graph.output_names)
        missing = {node.op_type for node in graph.nodes if node.op_type not in ops.OPS}
        if missing:
            raise ops.UnsupportedOp(f"model uses unimplemented ops: {sorted(missing)}")
        # Hoist input-independent ConvInteger weight prep (zero-point folds, lane
        # sizing) out of the per-frame path -- weights are initializers in any
        # trained export, so this runs once per model, not once per inference.
        for node in graph.nodes:
            if node.op_type != "ConvInteger" or len(node.inputs) < 2:
                continue
            w = self._initializers.get(node.inputs[1])
            if w is None:
                continue
            wzp = 0
            if len(node.inputs) > 3 and node.inputs[3]:
                wzp_t = self._initializers.get(node.inputs[3])
                if wzp_t is None:
                    continue  # dynamic zero point: prep at run time instead
                wzp = int(wzp_t.item())
            node.attrs["_prep"] = ops.prepare_conv_integer(w, wzp)

    def run(self, feeds: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the graph on ``feeds`` (name -> :class:`Tensor`); return all outputs."""
        unknown = set(feeds) - set(self.input_names)
        if unknown:
            raise ValueError(f"unknown inputs {sorted(unknown)}; expected {self.input_names}")
        env: dict[str, Tensor] = dict(self._initializers)
        env.update(feeds)
        for node in self.model.graph.nodes:
            try:
                ins: list[Tensor | None] = [env[n] if n else None for n in node.inputs]
            except KeyError as exc:
                raise ValueError(
                    f"node {node.name or node.op_type}: missing input {exc.args[0]!r}"
                ) from exc
            outs = ops.apply(node.op_type, ins, node.attrs)
            for name, value in zip(node.outputs, outs, strict=False):
                if name:
                    env[name] = value
        try:
            return {n: env[n] for n in self.output_names}
        except KeyError as exc:
            raise ValueError(f"graph never produced output {exc.args[0]!r}") from exc
