"""Quantize a tensor-factory detector to a static int8 ``.espdl`` model for the ESP32-P4.

The on-device graph is the convolutional part only: it returns the 4-channel edge heatmap
and the presence logit. The soft-argmax that turns the heatmap into a box runs in firmware
(a softmax over a few hundred cells), which keeps Softmax/Range/Reshape out of the
quantized graph.

Weights come from an exported tensor-factory ONNX model (``--onnx``; batch norm is already
folded into it) or are random (latency does not depend on them). With ``--eval`` it also
reports float vs int8 accuracy on a labeled dataset, on the desktop, before anything is
flashed.

Runs in its own environment (ESP-PPQ pins its own torch/onnx):

    uv venv ~/esp/ppq-venv --python 3.11
    VIRTUAL_ENV=~/esp/ppq-venv uv pip install esp-ppq torch pillow
    PYTHONPATH=packages/tensor-factory-train/src:packages/tensor-factory/src \
        ~/esp/ppq-venv/bin/python firmware/esp32-p4/tools/export_espdl.py --size 160 --out m.espdl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch  # ty: ignore[import-not-found]
from torch import nn  # ty: ignore[import-not-found]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[import-not-found]

from tensor_factory.zerort import parse_model
from tensor_factory_train.data import load_coco_boxes, load_negatives
from tensor_factory_train.model import soft_argmax_xyxy

_EXTS = {".png", ".jpg", ".jpeg"}


class DeviceGraph(nn.Module):
    """The detector minus the soft-argmax: image -> (heat [1,4,h,w], presence [1,1])."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        chans = [3, width, 2 * width, 4 * width, 4 * width]
        layers: list[nn.Module] = []
        for cin, cout in zip(chans, chans[1:], strict=False):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.heat = nn.Conv2d(4 * width, 4, 1)
        self.presence_head = nn.Linear(8 * width, 1)
        self.gain = 1.0  # soft-argmax inverse temperature, applied in firmware

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        pooled = torch.cat(
            [
                nn.functional.adaptive_avg_pool2d(f, 1).flatten(1),
                nn.functional.adaptive_max_pool2d(f, 1).flatten(1),
            ],
            dim=1,
        )
        return self.heat(f), self.presence_head(pooled)

    @classmethod
    def from_onnx(cls, path: str | Path) -> DeviceGraph:
        """Rebuild the float weights from a tensor-factory int8 ONNX: w = (q - zp) * scale."""
        init = {t.name: t for t in parse_model(Path(path).read_bytes()).graph.initializers}

        def weight(prefix: str) -> torch.Tensor:
            q = init[f"{prefix}_quantized"]
            zp, scale = init[f"{prefix}_zero_point"].data[0], init[f"{prefix}_scale"].data[0]
            return (torch.tensor(q.data, dtype=torch.float32) - zp).mul(scale).reshape(q.dims)

        def vector(name: str) -> torch.Tensor:
            return torch.tensor(init[name].data, dtype=torch.float32)

        convs = sorted(
            (
                n[: -len("_quantized")]
                for n in init
                if n.startswith("onnx::Conv_") and n.endswith("_quantized")
            ),
            key=lambda n: int(n.rsplit("_", 1)[1]),
        )
        biases = sorted(
            (
                n
                for n in init
                if n.startswith("onnx::Conv_") and len(init[n].dims) == 1 and init[n].data_type == 1
            ),
            key=lambda n: int(n.rsplit("_", 1)[1]),
        )
        # The exporter names this "presence_head.weight" or "..._Transposed" by version.
        head = [
            n[: -len("_quantized")]
            for n in init
            if n.startswith("presence_head.weight") and n.endswith("_quantized")
        ]
        if len(convs) != 4 or len(biases) != 4 or len(head) != 1:
            raise SystemExit(f"{path}: not a 4-block presence-head detector export")
        model = cls(width=init[f"{convs[0]}_quantized"].dims[0])
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(convs, biases, strict=True)):
                conv = model.features[2 * i]
                conv.weight.copy_(weight(w))
                conv.bias.copy_(vector(b))
            model.heat.weight.copy_(weight("heat.weight"))
            model.heat.bias.copy_(vector("heat.bias"))
            model.presence_head.weight.copy_(weight(head[0]).T)  # MatMul: x @ W
            model.presence_head.bias.copy_(vector("presence_head.bias"))
        model.gain = math.exp(init["log_gain"].data[0])
        return model


def load_image(path: Path, size: int) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return data.reshape(size, size, 3).permute(2, 0, 1).float() / 255.0


class _Images(Dataset):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return load_image(self.paths[i], self.size)


class _Noise(Dataset):
    def __init__(self, n: int, size: int) -> None:
        self.n, self.size = n, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.rand(3, self.size, self.size, generator=torch.Generator().manual_seed(i))


def _decode(heat: torch.Tensor, presence: torch.Tensor, gain: float) -> tuple[list[float], bool]:
    box = soft_argmax_xyxy(heat.float(), gain=gain)[0].tolist()
    return box, bool(torch.sigmoid(presence.float()).item() >= 0.5)


def evaluate(run, gain: float, size: int, positives, negatives, px: int = 480) -> dict:  # noqa: ANN001
    """Center error (px at ``px`` scale) on positives and presence accuracy over all images."""
    errs, correct = [], 0
    for path, box in positives:
        (x1, y1, x2, y2), present = _decode(*run(load_image(path, size)[None]), gain)
        errs.append(
            math.hypot((x1 + x2 - box.x1 - box.x2) / 2, (y1 + y2 - box.y1 - box.y2) / 2) * px
        )
        correct += present
    for path in negatives:
        correct += not _decode(*run(load_image(path, size)[None]), gain)[1]
    errs.sort()
    n = len(positives) + len(negatives)
    return {
        "center_err_px_mean": round(sum(errs) / len(errs), 2),
        "center_err_px_median": round(errs[len(errs) // 2], 2),
        "presence_acc": round(correct / n, 4),
        "images": n,
    }


def write_frames(run, gain: float, args, positives, negatives) -> None:  # noqa: ANN001
    """Raw RGB frames plus what the desktop int8 graph says about each, for the firmware."""
    out = Path(args.frames_out)
    out.mkdir(parents=True, exist_ok=True)
    n_neg = min(len(negatives), max(1, args.frames // 4))
    n_pos = args.frames - n_neg
    picks = [p for p, _ in positives[:: max(1, len(positives) // n_pos)][:n_pos]]
    picks += negatives[:: max(1, len(negatives) // n_neg)][:n_neg]
    lines = [f"// generated by export_espdl.py: size {args.size}, gain {gain!r}"]
    lines.append(f"#define DETECT_SIZE {args.size}")
    lines.append(f"#define DETECT_GAIN {gain!r}f")
    for i, path in enumerate(picks):
        chw = load_image(path, args.size)
        rgb = chw.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).contiguous()
        (out / f"frame_{i}.rgb").write_bytes(rgb.numpy().tobytes())
        heat, presence = run(chw[None])
        box, _ = _decode(heat, presence, gain)
        vals = ", ".join(f"{v:.6f}f" for v in [*box, float(presence.item())])
        lines.append(f"FRAME(frame_{i}, {vals})  // {path.name}")
    (out / "frames.inc").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, default=160, help="square input size (px)")
    ap.add_argument("--width", type=int, default=16, help="channel width (random weights only)")
    ap.add_argument("--onnx", default=None, help="tensor-factory int8 ONNX to take weights from")
    ap.add_argument("--calib", default=None, help="directory of calibration images; else noise")
    ap.add_argument("--calib-steps", type=int, default=64)
    ap.add_argument("--eval", default=None, help="dataset dir (annotations.coco.json + images/)")
    ap.add_argument("--negatives", default=None, help="no-object images for --eval")
    ap.add_argument("--frames", type=int, default=0, help="embed this many --eval frames")
    ap.add_argument("--frames-out", default=None, help="directory for frame_N.rgb + frames.inc")
    ap.add_argument("--out", required=True, help="output .espdl path")
    args = ap.parse_args()

    model = (DeviceGraph.from_onnx(args.onnx) if args.onnx else DeviceGraph(args.width)).eval()

    if args.calib:
        paths = sorted(p for p in Path(args.calib).rglob("*") if p.suffix.lower() in _EXTS)
        if not paths:
            raise SystemExit(f"no images under {args.calib}")
        step = max(1, len(paths) // args.calib_steps)  # spread over the set, not its head
        data: Dataset = _Images(paths[::step][: args.calib_steps], args.size)
    else:
        data = _Noise(args.calib_steps, args.size)

    from esp_ppq.api import espdl_quantize_torch  # ty: ignore[import-not-found]
    from esp_ppq.executor.torch import TorchExecutor  # ty: ignore[import-not-found]

    graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=args.out,
        calib_dataloader=DataLoader(data, batch_size=1, shuffle=False),
        calib_steps=len(data),
        input_shape=[1, 3, args.size, args.size],
        target="esp32p4",
        num_of_bits=8,
        device="cpu",
        error_report=False,
        export_test_values=True,  # lets the firmware check its output against this export
        verbose=0,
    )
    meta = {"size": args.size, "gain": model.gain, "source": args.onnx or "random"}
    report: dict = {"meta": meta}
    if args.eval:
        root = Path(args.eval)
        positives = load_coco_boxes(root / "annotations.coco.json", root)
        negatives = load_negatives(args.negatives) if args.negatives else []
        executor = TorchExecutor(graph=graph, device="cpu")
        with torch.no_grad():
            report["float"] = evaluate(model, model.gain, args.size, positives, negatives)
            report["int8"] = evaluate(executor.forward, model.gain, args.size, positives, negatives)
        if args.frames and args.frames_out:
            write_frames(executor.forward, model.gain, args, positives, negatives)
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size} B)")
    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
