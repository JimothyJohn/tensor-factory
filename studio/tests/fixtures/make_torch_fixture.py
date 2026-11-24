"""Ground truth for studio/js/nn: what torch computes for one small detector.

Writes ``torch_step.json`` beside this file: the initial parameters, the outputs and loss
of one training-mode forward pass, every gradient, the batch-norm running statistics after
it, the parameters after three Adam steps, and the eval-mode outputs after those steps.
The JavaScript reference must reproduce all of it (``tests/nn_ref.test.mjs``).

Regenerate (only when the model or the loss changes):

    uv run --all-packages --extra train python studio/tests/fixtures/make_torch_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from tensor_factory_train.model import TinyDetector

WIDTH, CONTEXT, SIZE, BATCH, STEPS = 4, 1, 40, 3, 3
BOX_WEIGHT, PRESENCE_WEIGHT = 5.0, 1.0


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same batch the JavaScript test builds from the same formulas."""
    n = BATCH * 3 * SIZE * SIZE
    x = ((torch.arange(n, dtype=torch.float64) * 7919) % 1000 / 1000).reshape(BATCH, 3, SIZE, SIZE)
    boxes = torch.tensor(
        [[0.10, 0.20, 0.60, 0.70], [0.0, 0.0, 0.0, 0.0], [0.35, 0.05, 0.95, 0.55]],
        dtype=torch.float64,
    )
    has_box = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    return x, boxes, has_box


def dump(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "data": [float(v) for v in t.detach().reshape(-1).tolist()]}


def loss_of(model: nn.Module, x, boxes, has_box) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box, presence = model(x)
    mask = has_box.bool()
    loss = BOX_WEIGHT * nn.SmoothL1Loss()(box[mask], boxes[mask]) + PRESENCE_WEIGHT * (
        nn.BCEWithLogitsLoss()(presence.reshape(-1), has_box)
    )
    return loss, box, presence


def main() -> None:
    torch.manual_seed(7)
    model = TinyDetector(WIDTH, presence=True, context=CONTEXT).double()
    with torch.no_grad():  # move the affine batch-norm terms and the gain off their inits
        for name, p in model.named_parameters():
            if name.endswith(("1.weight", "1.bias")) or name == "log_gain":
                p.add_(torch.randn_like(p) * 0.2)
    x, boxes, has_box = inputs()
    out: dict = {
        "config": {
            "width": WIDTH,
            "context": CONTEXT,
            "size": SIZE,
            "batch": BATCH,
            "steps": STEPS,
            "boxWeight": BOX_WEIGHT,
            "presenceWeight": PRESENCE_WEIGHT,
        },
        "params": {k: dump(v) for k, v in model.state_dict().items() if "num_batches" not in k},
    }
    model.train()
    loss, box, presence = loss_of(model, x, boxes, has_box)
    loss.backward()
    out["train"] = {"loss": float(loss), "box": dump(box), "presence": dump(presence)}
    out["grads"] = {k: dump(p.grad) for k, p in model.named_parameters()}
    out["running"] = {k: dump(v) for k, v in model.state_dict().items() if "running_" in k}

    torch.manual_seed(7)
    fresh = TinyDetector(WIDTH, presence=True, context=CONTEXT).double()
    fresh.load_state_dict(
        {k: torch.tensor(v["data"], dtype=torch.float64).reshape(v["shape"]) for k, v in out["params"].items()},
        strict=False,
    )
    opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    fresh.train()
    losses = []
    for _ in range(STEPS):
        opt.zero_grad()
        step_loss, _, _ = loss_of(fresh, x, boxes, has_box)
        step_loss.backward()
        opt.step()
        losses.append(float(step_loss))
    out["after"] = {
        "losses": losses,
        "params": {k: dump(v) for k, v in fresh.state_dict().items() if "num_batches" not in k},
    }
    fresh.eval()
    with torch.no_grad():
        box, presence = fresh(x)
    out["after"]["eval"] = {"box": dump(box), "presence": dump(presence)}
    path = Path(__file__).with_name("torch_step.json")
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path} ({path.stat().st_size} B)")


if __name__ == "__main__":
    main()
