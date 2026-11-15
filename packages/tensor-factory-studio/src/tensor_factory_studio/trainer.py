"""Continuous background trainer.

A daemon thread that retrains the tiny detector whenever new labels arrive, reusing
``tensor_factory_train.fit`` wholesale (so it produces the same int8 ONNX the rest of the
repo trusts) and streaming per-epoch metrics through fit's ``on_epoch`` hook. Keep-best
guardrail: a round's checkpoint is only *promoted* to the served model if its best val
center-error beats the global best; otherwise the previous served model stays live and the
round is flagged a regression with the likely-bad (most-recently-added) sample ids.

torch is imported lazily in :meth:`run` so importing this module needs no ``serve`` extra.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from .dataset import Dataset


def _f(v) -> float | None:
    """Coerce numpy/torch scalars to a JSON-serializable Python float (or None)."""
    return None if v is None else float(v)


class Trainer(threading.Thread):
    def __init__(
        self,
        dataset: Dataset,
        models_dir: str | Path,
        *,
        size: int = 480,
        width: int = 16,
        epochs: int = 20,
        batch: int = 16,
        min_positives: int = 4,
        device: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.dataset = dataset
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.width = width
        self.epochs = epochs
        self.batch = batch
        self.min_positives = min_positives
        self._device_pref = device

        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._stop = threading.Event()
        self.paused = False

        self.device = "?"
        self.version = 0
        self.served: Path | None = None
        self.global_best = float("inf")
        self.ids_at_best: set[int] = set()
        self.history: list[float] = []
        self._metrics: dict = {"phase": "starting", "status": "warming up"}

    # --- public API (called from the HTTP handler) ---
    def mark_dirty(self) -> None:
        self._dirty.set()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._dirty.set()

    def stop(self) -> None:
        self._stop.set()
        self._dirty.set()

    def reset(self) -> None:
        with self._lock:
            self.version = 0
            self.served = None
            self.global_best = float("inf")
            self.ids_at_best = set()
            self.history = []
            self._metrics = {"phase": "idle", "status": "session cleared"}

    def metrics(self) -> dict:
        with self._lock:
            m = dict(self._metrics)
            m["backend"] = self.device
            m["running"] = not self.paused
            m["bestErr"] = None if self.global_best == float("inf") else float(self.global_best)
            m["history"] = list(self.history)
            m["hasModel"] = self.served is not None
            return m

    # --- thread body ---
    def run(self) -> None:
        from tensor_factory_train.train import fit, resolve_device  # heavy (torch)

        self.device = self._device_pref or resolve_device()
        self._status(f"idle on {self.device}")
        while not self._stop.is_set():
            # Block until new labels arrive. Only train when actually dirty -- a bare
            # wait()+train would retrain every second forever, saturating the GIL during
            # ONNX export and starving the HTTP server.
            if not self._dirty.wait(timeout=1.0):
                continue
            if self._stop.is_set():
                break
            if self.paused:
                continue
            self._dirty.clear()
            c = self.dataset.counts()
            if c["positives"] < self.min_positives:
                self._status(f"need ≥{self.min_positives} positives ({c['positives']} now)")
                continue
            self._round(fit)

    def _round(self, fit) -> None:
        round_out = self.models_dir / "_round.onnx"
        round_best = {"err": float("inf")}

        def on_epoch(m: dict) -> None:
            with self._lock:
                self._metrics = self._shape(m)
                if m["val_err"] is not None:
                    self.history.append(float(m["val_err"]))
                    del self.history[:-120]
            if m["is_best"] and m["best_err"] is not None:
                round_best["err"] = float(m["best_err"])

        try:
            path = fit(
                self.dataset.root,
                round_out,
                epochs=self.epochs,
                batch=self.batch,
                size=self.size,
                width=self.width,
                device=self._device_pref,
                presence=True,
                val_frac=0.2,
                negatives=[self.dataset.root / "negatives"],
                on_epoch=on_epoch,
            )
        except Exception as exc:  # noqa: BLE001 -- surface to the UI, keep the thread alive
            self._status(f"train error: {type(exc).__name__}: {exc}")
            return

        with self._lock:
            err = round_best["err"]
            regressed = False
            if err < self.global_best:
                self.version += 1
                served = self.models_dir / f"served-v{self.version}.onnx"
                shutil.copyfile(path, served)
                self._prune(self.version)
                self.served = served
                self.global_best = err
                self.ids_at_best = self.dataset.ids()
            elif self.global_best < float("inf"):
                regressed = True
            self._metrics["phase"] = "idle"
            self._metrics["regressed"] = regressed
            self._metrics["suspects"] = (
                [{"frameId": i} for i in self.dataset.recent(self.ids_at_best)] if regressed else []
            )

    def _prune(self, keep_version: int) -> None:
        for p in self.models_dir.glob("served-v*.onnx"):
            try:
                v = int(p.stem.split("-v")[1])
            except (IndexError, ValueError):
                continue
            if v < keep_version - 1:
                p.unlink(missing_ok=True)

    def _shape(self, m: dict) -> dict:
        return {
            "phase": "training",
            "epoch": int(m["epoch"]),
            "epochs": int(m["epochs"]),
            "loss": _f(m["loss"]),
            "err": _f(m["val_err"]),
            "baseline": _f(m["baseline"]),
            "presenceAcc": _f(m["presence_acc"]),
            "classAcc": _f(m.get("class_acc")),
            "numClasses": int(m.get("num_classes", 1)),
            "gain": _f(m["gain"]),
            "trainCount": int(m["train_count"]),
            "valCount": int(m["val_count"]),
            "regressed": False,
            "suspects": [],
        }

    def _status(self, text: str) -> None:
        with self._lock:
            self._metrics = {"phase": "idle", "status": text}
