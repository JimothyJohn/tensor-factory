# Model size & capacity: the path from v5 to SBC, distilled back to MCU

How much a larger model buys tensor-factory, and a concrete versioned roadmap to the next
goal — going beyond presence/box to understand whether a helicoil is **damaged,
over-inserted, under-inserted, or correctly seated** — with a deployment ladder across
**MCU → SBC → PC** and the distillation arrows that connect them.

Grounded in the shipped **v5** model.

## Two models, not one

This is **two distinct products**, deliberately separated:

1. **Basic object detection — shipping today.** v5: box + presence. Finds the helicoil and
   says present/absent. Trained on real data, int8, bundled. *Done.*
2. **Advanced manifestation detection — future, blocked on labels.** v6 stage-2: damage
   classification + insertion measurement (over/under/correct). Architecture built and tested;
   **not trained, and cannot be, until a labeled failure-mode dataset exists.**

> **⚠ Label blocker (model 2).** No current dataset carries the labels model 2 needs — every
> image (`real_ds_combined`, `negatives_pool`) is box+presence only, all in the "ok" state:
> zero damaged/over/under examples, zero seating depths. Training the damage softmax or the
> seating regressor on single-state data is meaningless. **The gate is data, not capacity or
> architecture** — synthesize damaged/over/under coils (Gemini / Nano Banana, needs
> `GEMINI_API_KEY`) and label them first. Everything downstream (PC teacher, SBC student, MCU
> distill) waits on this one step.

---

## Where v5 leaves us (the base of the ladder)

`TinyDetector(width=16, presence=True)` — four stride-2 conv blocks → 30×30 map, soft-argmax
box head + a **single YOLO-style objectness logit** (`presence_head = nn.Linear(8·c, 1)`,
`model.py:87`). One box when `sigmoid(presence)` clears 0.5, nothing when it doesn't. No class
label, no `background` class.

- **81 KB int8, 204 fps CPU @480px** (architecture/throughput unchanged from v4).
- **18.3 px held-out box median, 79% presence acc.** Clean in-distribution separation —
  positives median 0.998, negatives 0.012.
- The mock model still hits ~1.9 px because mock geometry is exact.

Two things matter for what comes next:

1. **v5 *is* the MCU-class base model** — and simultaneously the **stage-1 region proposer**
   for the SBC model. We don't throw it away to scale up; we build around it.
2. **v5 has no state head.** The old multi-class head was removed in the YOLO rework — it's
   objectness only now. So adding damage/over/under understanding means **introducing new
   structured heads**, not extending a vestigial linear layer. That's a feature: it lets us
   put state understanding on a *crop* at high resolution instead of on a washed-out global
   pool.

---

## The key reframe: two scaling problems, not one

Box localization and state understanding scale **completely differently**.

### 1. Box precision is NOT capacity-bound

The ceiling (18.3 px on real data) is set by **loose labels** (GroundingDINO bounds the whole
boss, not the insert) and **480px resolution** (quant step already 1.88 px). Same tiny model
→ 1.9 px on exact mock geometry. A bigger backbone barely moves this.

**Order of levers for box:** tight labels > native 1024px > width. Capacity is a distant third.

### 2. State understanding IS capacity-, resolution-, and data-bound — all first-order

Distinguishing a cross-threaded coil from a clean one, or a coil seated 0.5 mm proud from one
flush, is sub-thread texture/geometry detail. The 30×30 global-pooled feature map can't carry
it. This is where capacity, resolution, and — above all — **labeled data per failure mode**
pay off.

| Lever | Box precision | State understanding |
|---|---|---|
| width 16 → 32 | small, diminishing | **large** |
| 480 → 1024px native | **large** | **large** |
| Deeper backbone (MobileNetV3 / EfficientNet-Lite) | small | **large** |
| Two-stage (detect → crop → analyze at high res) | n/a | **largest single win** |
| More labeled data per failure mode | gated by labels | **the actual ceiling** |

**Blunt version:** a 10× backbone on box error gets ~25%. The same 10× on state is the
difference between "can't do it" and "90%+". But the binding constraint on state is **data** —
278 images today. Budget for **hundreds per failure mode**, controlled lighting, or a bigger
model just overfits.

---

## Architecture principle: measure insertion, classify damage

**Over/under-insertion is a measurement, not a category.** Seating depth ≈ how many thread
turns protrude above the chamfer. A **segmentation/keypoint head** that outlines the coil and
locates the chamfer plane lets you measure protrusion geometrically — far more objective and
generalizable than a classifier guessing "over" vs "under" from monocular texture.

Reserve **classification** for *damage* (deformed / cross-threaded / broken-tang / burr),
which genuinely is a category / anomaly problem.

| Failure mode | Best-fit head | Why |
|---|---|---|
| Absent | objectness (have it in v5) | already separates 0.998 / 0.012 |
| Over / under-inserted | **segmentation + geometry** | objective, label-able, generalizes |
| Damaged (deformed, cross-thread, broken tang) | **classification / anomaly** | a category, not a measurement |

---

## The versioned roadmap

### v5 — shipped — the MCU-class base / SBC stage-1 proposer

Single-stage box + objectness, width 16, 480px, 81 KB int8. Already runs on a CPU at 204 fps;
already small enough to be the MCU target after coarsening (below). Reused as-is for stage 1.

### v6 — the target — two-stage, SBC-class state understanding

Keep v5 as **stage 1** (region proposer: box + presence). Add **stage 2**: crop the predicted
box at native/high resolution and run a dedicated head on the crop, where the detail lives.

- **Damage head (classification):** multi-way `ok / deformed / cross-threaded / broken-tang /
  burr`. This re-introduces the head v5 removed — but on the *crop*, as a small conv head, not
  the 30×30 global pool. The existing `presence_head` (mean+max pool → linear, `model.py:93`)
  is the seed pattern to grow from.
- **Insertion head (measurement):** a small segmentation or keypoint head on the crop —
  coil outline + chamfer plane → protrusion in pixels → `over / correct / under` by geometry.

Why two-stage: stage 1 stays tiny and MCU-portable; resolution is spent only on the crop;
each head trains and labels independently; stage-2 complexity can be dialed per tier.

**Backbone for stage 2 on SBC:** MobileNetV3-Large / EfficientNet-Lite0/1, or NanoDet-class
(NanoDet is **Apache-2.0** — the YOLO-class option without Ultralytics AGPL exposure). 480–640
px crops. Ballpark (not benchmarked): 30–100 fps on an SBC NPU, real-time for inspection.

### PC teacher — trained alongside v6, two jobs

A larger multi-task model — ConvNeXt-Tiny / EfficientNet-B0, 1024px+, box + segmentation +
damage class. Capacity is effectively free here. It serves as:

1. **Distillation teacher** for the SBC stage-2 student (soft targets + seg distillation).
2. **Model-in-the-loop pre-annotator** — bootstraps labels to escape the GroundingDINO label
   ceiling (the existing `TODO.md` item). The richer the teacher, the tighter the labels that
   then feed every model below it.

### MCU — distilled/coarsened down from the SBC student

Distill the SBC stage-2's knowledge into the v5-shaped single-stage model, coarsening the
multi-way damage head into **one binary `suspect` flag** (good/bad) — the same objectness
mechanism v5 already exports cleanly. Result: `presence + box + good/bad` at width 8, 128px,
int8, on an Ethos-U55 / ESP32-S3. The MCU path is literally a distilled, coarsened v5 with one
extra binary head — no new machinery on the device.

```
                         PC teacher (1024px, multi-task)
                          |                    |
            soft targets / seg distill         pre-annotated labels
                          v                    v
   v5 (stage 1) ──────►  v6  SBC two-stage (damage class + insertion seg)
   box+objectness          |
   (reuse as proposer)     │ logit/feature distill + coarsen damage→binary
                           v
                  MCU single-stage (presence + box + good/bad), width 8, 128px, int8
```

| Tier | Model | State it reports | Resolution / width | Role |
|---|---|---|---|---|
| **MCU** (ESP32-S3, Cortex-M55+Ethos-U55) | distilled v5 | present / box / good-bad (1 bit) | 128px, w8, int8 | distilled deployment |
| **SBC** (Pi 5, Orin Nano, RK3588, Coral) | **v6 two-stage** | full: damage class + insertion measure | 480–640px crops | **primary deployment** |
| **PC** (x86 + dGPU) | teacher | everything, highest acc | 1024px+ | train + distill + label |

---

## Sequenced milestones for v6

1. **Grow a crop head.** Extend `TinyDetector` (or a stage-2 module) from the `presence_head`
   pattern to a small conv head operating on the predicted-box crop.
2. **Define the label schema.** Damage classes + insertion keypoints/mask; extend the dataset
   format and loaders. (Ties into the open multi-`--data` item in `TODO.md`.)
3. **Get per-class data.** Synthesize damaged/over/under variants with Nano Banana, reusing
   `build_ds.py`'s reference conditioning so they match the real cast-aluminum application;
   backfill with real photos under controlled lighting. **This is the long pole.**
4. **Train the PC teacher**, then use it to pre-annotate (bootstrapping label quality).
5. **Train the SBC two-stage student**, distilling from the teacher.
6. **Distill + coarsen to the MCU single-stage.**

---

## TL;DR

- v5 is the base: tiny, box + objectness, no state head — and it doubles as the SBC stage-1
  proposer and the MCU distillation target.
- More capacity barely moves **box precision** (a label + resolution problem); it's essential
  for **state understanding**, but **data per failure mode is the real ceiling**.
- v6 = **two-stage**: insertion via **segmentation/measurement**, damage via
  **classification** — on the crop, at resolution.
- Ladder: **PC teacher → SBC v6 (primary) → distilled MCU**. One teacher, two students, v5
  reused at both ends.
