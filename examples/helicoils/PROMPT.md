# helicoils — Project Brief

## The challenge
A customer has challenged me to develop a method to detect helicoils (coiled-wire
threaded inserts) in machined parts.

## Goal
Build a fully simulated dataset so I can develop a library and model weights using an
**open model architecture** — preferably a lightweight CNN that runs comfortably on a
**CPU via onnxruntime** rather than requiring an expensive GPU. That constraint is what
makes the demo compelling.

## Synthetic dataset
Generate photorealistic, microscopic, isometric views of machined aluminum with holes,
where the imagery resembles the output of an extremely low-cost microscope zoomed in
heavily — the helicoil should cover roughly **80% of the frame**. Target resolution
**480×480** (or the nearest convenient resolution) to keep throughput high.

Priorities:
- Use as many open-source frameworks as possible.
- The project itself will be open-sourced, so **open weights and permissive licenses are
  critical throughout**.

## Annotation / labeling
The training feed will be a series of images acquired from a microscope, so I need a
**frontend UI** to easily label them and build the training set. I need to draw simple
bounding boxes or segmentation masks depending on which kind of dataset I'm building. The
tool should **export to the most common detection and segmentation annotation formats** to
maximize extensibility and let users import their own training sets.

## Long-term: robotic inspection
A key future capability is using a robot to locate these features and run an inspection as
a sequence of steps. That's far down the line, but I want the **image-acquisition side kept
open during inference** so the design accommodates it.

## Pipeline
1. Annotate the dataset.
2. Train an easily-reproducible model.
3. Drive it from a **CLI** — a harness that I (and Claude) can actually use, describable as
   a skill or exposed via MCP.

## Synthesis pipeline (with preconditioning)
Build a pipeline that not only synthesizes the images but also automatically derives
preconditioning values. I want to provide a **detailed prompt** and, **separately, the
features I want to extract** — yielding a full end-to-end synthesis-to-training pipeline.

During generation I need to **see samples** so I can iterate on the prompt until the images
match what I expect. After that, Claude can iterate through samples and refine the prompt
further, eventually just **randomizing seeds** to produce samples. Ideally I could feed in
**one or two reference images** as a nice-to-have, unless that's prohibitively difficult.

## Compute
Training will need far more compute than runtime — assume it runs on a heavy compute
training device or in the cloud. **Inference should be minimal**: an SBC, or ideally an MCU
(though that may not be practical). For now I need at least **10 fps** for basic detection.
I'd define "basic detection" as roughly **4 8-bit values landing within 3 pixels** of
ground truth on the 480px image — small enough to do 8-bit math. Clever optimizations like
that are the point.
