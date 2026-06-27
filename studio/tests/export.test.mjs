import assert from "node:assert/strict";
import { test } from "node:test";

import { buildExport } from "../js/export.js";

const classes = [{ name: "object", color: "#0f0" }];

function frames() {
  return [
    { id: 1, width: 480, height: 480, blob: "A" },
    { id: 2, width: 480, height: 480, blob: "B" },
    { id: 3, width: 480, height: 480, blob: "C" },
    { id: 4, width: 480, height: 480, blob: "D" },
  ];
}

test("buildExport routes approved positives to COCO, negatives aside, drops the rest", () => {
  const labels = new Map([
    [1, { frameId: 1, review: "approved", present: true, boxes: [{ x1: 0.1, y1: 0.1, x2: 0.5, y2: 0.5, cls: 0 }] }],
    [2, { frameId: 2, review: "approved", present: false, boxes: [] }],
    [3, { frameId: 3, review: "pending", present: true, boxes: [{ x1: 0, y1: 0, x2: 1, y2: 1, cls: 0 }] }],
    // frame 4: no label at all
  ]);
  const { coco, positives, negatives } = buildExport(frames(), labels, classes);

  assert.equal(coco.images.length, 1);
  assert.equal(coco.annotations.length, 1);
  assert.equal(coco.images[0].file_name, "images/frame_00000.png");
  assert.equal(coco.images[0].review, "approved");
  assert.equal(coco.annotations[0].review, "approved");
  assert.equal(coco.annotations[0].source, "human");
  assert.equal(coco.annotations[0].category_id, 1);
  assert.deepEqual(coco.categories, [{ id: 1, name: "object" }]);
  assert.equal(positives.length, 1);
  assert.equal(positives[0].blob, "A");
  assert.equal(negatives.length, 1);
  assert.equal(negatives[0].blob, "B");
  assert.equal(negatives[0].name, "frame_00000.png");
});

test("buildExport bbox math is correct pixels", () => {
  const labels = new Map([
    [1, { frameId: 1, review: "approved", present: true, boxes: [{ x1: 0.25, y1: 0.5, x2: 0.75, y2: 1, cls: 0 }] }],
  ]);
  const { coco } = buildExport([{ id: 1, width: 480, height: 480, blob: "A" }], labels, classes);
  assert.deepEqual(coco.annotations[0].bbox, [120, 240, 240, 240]);
});

test("buildExport multi-box positive emits one annotation per box", () => {
  const labels = new Map([
    [
      1,
      {
        frameId: 1,
        review: "approved",
        present: true,
        boxes: [
          { x1: 0, y1: 0, x2: 0.2, y2: 0.2, cls: 0 },
          { x1: 0.5, y1: 0.5, x2: 0.6, y2: 0.6, cls: 0 },
        ],
      },
    ],
  ]);
  const { coco } = buildExport([{ id: 1, width: 100, height: 100, blob: "A" }], labels, classes);
  assert.equal(coco.images.length, 1);
  assert.equal(coco.annotations.length, 2);
  assert.deepEqual(
    coco.annotations.map((a) => a.image_id),
    [1, 1],
  );
});

test("buildExport empty input yields empty COCO with categories", () => {
  const { coco, positives, negatives } = buildExport([], new Map(), classes);
  assert.deepEqual(coco.images, []);
  assert.deepEqual(coco.annotations, []);
  assert.equal(coco.categories.length, 1);
  assert.equal(positives.length + negatives.length, 0);
});
