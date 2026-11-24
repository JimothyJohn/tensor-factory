// 4×uint8 detection codec + BBox helpers — browser port of tensor_factory.codec
// / tensor_factory.geometry. A box is normalized xyxy in [0,1], top-left origin.
// Encoding each coord to one uint8 gives a ~480/255 ≈ 1.88 px step at 480 px,
// keeping round-trip error under 1 px — the repo's on-the-wire contract.

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Order + clamp a raw box into a valid normalized xyxy box. */
export function orderClamp({ x1, y1, x2, y2 }) {
  x1 = clamp01(x1);
  y1 = clamp01(y1);
  x2 = clamp01(x2);
  y2 = clamp01(y2);
  if (x2 < x1) [x1, x2] = [x2, x1];
  if (y2 < y1) [y1, y2] = [y2, y1];
  return { x1, y1, x2, y2 };
}

/** Normalized box → [x1,y1,x2,y2] as four uint8 (0–255). */
export function encodeUint8(box) {
  const b = orderClamp(box);
  return [b.x1, b.y1, b.x2, b.y2].map((v) => Math.round(clamp01(v) * 255));
}

/** Four uint8 → normalized box, repairing order + clamping. */
export function decodeUint8(values) {
  const [x1, y1, x2, y2] = values.map((v) => Math.min(255, Math.max(0, v)) / 255);
  return orderClamp({ x1, y1, x2, y2 });
}

/** Worst-case per-coordinate round-trip error in pixels. */
export function maxErrorPx(imageSize) {
  return imageSize / 255 / 2;
}

export function area(box) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

/** Normalized box → COCO [x, y, w, h] in absolute pixels. */
export function toCocoBbox(box, width, height) {
  const b = orderClamp(box);
  return [b.x1 * width, b.y1 * height, (b.x2 - b.x1) * width, (b.y2 - b.y1) * height];
}
