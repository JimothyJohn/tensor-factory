// What surrounds the training step: resizing frames the way the Python trainer does,
// holding some out, and scoring. Pure (no DOM, no GPU) so node can test it.
//
// Resizing matters more than it looks: a model only works on pixels like the ones it was
// trained on. `resizeBilinear` is PIL's BILINEAR (a triangle filter whose support grows
// when shrinking, in PIL's 22-bit fixed point), `resizeNearest` is PIL's NEAREST;
// `tests/test_browser_resize.py` requires both to equal PIL's output byte for byte.

import { rng } from "./ref.js";

function triangleWeights(src, dst) {
  const scale = src / dst;
  const support = Math.max(scale, 1);
  const out = [];
  for (let i = 0; i < dst; i++) {
    const center = (i + 0.5) * scale;
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const hi = Math.min(Math.trunc(center + support + 0.5), src);
    const ws = new Float64Array(Math.max(hi - lo, 0));
    let total = 0;
    for (let j = lo; j < hi; j++) {
      ws[j - lo] = Math.max(0, 1 - Math.abs((j + 0.5 - center) / support));
      total += ws[j - lo];
    }
    // PIL turns the normalized weights into 22-bit fixed point and works in integers
    for (let j = 0; j < ws.length; j++) ws[j] = fixed(ws[j] / (total || 1));
    out.push({ lo, ws });
  }
  return out;
}

const FIXED_ONE = 2 ** 22;

function fixed(w) {
  return Math.trunc(w < 0 ? w * FIXED_ONE - 0.5 : w * FIXED_ONE + 0.5);
}

// A weighted sum of bytes in PIL's fixed point: start at one half, shift down, clip.
function fromFixed(acc) {
  const v = Math.floor((acc + FIXED_ONE / 2) / FIXED_ONE);
  return v <= 0 ? 0 : v >= 255 ? 255 : v;
}

function checkImage(pixels, width, height, channels) {
  if (!(width > 0 && height > 0) || pixels.length !== width * height * channels) {
    throw new Error(`resize: ${pixels.length} bytes is not a ${width}x${height}x${channels} image`);
  }
}

// One pass along x: `src` is `rows` rows of `width` pixels, `channels` bytes each; the
// result is packed RGB, `rows` rows of `xw.length` pixels.
function passHorizontal(src, width, rows, channels, xw) {
  const out = new Uint8Array(rows * xw.length * 3);
  for (let y = 0; y < rows; y++) {
    for (let ox = 0; ox < xw.length; ox++) {
      const { lo, ws } = xw[ox];
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < ws.length; k++) {
        const base = (y * width + lo + k) * channels;
        r += ws[k] * src[base];
        g += ws[k] * src[base + 1];
        b += ws[k] * src[base + 2];
      }
      const o = (y * xw.length + ox) * 3;
      out[o] = fromFixed(r);
      out[o + 1] = fromFixed(g);
      out[o + 2] = fromFixed(b);
    }
  }
  return out;
}

// One pass along y: `src` rows are `width` pixels of `channels` bytes; result is packed RGB.
function passVertical(src, width, channels, yw) {
  const out = new Uint8Array(yw.length * width * 3);
  for (let oy = 0; oy < yw.length; oy++) {
    const { lo, ws } = yw[oy];
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let k = 0; k < ws.length; k++) acc += ws[k] * src[((lo + k) * width + x) * channels + c];
        out[(oy * width + x) * 3 + c] = fromFixed(acc);
      }
    }
  }
  return out;
}

/**
 * PIL-style bilinear resize. `pixels` is `channels` bytes per pixel (3 = RGB, 4 = RGBA as
 * a canvas gives it; alpha is dropped). Returns packed RGB, `size * size * 3` bytes.
 *
 * Two passes, each rounded to a byte, so their order changes the result. Pillow (13.0)
 * goes horizontal first unless the height shrinks by more than twice what the width does;
 * the same rule is used here.
 */
export function resizeBilinear(pixels, width, height, size, channels = 3) {
  checkImage(pixels, width, height, channels);
  const xw = triangleWeights(width, size);
  const yw = triangleWeights(height, size);
  const verticalFirst = height - size > 0 && height - size > (width - size) * 2;
  if (verticalFirst) {
    return passHorizontal(passVertical(pixels, width, channels, yw), width, size, 3, xw);
  }
  return passVertical(passHorizontal(pixels, width, height, channels, xw), size, 3, yw);
}

// Source index for each output index, the way PIL's affine scaler walks them: start at
// half a step and keep adding the step. The running sum's rounding decides which source
// pixel wins when a sample point lands exactly between two, so it is reproduced as is.
function nearestIndices(src, dst) {
  const step = src / dst;
  const out = new Int32Array(dst);
  let pos = step * 0.5;
  for (let i = 0; i < dst; i++) {
    out[i] = Math.min(src - 1, Math.trunc(pos));
    pos += step;
  }
  return out;
}

/** PIL-style nearest-neighbour resize: what a runtime that can only subsample sees. */
export function resizeNearest(pixels, width, height, size, channels = 3) {
  checkImage(pixels, width, height, channels);
  const xs = nearestIndices(width, size);
  const ys = nearestIndices(height, size);
  const out = new Uint8Array(size * size * 3);
  for (let oy = 0; oy < size; oy++) {
    for (let ox = 0; ox < size; ox++) {
      const base = (ys[oy] * width + xs[ox]) * channels;
      const o = (oy * size + ox) * 3;
      out[o] = pixels[base];
      out[o + 1] = pixels[base + 1];
      out[o + 2] = pixels[base + 2];
    }
  }
  return out;
}

/** Packed RGB bytes (HWC) -> floats in [0, 1], CHW, written into `dst` at sample `index`. */
export function writeChw(dst, index, rgb, size) {
  const plane = size * size;
  const base = index * 3 * plane;
  for (let i = 0; i < plane; i++) {
    dst[base + i] = rgb[i * 3] / 255;
    dst[base + plane + i] = rgb[i * 3 + 1] / 255;
    dst[base + 2 * plane + i] = rgb[i * 3 + 2] / 255;
  }
}

/** Seeded Fisher-Yates; returns a new array. */
export function shuffled(items, rand) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/**
 * Hold `valFrac` of the items out, positives and negatives separately so neither split is
 * left without one kind. Deterministic for a seed and a set of ids, whatever order the
 * items arrive in. Items are {id, box|null}.
 */
export function split(items, valFrac, seed = 0) {
  const train = [];
  const val = [];
  for (const group of [items.filter((s) => s.box), items.filter((s) => !s.box)]) {
    const ordered = shuffled(group.slice().sort((a, b) => a.id - b.id), rng(seed));
    const hold = valFrac > 0 && ordered.length > 1 ? Math.max(1, Math.round(ordered.length * valFrac)) : 0;
    val.push(...ordered.slice(0, hold));
    train.push(...ordered.slice(hold));
  }
  return { train, val };
}

export function median(values) {
  if (!values.length) return null;
  const s = Array.from(values).sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function centerOf(b) {
  return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
}

/**
 * Median center error (px at `size`) of always predicting the mean training box: the
 * floor a real localizer has to beat. null when either split has no box.
 */
export function baselineCenterErr(train, val, size) {
  const tc = train.filter((s) => s.box).map((s) => centerOf(s.box));
  const vc = val.filter((s) => s.box).map((s) => centerOf(s.box));
  if (!tc.length || !vc.length) return null;
  const mx = tc.reduce((a, c) => a + c[0], 0) / tc.length;
  const my = tc.reduce((a, c) => a + c[1], 0) / tc.length;
  return median(vc.map(([x, y]) => Math.hypot(mx - x, my - y) * size));
}

/**
 * Score held-out predictions. `preds[i]` = {box: [x1,y1,x2,y2], presence: logit} for
 * `val[i]`. Returns {err, presenceAcc, valBoxes}: err is the median center error in px
 * over frames that have a box (null if none), presenceAcc the share of frames whose
 * present/absent call is right.
 */
export function score(val, preds, size) {
  const errs = [];
  let right = 0;
  val.forEach((s, i) => {
    const present = preds[i].presence >= 0;
    if (present === !!s.box) right++;
    if (s.box) {
      const [px, py] = centerOf(preds[i].box);
      const [tx, ty] = centerOf(s.box);
      errs.push(Math.hypot(px - tx, py - ty) * size);
    }
  });
  return {
    err: median(errs),
    presenceAcc: val.length ? right / val.length : null,
    valBoxes: errs.length,
  };
}

/** Higher is better: presence accuracy leads, then center error as a share of the image. */
export function checkpointScore({ err, presenceAcc }, size) {
  return (presenceAcc ?? 0) - (err ?? 0) / size;
}
