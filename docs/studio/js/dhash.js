// 64-bit difference hash of the centre square — browser port of the dHash gate in
// examples/helicoils/scripts/extract_frames.py. Used to skip frames too similar to
// what's already in the labeled set, so attention is spent only on novel frames.

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function srcSize(src) {
  return {
    w: src.naturalWidth || src.videoWidth || src.width,
    h: src.naturalHeight || src.videoHeight || src.height,
  };
}

/**
 * 64-bit dHash (BigInt) of the largest centred square of `src`
 * (an ImageBitmap, <img>, <video>, or canvas). Mirrors extract_frames.py:
 * resize the centre square to 9×8 grayscale, compare adjacent horizontal pixels.
 */
export function dhash(src) {
  const { w: sw, h: sh } = srcSize(src);
  const side = Math.min(sw, sh);
  const sx = (sw - side) / 2;
  const sy = (sh - side) / 2;
  const W = 9;
  const H = 8;
  const cv = makeCanvas(W, H);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(src, sx, sy, side, side, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);
  let hash = 0n;
  let bit = 0n;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W - 1; x++) {
      const i = (y * W + x) * 4;
      const j = (y * W + x + 1) * 4;
      const gl = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const gr = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
      if (gl > gr) hash |= 1n << bit;
      bit++;
    }
  }
  return hash; // up to 64 bits
}

/** Hamming distance (number of differing bits) between two dHashes. */
export function hamming(a, b) {
  let x = a ^ b;
  let c = 0;
  while (x) {
    c += Number(x & 1n);
    x >>= 1n;
  }
  return c;
}

/** True if `h` is at least `minDistance` bits away from every hash in `known`. */
export function isNovel(h, known, minDistance) {
  for (const k of known) {
    if (hamming(h, k) <= minDistance) return false;
  }
  return true;
}
