// Write the trained detector as the same int8 ONNX file the Python trainer exports.
//
// The graph (160-odd nodes for the soft-argmax, pooling and integer convolutions) is not
// rebuilt here. `studio/models/template-c<context>.onnx` is a real export from
// `tensor_factory_train.export_onnx`, made with width 1 so it is a few kilobytes; the
// graph does not depend on width or input size. This module replaces the template's
// weight tensors with the trained ones: batch norm folded into each conv, weights
// quantized to uint8 exactly as onnxruntime's dynamic quantizer does. Everything else in
// the file is copied byte for byte. `tests/test_browser_export.py` holds this to the
// Python exporter's output for the same weights.

import { BN_EPS, configOf } from "./ref.js";

// --- protobuf wire format: just enough to walk ModelProto -> GraphProto -> TensorProto ---

function readVarint(buf, pos) {
  let value = 0n;
  let shift = 0n;
  for (;;) {
    if (pos >= buf.length) throw new Error("onnx: truncated varint");
    const b = buf[pos++];
    value |= BigInt(b & 0x7f) << shift;
    if (!(b & 0x80)) break;
    shift += 7n;
    if (shift > 63n) throw new Error("onnx: varint too long");
  }
  return [value, pos];
}

function writeVarint(value) {
  let v = BigInt(value);
  const out = [];
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v) b |= 0x80;
    out.push(b);
  } while (v);
  return out;
}

/** Top-level fields of a message: [{field, wire, start, end, body?}] with byte ranges. */
function fields(buf) {
  const out = [];
  let pos = 0;
  while (pos < buf.length) {
    const start = pos;
    let key;
    [key, pos] = readVarint(buf, pos);
    const field = Number(key >> 3n);
    const wire = Number(key & 7n);
    let body = null;
    if (wire === 0) {
      [, pos] = readVarint(buf, pos);
    } else if (wire === 1) {
      pos += 8;
    } else if (wire === 5) {
      pos += 4;
    } else if (wire === 2) {
      let len;
      [len, pos] = readVarint(buf, pos);
      const n = Number(len);
      if (pos + n > buf.length) throw new Error("onnx: field runs past the end of the file");
      body = buf.subarray(pos, pos + n);
      pos += n;
    } else {
      throw new Error(`onnx: unsupported wire type ${wire}`);
    }
    if (pos > buf.length) throw new Error("onnx: truncated field");
    out.push({ field, wire, start, end: pos, body });
  }
  return out;
}

function concat(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out;
}

function lengthDelimited(field, body) {
  return concat([Uint8Array.from(writeVarint((field << 3) | 2)), Uint8Array.from(writeVarint(body.length)), body]);
}

const utf8 = new TextEncoder();
const fromUtf8 = new TextDecoder();
const FLOAT = 1;
const UINT8 = 2;

/** A TensorProto: dims (field 1), data_type (2), name (8), raw_data (9). */
function tensorProto(name, dims, dataType, raw) {
  const chunks = [];
  for (const d of dims) chunks.push(Uint8Array.from([0x08, ...writeVarint(d)]));
  chunks.push(Uint8Array.from([0x10, ...writeVarint(dataType)]));
  chunks.push(lengthDelimited(8, utf8.encode(name)));
  chunks.push(lengthDelimited(9, raw));
  return concat(chunks);
}

/** The graph input: ValueInfoProto "image", float, [batch, 3, size, size]. */
function imageInput(size) {
  const dim = (d) =>
    lengthDelimited(1, typeof d === "string" ? lengthDelimited(2, utf8.encode(d)) : Uint8Array.from([0x08, ...writeVarint(d)]));
  const shape = concat(["batch", 3, size, size].map(dim));
  const tensorType = concat([Uint8Array.from([0x08, FLOAT]), lengthDelimited(2, shape)]);
  return concat([lengthDelimited(1, utf8.encode("image")), lengthDelimited(2, lengthDelimited(1, tensorType))]);
}

function tensorName(body) {
  for (const f of fields(body)) if (f.field === 8 && f.body) return fromUtf8.decode(f.body);
  return "";
}

function tensorMeta(body) {
  const dims = [];
  let dataType = 0;
  for (const f of fields(body)) {
    if (f.field === 1 && f.wire === 0) dims.push(Number(readVarint(body, f.start + 1)[0]));
    else if (f.field === 1 && f.body) {
      let pos = 0;
      while (pos < f.body.length) {
        let v;
        [v, pos] = readVarint(f.body, pos);
        dims.push(Number(v));
      }
    } else if (f.field === 2) dataType = Number(readVarint(body, f.start + 1)[0]);
  }
  return { dims, dataType };
}

// --- the numbers ---

/** Round half to even, as numpy does. */
function rint(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Fold each block's batch norm into its conv: returns [{weight, bias, shape}], in order. */
export function foldBatchNorm(params) {
  const { context } = configOf(params);
  const out = [];
  for (let i = 0; i < 4 + context; i++) {
    const w = params[`features.${i}.0.weight`];
    const b = params[`features.${i}.0.bias`].data;
    const gamma = params[`features.${i}.1.weight`].data;
    const beta = params[`features.${i}.1.bias`].data;
    const mean = params[`features.${i}.1.running_mean`].data;
    const variance = params[`features.${i}.1.running_var`].data;
    const per = w.data.length / w.shape[0];
    const weight = new Float32Array(w.data.length);
    const bias = new Float32Array(w.shape[0]);
    for (let co = 0; co < w.shape[0]; co++) {
      // float32 at every step, the way torch folds: the quantizer sees float32 weights
      const scale = Math.fround(Math.fround(gamma[co]) / Math.fround(Math.sqrt(Math.fround(Math.fround(variance[co]) + Math.fround(BN_EPS)))));
      for (let j = 0; j < per; j++) weight[co * per + j] = Math.fround(w.data[co * per + j]) * scale;
      bias[co] = Math.fround(Math.fround(Math.fround(b[co]) - Math.fround(mean[co])) * scale) + Math.fround(beta[co]);
    }
    out.push({ weight, bias, shape: w.shape.slice() });
  }
  return out;
}

/**
 * uint8 quantization of a float32 weight tensor, as onnxruntime's quantize_dynamic does
 * (asymmetric, whole tensor, range widened to include zero).
 */
export function quantizeWeights(weights) {
  let lo = 0;
  let hi = 0;
  for (const v of weights) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  // onnxruntime's order, which decides the ties: the zero point comes from the float64
  // scale, the weights are divided by the float32 scale *in float32*, rounded half to
  // even, and only then shifted by the zero point.
  const exact = (hi - lo) / 255;
  const degenerate = !(exact >= 1.1754943508222875e-38);
  const scale = degenerate ? 1 : Math.fround(exact);
  const zeroPoint = degenerate ? 0 : Math.min(255, Math.max(0, rint(-lo / exact)));
  const q = new Uint8Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    q[i] = Math.min(255, Math.max(0, rint(Math.fround(weights[i] / scale)) + zeroPoint));
  }
  return { q, scale, zeroPoint };
}

function f32bytes(values) {
  return new Uint8Array(Float32Array.from(values).buffer);
}

/**
 * The trained model as int8 ONNX bytes. `template` is the bytes of
 * `studio/models/template-c<context>.onnx` for the model's context count; `size` is the
 * square input the model was trained at (the file declares it, as the Python export does).
 */
export function exportOnnx(template, params, { size } = {}) {
  if (!Number.isInteger(size) || size < 16) throw new Error(`onnx: input size ${size} is not valid`);
  const model = fields(template);
  const graphField = model.find((f) => f.field === 7 && f.body);
  if (!graphField) throw new Error("onnx: template has no graph");
  const graph = fields(graphField.body);
  const inits = graph
    .filter((f) => f.field === 5 && f.body)
    .map((f) => ({ f, name: tensorName(f.body), ...tensorMeta(f.body) }));

  const byNumber = (a, b) => Number(a.name.match(/_(\d+)/)[1]) - Number(b.name.match(/_(\d+)/)[1]);
  const convW = inits.filter((t) => /^onnx::Conv_\d+_quantized$/.test(t.name)).sort(byNumber);
  const convB = inits.filter((t) => /^onnx::Conv_\d+$/.test(t.name)).sort(byNumber);
  const folded = foldBatchNorm(params);
  if (convW.length !== folded.length || convB.length !== folded.length) {
    throw new Error(
      `onnx: template has ${convW.length} conv blocks, the model has ${folded.length} ` +
        "(use the template for this context count)",
    );
  }
  const replace = new Map();
  const quantized = (prefix, weights, dims) => {
    const { q, scale, zeroPoint } = quantizeWeights(weights);
    replace.set(`${prefix}_quantized`, tensorProto(`${prefix}_quantized`, dims, UINT8, q));
    replace.set(`${prefix}_scale`, tensorProto(`${prefix}_scale`, [], FLOAT, f32bytes([scale])));
    replace.set(`${prefix}_zero_point`, tensorProto(`${prefix}_zero_point`, [], UINT8, Uint8Array.from([zeroPoint])));
  };
  folded.forEach((blk, i) => {
    quantized(convW[i].name.replace(/_quantized$/, ""), blk.weight, blk.shape);
    replace.set(convB[i].name, tensorProto(convB[i].name, [blk.bias.length], FLOAT, f32bytes(blk.bias)));
  });
  quantized("heat.weight", Float32Array.from(params["heat.weight"].data), params["heat.weight"].shape);
  const float = (name) => {
    replace.set(name, tensorProto(name, params[name].shape, FLOAT, f32bytes(params[name].data)));
  };
  float("heat.bias");
  float("log_gain");
  float("presence_head.bias");
  // torch names the linear weight "presence_head.weight" or "..._Transposed" by version;
  // either way the graph multiplies x @ W, so it is stored transposed: [8c, 1].
  const head = inits.find((t) => /^presence_head\.weight.*_quantized$/.test(t.name));
  if (!head) throw new Error("onnx: template has no presence head");
  const pw = params["presence_head.weight"];
  quantized(head.name.replace(/_quantized$/, ""), Float32Array.from(pw.data), [pw.shape[1], pw.shape[0]]);

  const missing = [...replace.keys()].filter((name) => !inits.some((t) => t.name === name));
  if (missing.length) throw new Error(`onnx: template lacks ${missing.join(", ")}`);
  // value_info (field 13) is optional shape annotation; the template's is for width 1 and
  // would contradict the real weights, so it is dropped and the runtime infers shapes.
  const newGraph = concat(
    graph.filter((f) => f.field !== 13).map((f) => {
      if (f.field === 11 && f.body) return lengthDelimited(11, imageInput(size));
      const hit = f.field === 5 && f.body ? replace.get(tensorName(f.body)) : null;
      return hit ? lengthDelimited(5, hit) : graphField.body.subarray(f.start, f.end);
    }),
  );
  return concat(
    model.map((f) => (f === graphField ? lengthDelimited(7, newGraph) : template.subarray(f.start, f.end))),
  );
}

/** Which template file a model needs. */
export function templateName(params) {
  return `template-c${configOf(params).context}.onnx`;
}
