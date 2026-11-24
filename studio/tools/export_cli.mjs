// node studio/tools/export_cli.mjs PARAMS.json OUT.onnx SIZE
// PARAMS.json is {name: {shape, data}} with torch state_dict names. Writes the int8 ONNX
// the browser would download. Used by tests to hold the browser exporter to the Python one.
import fs from "node:fs";

import { exportOnnx, templateName } from "../js/nn/onnx.js";

const [paramsPath, outPath, size] = process.argv.slice(2);
if (!paramsPath || !outPath || !size) {
  console.error("usage: node export_cli.mjs PARAMS.json OUT.onnx SIZE");
  process.exit(2);
}
const raw = JSON.parse(fs.readFileSync(paramsPath, "utf8"));
const params = Object.fromEntries(
  Object.entries(raw).map(([k, v]) => [k, { shape: v.shape, data: Float64Array.from(v.data) }]),
);
const template = fs.readFileSync(new URL(`../models/${templateName(params)}`, import.meta.url));
fs.writeFileSync(outPath, exportOnnx(new Uint8Array(template), params, { size: Number(size) }));
