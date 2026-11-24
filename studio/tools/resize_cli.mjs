// node studio/tools/resize_cli.mjs MODE WIDTH HEIGHT SIZE < raw RGB > raw RGB
// MODE is bilinear or nearest. Used by tests to hold the browser resize to PIL.
import fs from "node:fs";

import { resizeBilinear, resizeNearest } from "../js/nn/data.js";

const [mode, width, height, size] = process.argv.slice(2);
const fn = { bilinear: resizeBilinear, nearest: resizeNearest }[mode];
if (!fn) {
  console.error("usage: node resize_cli.mjs bilinear|nearest WIDTH HEIGHT SIZE");
  process.exit(2);
}
const input = new Uint8Array(fs.readFileSync(0));
process.stdout.write(fn(input, Number(width), Number(height), Number(size)));
