// Bounding-box canvas editor. The mouse (right hand) draws and selects boxes; all
// workflow actions live on the keyboard (left hand) and are driven from app.js.
// Boxes are stored normalized (xyxy in [0,1]); the editor handles the letterboxed
// mapping between normalized space and on-screen pixels.

import { orderClamp, clamp01 } from "./codec.js";

const MIN_SIDE = 0.01; // discard accidental click-boxes smaller than this (normalized)

export class CanvasEditor {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {() => {index:number, color:string}} getActiveClass  active class accessor
   * @param {() => void} onChange  called whenever boxes change (for autosave)
   */
  constructor(canvas, getActiveClass, onChange) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.getActiveClass = getActiveClass;
    this.onChange = onChange;
    this.classes = []; // [{name, color}] — set via setClasses

    this.bitmap = null; // current frame (ImageBitmap | HTMLImageElement)
    this.boxes = []; // [{x1,y1,x2,y2,cls}] normalized
    this.selected = -1;
    this.drawing = null; // {x1,y1,x2,y2} in-progress, normalized
    this.cursorPx = null; // {x,y} in canvas-buffer px, for the crosshair guides

    canvas.addEventListener("mousedown", (e) => this._down(e));
    canvas.addEventListener("mousemove", (e) => this._move(e));
    canvas.addEventListener("mouseleave", () => {
      this.cursorPx = null;
      this.render();
    });
    window.addEventListener("mouseup", (e) => this._up(e));
  }

  /**
   * Match the canvas drawing buffer to its displayed size so a resized preview stays
   * crisp (and the crosshair/box math stays pixel-accurate). Called by a ResizeObserver.
   */
  fitToDisplay() {
    const w = Math.round(this.canvas.clientWidth) || this.canvas.width;
    const h = Math.round(this.canvas.clientHeight) || this.canvas.height;
    if (w && h && (this.canvas.width !== w || this.canvas.height !== h)) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.render();
  }

  setClasses(classes) {
    this.classes = classes;
  }

  setFrame(bitmap, boxes) {
    this.bitmap = bitmap;
    this.boxes = boxes.map((b) => ({ ...b }));
    this.selected = -1;
    this.drawing = null;
    this.render();
  }

  getBoxes() {
    return this.boxes.map((b) => ({ ...b }));
  }

  // --- normalized <-> screen mapping (image fit inside canvas, letterboxed) ---
  _rect() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    if (!this.bitmap) return { ox: 0, oy: 0, dw: cw, dh: ch };
    const iw = this.bitmap.width;
    const ih = this.bitmap.height;
    const scale = Math.min(cw / iw, ch / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh };
  }

  // client (CSS) coords -> canvas-buffer px (works at any display size)
  _clientToCanvas(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: ((clientX - r.left) / r.width) * this.canvas.width,
      y: ((clientY - r.top) / r.height) * this.canvas.height,
    };
  }

  _toNorm(clientX, clientY) {
    const { x: px, y: py } = this._clientToCanvas(clientX, clientY);
    const { ox, oy, dw, dh } = this._rect();
    return { x: clamp01((px - ox) / dw), y: clamp01((py - oy) / dh) };
  }

  _hit(nx, ny) {
    // topmost box containing the point
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      const b = this.boxes[i];
      if (nx >= b.x1 && nx <= b.x2 && ny >= b.y1 && ny <= b.y2) return i;
    }
    return -1;
  }

  _down(e) {
    if (!this.bitmap) return;
    const { x, y } = this._toNorm(e.clientX, e.clientY);
    const hit = this._hit(x, y);
    if (hit >= 0) {
      this.selected = hit;
      this.render();
      return;
    }
    this.selected = -1;
    this.drawing = { x1: x, y1: y, x2: x, y2: y };
  }

  _move(e) {
    // Track the cursor for the crosshair guides whether or not we're drawing.
    this.cursorPx = this._clientToCanvas(e.clientX, e.clientY);
    if (this.drawing) {
      const { x, y } = this._toNorm(e.clientX, e.clientY);
      this.drawing.x2 = x;
      this.drawing.y2 = y;
    }
    this.render();
  }

  _up() {
    if (!this.drawing) return;
    const box = orderClamp(this.drawing);
    this.drawing = null;
    if (box.x2 - box.x1 >= MIN_SIDE && box.y2 - box.y1 >= MIN_SIDE) {
      box.cls = this.getActiveClass().index;
      this.boxes.push(box);
      this.selected = this.boxes.length - 1;
      this.onChange();
    }
    this.render();
  }

  // --- keyboard-driven edits (called from app.js) ---
  undo() {
    if (this.boxes.length) {
      this.boxes.pop();
      this.selected = -1;
      this.onChange();
      this.render();
    }
  }

  deleteSelected() {
    const i = this.selected >= 0 ? this.selected : this.boxes.length - 1;
    if (i >= 0) {
      this.boxes.splice(i, 1);
      this.selected = -1;
      this.onChange();
      this.render();
    }
  }

  clearAll() {
    if (this.boxes.length) {
      this.boxes = [];
      this.selected = -1;
      this.onChange();
      this.render();
    }
  }

  cancelDraw() {
    if (this.drawing) {
      this.drawing = null;
      this.render();
    }
  }

  /** Reassign the selected box (or the last box) to the active class. */
  setSelectedClass(clsIndex) {
    const i = this.selected >= 0 ? this.selected : this.boxes.length - 1;
    if (i >= 0) {
      this.boxes[i].cls = clsIndex;
      this.onChange();
      this.render();
    }
  }

  _color(clsIndex) {
    const c = this.classes[clsIndex];
    return c ? c.color : "#4ade80";
  }

  render() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, cw, ch);
    if (!this.bitmap) return;

    const { ox, oy, dw, dh } = this._rect();
    ctx.drawImage(this.bitmap, ox, oy, dw, dh);

    const drawBox = (b, color, selected) => {
      const x = ox + b.x1 * dw;
      const y = oy + b.y1 * dh;
      const w = (b.x2 - b.x1) * dw;
      const h = (b.y2 - b.y1) * dh;
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      if (selected) {
        ctx.fillStyle = color + "22";
        ctx.fillRect(x, y, w, h);
      }
    };

    this.boxes.forEach((b, i) => drawBox(b, this._color(b.cls), i === this.selected));
    if (this.drawing) {
      drawBox(orderClamp(this.drawing), this.getActiveClass().color, true);
    }
    this._drawCrosshair(ox, oy, dw, dh);
  }

  // Faint full-image vertical + horizontal guides at the cursor, so the user can see
  // exactly where an edge will land. Only drawn while the cursor is over the image.
  _drawCrosshair(ox, oy, dw, dh) {
    const c = this.cursorPx;
    if (!c) return;
    if (c.x < ox || c.x > ox + dw || c.y < oy || c.y > oy + dh) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineWidth = 1;
    // dark underlay then light line, so the guides read on both bright and dark frames
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    line(ctx, c.x + 0.5, oy, c.x + 0.5, oy + dh);
    line(ctx, ox, c.y + 0.5, ox + dw, c.y + 0.5);
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.setLineDash([4, 4]);
    line(ctx, c.x + 0.5, oy, c.x + 0.5, oy + dh);
    line(ctx, ox, c.y + 0.5, ox + dw, c.y + 0.5);
    ctx.restore();
  }
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
