// Left-hand WASD keymap. The mouse stays in the right hand on the canvas; every
// workflow action is a key near the WASD cluster so labeling feels like a game.
// app.js supplies the action for each key; this module owns the binding table,
// dispatch, and the help text rendered in the sidebar.

export const KEYMAP = [
  { keys: ["a"], id: "prev", label: "◄ previous frame" },
  { keys: ["d"], id: "next", label: "next frame ►" },
  { keys: [" "], display: "Space", id: "accept", label: "accept auto-label, advance" },
  { keys: ["w"], id: "commit", label: "commit boxes, advance" },
  { keys: ["s"], id: "skip", label: "skip frame (leave unlabeled)" },
  { keys: ["c"], id: "negative", label: "mark empty / negative" },
  { keys: ["q"], id: "classPrev", label: "◄ class" },
  { keys: ["e"], id: "classNext", label: "class ►" },
  { keys: ["1", "2", "3", "4", "5"], display: "1–5", id: "classNum", label: "select class N" },
  { keys: ["z"], id: "undo", label: "undo last box" },
  { keys: ["x"], id: "delete", label: "delete selected box" },
  { keys: ["r"], id: "clear", label: "clear all boxes" },
  { keys: ["f"], id: "flag", label: "flag frame for review" },
  { keys: ["escape"], display: "Esc", id: "cancel", label: "cancel box being drawn" },
];

function isTyping(target) {
  const tag = target?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable;
}

/**
 * @param {Record<string, (e:KeyboardEvent, key:string) => void>} actions
 *   map of binding id → handler
 */
export function installKeymap(actions) {
  const byKey = new Map();
  for (const b of KEYMAP) {
    for (const k of b.keys) byKey.set(k, b.id);
  }
  window.addEventListener("keydown", (e) => {
    if (isTyping(e.target)) return;
    const id = byKey.get(e.key.toLowerCase());
    if (!id) return;
    const fn = actions[id];
    if (!fn) return;
    e.preventDefault();
    fn(e, e.key.toLowerCase());
  });
}

/** Render the keymap into a container as a help table. */
export function renderKeymapHelp(container) {
  container.innerHTML = "";
  for (const b of KEYMAP) {
    const row = document.createElement("div");
    row.className = "key-row";
    const kbd = document.createElement("span");
    kbd.className = "kbd";
    kbd.textContent = b.display || b.keys[0].toUpperCase();
    const desc = document.createElement("span");
    desc.className = "key-desc";
    desc.textContent = b.label;
    row.append(kbd, desc);
    container.appendChild(row);
  }
}
