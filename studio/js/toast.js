// Minimal, dependency-free toast notifications so the user always knows what is and
// isn't working. info/success auto-dismiss; errors stick until dismissed; progress()
// returns a handle you update in place then resolve to success/error. Identical
// fire-and-forget messages are de-duped so a failing poll loop can't spam the screen.

const AUTO_MS = { info: 4000, success: 4500, error: 0, progress: 0 }; // 0 = sticky

let container = null;
const deduped = new Map(); // message -> {el, count, countEl}

function ensureContainer() {
  if (!container) {
    container = document.createElement("div");
    container.id = "toasts";
    document.body.appendChild(container);
  }
  return container;
}

function makeToast(kind, message) {
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  const text = document.createElement("span");
  text.className = "toast-text";
  text.textContent = message;
  const countEl = document.createElement("span");
  countEl.className = "toast-count";
  countEl.hidden = true;
  const close = document.createElement("button");
  close.className = "toast-close";
  close.textContent = "×";
  close.setAttribute("aria-label", "dismiss");
  el.append(text, countEl, close);
  ensureContainer().appendChild(el);
  return { el, text, countEl, close };
}

function remove(el) {
  el.classList.add("leaving");
  setTimeout(() => el.remove(), 200);
}

// Fire-and-forget toast, de-duped by message.
function flash(kind, message) {
  const hit = deduped.get(message);
  if (hit) {
    hit.count += 1;
    hit.countEl.textContent = `×${hit.count}`;
    hit.countEl.hidden = false;
    return hit.el;
  }
  const t = makeToast(kind, message);
  const rec = { el: t.el, count: 1, countEl: t.countEl };
  deduped.set(message, rec);
  const drop = () => {
    deduped.delete(message);
    remove(t.el);
  };
  t.close.onclick = drop;
  const ms = AUTO_MS[kind] ?? 4000;
  if (ms > 0) setTimeout(drop, ms);
  return t.el;
}

export const toast = {
  info: (m) => flash("info", m),
  success: (m) => flash("success", m),
  error: (m) => flash("error", m),
  /**
   * A sticky toast you update in place (not de-duped). Returns a handle:
   *   p.update(msg)                — change the text
   *   p.success(msg) / p.error(msg)— finalize (success auto-dismisses, error sticks)
   *   p.close()                    — remove now
   */
  progress(message) {
    const t = makeToast("progress", message);
    t.close.onclick = () => remove(t.el);
    return {
      el: t.el,
      update: (msg) => {
        t.text.textContent = msg;
      },
      success: (msg) => {
        t.text.textContent = msg;
        t.el.className = "toast toast-success";
        setTimeout(() => remove(t.el), AUTO_MS.success);
      },
      error: (msg) => {
        t.text.textContent = msg;
        t.el.className = "toast toast-error";
      },
      close: () => remove(t.el),
    };
  },
};
