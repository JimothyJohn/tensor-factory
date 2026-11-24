// The one-line answer to "is this model actually finding the object?". Pure (no DOM).
//
// A detector that only ever emits the average box still gets a small error on centred
// data, so the error alone says nothing. The backend reports the error of that do-nothing
// predictor (`baseline`); a model that does not beat it is not localizing, however good
// its loss curve looks. And a held-out set of a couple of dozen boxes cannot rank two
// models at all, so the line says when the number is that rough.

export const FEW_BOXES = 30;

/** @returns {{tone: ""|"beats"|"loses"|"unsure", text: string}} */
export function verdict(m) {
  if (!m || typeof m.err !== "number" || typeof m.baseline !== "number") {
    return { tone: "", text: "" };
  }
  const boxes = typeof m.valBoxes === "number" ? m.valBoxes : null;
  const few = boxes != null && boxes < FEW_BOXES;
  const rough = few
    ? ` Measured on only ${boxes} held-out ${boxes === 1 ? "frame" : "frames"}, so treat it as rough.`
    : "";
  if (m.err < m.baseline) {
    const gain = m.baseline > 0 ? Math.round((1 - m.err / m.baseline) * 100) : 0;
    return {
      tone: few ? "unsure" : "beats",
      text: `Finding the object: ${gain}% closer than always guessing the average box.${rough}`,
    };
  }
  const context = m.config && typeof m.config.context === "number" ? m.config.context : null;
  const why =
    context === 0
      ? " This network only sees a small window; restart with --context 4 if the object changes size."
      : " Usually too few labels, or boxes drawn inconsistently.";
  return {
    tone: "loses",
    text: `Not finding the object yet: no better than always guessing the average box.${why}${rough}`,
  };
}

/** "224 px · 4 context · mixed resize" for the panel header. */
export function configLine(config) {
  if (!config || typeof config.size !== "number") return "";
  const parts = [`${config.size} px`];
  if (typeof config.context === "number") parts.push(`${config.context} context`);
  if (config.resize) parts.push(`${config.resize} resize`);
  return parts.join(" · ");
}
