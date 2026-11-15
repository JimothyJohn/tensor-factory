// Per-source rows for the trainer panel. Pure (no DOM) so it is unit-testable in node.
//
// The backend reports one entry per data source each epoch:
//   {name, train, val, err|null, presenceAcc|null, classAcc|null}
// A source with no held-out box (a negatives pool) has err null: shown as a dash, never
// as 0, which would read as a perfect score.

const dash = "—";

/** @returns {{cells: string[], tone: ""|"beats"|"loses"}[]} cells = name, train, val, err, presence */
export function sourceRows(sources, baseline) {
  if (!Array.isArray(sources)) return [];
  return sources.map((s) => {
    const err = typeof s.err === "number" ? s.err : null;
    return {
      cells: [
        String(s.name ?? "?"),
        String(s.train ?? 0),
        String(s.val ?? 0),
        err == null ? dash : `${err.toFixed(1)}px`,
        typeof s.presenceAcc === "number" ? `${(s.presenceAcc * 100).toFixed(0)}%` : dash,
      ],
      // same verdict as the headline number: does this source beat the constant predictor?
      tone: err == null || baseline == null ? "" : err < baseline ? "beats" : "loses",
    };
  });
}
