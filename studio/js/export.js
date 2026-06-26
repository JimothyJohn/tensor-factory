// Export the labeled set to the exact layout tensor-factory-train reads:
//   <dir>/annotations.coco.json   (+ images/)  — approved positives, boxes
//   <dir>/negatives/images/                     — approved empty frames
// Uses the File System Access API (Chromium) to write a real directory, so a
// Studio session drops straight into the existing trainer with no conversion.

import { toCocoBbox } from "./codec.js";

function pad(n) {
  return String(n).padStart(5, "0");
}

/** Build the COCO dict + the lists of files to write. */
export function buildExport(frames, labelsByFrame, classes) {
  const coco = {
    images: [],
    annotations: [],
    categories: classes.map((c, i) => ({ id: i + 1, name: c.name })),
  };
  const positives = []; // {name, blob}
  const negatives = []; // {name, blob}
  let imgId = 0;
  let annId = 0;

  frames.forEach((frame) => {
    const label = labelsByFrame.get(frame.id);
    if (!label || label.review !== "approved") return;
    const name = `frame_${pad(imgId)}.png`;

    if (label.present && label.boxes.length) {
      imgId++;
      coco.images.push({
        id: imgId,
        file_name: `images/${name}`,
        width: frame.width,
        height: frame.height,
        review: "approved",
      });
      for (const box of label.boxes) {
        annId++;
        const bbox = toCocoBbox(box, frame.width, frame.height);
        coco.annotations.push({
          id: annId,
          image_id: imgId,
          category_id: (box.cls ?? 0) + 1,
          bbox,
          area: bbox[2] * bbox[3],
          iscrowd: 0,
          review: "approved",
          source: "human",
        });
      }
      positives.push({ name, blob: frame.blob });
    } else if (!label.present) {
      negatives.push({ name: `frame_${pad(negatives.length)}.png`, blob: frame.blob });
    }
  });

  return { coco, positives, negatives };
}

async function writeFile(dirHandle, name, contents) {
  const fh = await dirHandle.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(contents);
  await w.close();
}

/**
 * Prompt for a target directory and write the dataset. Returns a summary.
 * Throws if the File System Access API is unavailable.
 */
export async function exportDataset(frames, labelsByFrame, classes) {
  if (!window.showDirectoryPicker) {
    throw new Error("File System Access API unavailable — use a Chromium-based browser.");
  }
  const { coco, positives, negatives } = buildExport(frames, labelsByFrame, classes);
  const root = await window.showDirectoryPicker({ mode: "readwrite" });

  await writeFile(root, "annotations.coco.json", JSON.stringify(coco, null, 2));
  const imagesDir = await root.getDirectoryHandle("images", { create: true });
  for (const { name, blob } of positives) await writeFile(imagesDir, name, blob);

  if (negatives.length) {
    const negDir = await root.getDirectoryHandle("negatives", { create: true });
    const negImages = await negDir.getDirectoryHandle("images", { create: true });
    for (const { name, blob } of negatives) await writeFile(negImages, name, blob);
  }

  return { positives: positives.length, negatives: negatives.length, annotations: coco.annotations.length };
}
