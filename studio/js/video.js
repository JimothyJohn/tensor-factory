// Video-native ingestion. Sample a clip at N fps, dHash each candidate frame, and
// keep only frames that are novel relative to the dataset's existing dHashes (and
// to frames kept earlier in this same pass). The live, in-loop version of the
// repo's diverse-frame extractor.

import { dhash, isNovel } from "./dhash.js";

const LOAD_TIMEOUT_MS = 20000;

// Turn a MediaError into something a human can act on.
export function mediaErrorMessage(video) {
  const err = video.error;
  const codes = {
    1: "loading was aborted",
    2: "a network error occurred",
    3: "the video could not be decoded (codec unsupported by this browser)",
    4: "this video format is not supported (try MP4/H.264, WebM, or convert with ffmpeg)",
  };
  return (err && codes[err.code]) || "the browser could not open this file as a video";
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    let settled = false;
    const fail = (msg) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      URL.revokeObjectURL(video.src);
      reject(new Error(msg));
    };
    // No metadata and no error within the window: usually a format the browser is
    // silently refusing to decode. Fail loud instead of hanging forever.
    const timer = setTimeout(
      () => fail("timed out opening the video (the format may be unsupported)"),
      LOAD_TIMEOUT_MS,
    );
    video.onerror = () => fail(mediaErrorMessage(video));
    video.onloadedmetadata = () => {
      if (settled) return;
      if (!video.videoWidth || !video.videoHeight) {
        fail("the video has no visual track (audio-only or zero-size frames)");
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve(video);
    };
    video.src = URL.createObjectURL(file);
  });
}

function seek(video, t) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.onseeked = null;
      resolve();
    };
    video.onseeked = finish;
    // some containers never re-fire 'seeked' for a no-op seek — don't hang on it
    setTimeout(finish, 3000);
    video.currentTime = t;
  });
}

// Many WebM/streamed clips report duration === Infinity until the browser is
// forced to scan to the end. Coerce a finite duration before sampling.
function resolveDuration(video) {
  return new Promise((resolve) => {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      resolve(video.duration);
      return;
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.removeEventListener("durationchange", onChange);
      resolve(Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0);
    };
    const onChange = () => {
      if (Number.isFinite(video.duration) && video.duration > 0) finish();
    };
    video.addEventListener("durationchange", onChange);
    setTimeout(finish, 5000);
    video.currentTime = 1e9; // forces the browser to seek to the true end
  });
}

function toBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

/**
 * Async generator over a video file. Yields one event per sampled timestamp:
 *   { index, t, kept, total }              (skipped — too similar)
 *   { index, t, kept, total, frame }       (novel — `frame` ready to persist)
 * `frame` = {blob, dhash(BigInt), width, height, source, srcIndex}.
 *
 * @param {File} file
 * @param {{fps?:number, minDistance?:number, knownHashes?:bigint[]}} opts
 */
export async function* extractFrames(file, { fps = 1, minDistance = 6, knownHashes = [] } = {}) {
  const video = await loadVideo(file);
  const W = video.videoWidth;
  const H = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const step = 1 / fps;
  const duration = await resolveDuration(video);
  if (!duration) {
    URL.revokeObjectURL(video.src);
    throw new Error("could not determine the video's length (it may be corrupt or truncated)");
  }
  await seek(video, 0); // rewind after the duration-coercion seek
  const total = Math.max(1, Math.floor(duration / step));
  const kept = [...knownHashes];

  let index = 0;
  for (let t = 0; t < duration; t += step, index++) {
    await seek(video, t);
    ctx.drawImage(video, 0, 0, W, H);
    const h = dhash(canvas);
    if (!isNovel(h, kept, minDistance)) {
      yield { index, t, kept: false, total };
      continue;
    }
    kept.push(h);
    const blob = await toBlob(canvas);
    yield {
      index,
      t,
      kept: true,
      total,
      frame: { blob, dhash: h, width: W, height: H, source: file.name, srcIndex: index },
    };
  }
  URL.revokeObjectURL(video.src);
}
