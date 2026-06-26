# Vendored libraries

The training half of Studio runs TensorFlow.js in a Web Worker. These bundles are
vendored (pinned, committed) so the app is fully self-contained — "just open it in a
browser," no runtime CDN, no network, works offline.

| File                        | Package                              | Version | License    |
| --------------------------- | ------------------------------------ | ------- | ---------- |
| `tf.min.js`                 | `@tensorflow/tfjs`                   | 4.22.0  | Apache-2.0 |
| `tf-backend-webgpu.min.js`  | `@tensorflow/tfjs-backend-webgpu`    | 4.22.0  | Apache-2.0 |

Apache-2.0 matches this repo's license — no AGPL exposure, consistent with the
project's zero-AGPL stance.

## Refresh

```sh
V=4.22.0
curl -sSL -o tf.min.js                "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@${V}/dist/tf.min.js"
curl -sSL -o tf-backend-webgpu.min.js "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@${V}/dist/tf-backend-webgpu.min.js"
```

The worker loads these via `importScripts` and selects a backend `webgpu → webgl → cpu`.
The union `tfjs` bundle already carries the cpu + webgl backends; the separate webgpu
bundle adds the `webgpu` backend.
