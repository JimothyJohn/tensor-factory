# Security policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for an unpatched
vulnerability.

- Use GitHub's [private vulnerability reporting](https://github.com/JimothyJohn/tensor-factory/security/advisories/new)
  (Security → Report a vulnerability), or
- email the maintainer listed on the GitHub profile.

Include what it affects, reproduction steps, and impact. We aim to acknowledge within a few
days and will coordinate a fix and disclosure timeline with you.

## Scope and threat model

tensor-factory is a library + CLI + local servers, not a hosted service. The most relevant
surfaces:

- **`tensor-factory-http`** binds `127.0.0.1` by default and is **not hardened for public
  exposure**. Treat `--host 0.0.0.0` as a deliberate choice on a trusted network. It guards
  request-body size and rejects undecodable images, but has no auth, rate limiting, or TLS.
- **Model files are untrusted input.** Loading an ONNX model executes whatever the
  onnxruntime graph describes. Only load models you trust; the bundled models are built from
  this repo's own pipeline.
- **Images are untrusted input** and are decoded with Pillow — keep it patched.
- **The in-browser demo (`docs/demo.html`)** runs entirely client-side; the uploaded image
  never leaves the browser. It fetches the onnxruntime-web runtime from a pinned CDN.

## What we care about

- No secrets in the repo or in CI logs. Secrets are runtime-resolved (`.env` locally), never
  committed.
- Input sanitized before logging; no PII logged.
- Dependencies pinned via `uv.lock`; CI runs `--locked`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CLAUDE.md`](CLAUDE.md) for the full engineering
and testing conventions that back these properties.
