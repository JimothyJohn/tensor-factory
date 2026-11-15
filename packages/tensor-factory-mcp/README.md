# tensor-factory-mcp

An MCP server that exposes a tensor-factory detector as tools over stdio — plus a
lightweight HTTP endpoint for callers that just want JSON over HTTP. Ships with a bundled
int8 model (`helicoil-presence-cam-v1.onnx`, real-camera detector with a presence head) so it
works with zero setup; the synthetic box-only demo (`helicoil-mock-v1.onnx`) is bundled
too and selectable via `model_path`.

## Tools

| Tool | What it does |
|---|---|
| `tensor_factory_detect` | Detect the target in an image → normalized, pixel, and 8-bit boxes |
| `tensor_factory_model_info` | Resolved model path, input size, ORT providers |
| `tensor_factory_benchmark` | CPU inference throughput (fps) |

All tools are read-only and take an optional `model_path` to use your own ONNX model.

A model trained with a **presence head** (`tensor-factory-train --negatives`) also makes
`detect` return `present` (bool) and `score` (the objectness probability). It's YOLO-style:
when `present` is `false` the target is absent and the box fields (`box_norm`, `box_pixels`,
`uint8`) come back `null` — one box or no box at all. Box-only models omit `present`/`score`
and always return a box.

## Run

```bash
tensor-factory-mcp            # stdio MCP server
tensor-factory-mcp --version
```

Register in a project via `.mcp.json`:

```json
{ "mcpServers": { "tensor-factory": { "command": "uv", "args": ["run", "tensor-factory-mcp"] } } }
```

## HTTP endpoint

When MCP is heavier than you need, `tensor-factory-http` serves the same detector over
plain HTTP (stdlib `http.server`, no extra deps). Binds `127.0.0.1` by default — pass
`--host 0.0.0.0` to expose it deliberately.

```bash
tensor-factory-http --port 8088              # then:
curl http://127.0.0.1:8088/health            # {"status": "ok"}
curl http://127.0.0.1:8088/model_info        # resolved model + sha256, engine, providers
curl --data-binary @frame.png http://127.0.0.1:8088/detect   # raw image bytes -> detection JSON
```

`POST /detect` takes the raw image bytes as the request body and returns the same JSON as
the `tensor_factory_detect` tool. Bad image bytes → `400`; bodies over `--max-mb` → `413`.

**Swap the model without a restart.** The server re-checks the model file on every
request. Replace it atomically and the next request runs the new model:

```bash
scp new.onnx device:model.onnx.incoming && ssh device mv model.onnx.incoming model.onnx
curl http://device:8088/model_info           # "sha256" is now the new file's
```

If the replacement does not load (half-copied, corrupt), the model that was working keeps
serving and `/model_info` carries a `reload_error` saying why.

## Edge: the whole deployment is one file

`tensor-factory-bundle` packs the runtime, the HTTP server and a model into a single
runnable `.pyz` (about 100 KB with the bundled detector). The device needs CPython ≥ 3.11
and nothing else: no pip, no venv, no wheels.

```bash
tensor-factory-bundle --model model.onnx --out edge.pyz   # on your machine
scp edge.pyz device:                                      # the deployment
ssh device python3 edge.pyz --host 0.0.0.0 --port 8088    # serves POST /detect
```

- **Engine.** If onnxruntime is installed on the device it is used (~200 fps on a laptop
  CPU); otherwise the bundled pure-Python engine runs (~1 fps). Same JSON either way.
- **Verify.** The build prints the model's sha256; `/model_info` on the device reports the
  sha256 of what is actually loaded. They must match.
- **Flags.** Every `tensor-factory-http` flag passes through. `--model PATH` serves a file
  outside the bundle, which is the one to use with the hot swap above.
- **Reproducible.** The same sources and model produce a byte-identical bundle.
- **Proven on ARM.** CI builds a bundle and serves a detection from it on a bare
  `python:3.12-slim` container on both x86-64 and aarch64 Linux.

To keep it running across reboots on a systemd device, a user unit is enough (no root):

```ini
# ~/.config/systemd/user/tensor-factory.service
[Unit]
Description=tensor-factory detector

[Service]
ExecStart=/usr/bin/python3 %h/edge.pyz --host 0.0.0.0 --port 8088
Restart=on-failure

[Install]
WantedBy=default.target
```

`systemctl --user enable --now tensor-factory`, and `loginctl enable-linger $USER` so it
starts without a login. The endpoint has no authentication: bind it to a network you trust.

Licensed under Apache-2.0.
