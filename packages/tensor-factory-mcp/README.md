# tensor-factory-mcp

An MCP server that exposes a tensor-factory detector as tools over stdio — plus a
lightweight HTTP endpoint for callers that just want JSON over HTTP. Ships with a bundled
int8 model (`helicoil-presence-v5.onnx`, real-data detector with a presence head) so it
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
curl http://127.0.0.1:8088/model_info        # resolved model + ORT providers
curl --data-binary @frame.png http://127.0.0.1:8088/detect   # raw image bytes -> detection JSON
```

`POST /detect` takes the raw image bytes as the request body and returns the same JSON as
the `tensor_factory_detect` tool. Bad image bytes → `400`; bodies over `--max-mb` → `413`.

Licensed under Apache-2.0.
