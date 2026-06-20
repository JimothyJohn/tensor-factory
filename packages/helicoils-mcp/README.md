# helicoils-mcp

An MCP server that exposes the helicoil detector as tools over stdio. Ships with a
bundled int8 demo model (`helicoil-mock-v1.onnx`, trained on synthetic data) so it works
with zero setup.

## Tools

| Tool | What it does |
|---|---|
| `helicoils_detect` | Detect the helicoil in an image → normalized, pixel, and 8-bit boxes |
| `helicoils_model_info` | Resolved model path, input size, ORT providers |
| `helicoils_benchmark` | CPU inference throughput (fps) |

All tools are read-only and take an optional `model_path` to use your own ONNX model.

## Run

```bash
helicoils-mcp            # stdio MCP server
helicoils-mcp --version
```

Register in a project via `.mcp.json`:

```json
{ "mcpServers": { "helicoils": { "command": "uv", "args": ["run", "helicoils-mcp"] } } }
```

Licensed under Apache-2.0.
