# tensor-factory-mcp

An MCP server that exposes a tensor-factory detector as tools over stdio. Ships with a
bundled int8 demo model (`helicoil-mock-v1.onnx`, trained on synthetic data) so it works
with zero setup.

## Tools

| Tool | What it does |
|---|---|
| `tensor_factory_detect` | Detect the target object in an image → normalized, pixel, and 8-bit boxes |
| `tensor_factory_model_info` | Resolved model path, input size, ORT providers |
| `tensor_factory_benchmark` | CPU inference throughput (fps) |

All tools are read-only and take an optional `model_path` to use your own ONNX model.

## Run

```bash
tensor-factory-mcp            # stdio MCP server
tensor-factory-mcp --version
```

Register in a project via `.mcp.json`:

```json
{ "mcpServers": { "tensor-factory": { "command": "uv", "args": ["run", "tensor-factory-mcp"] } } }
```

Licensed under Apache-2.0.
