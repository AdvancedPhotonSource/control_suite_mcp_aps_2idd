# Control Suite MCP APS 2-ID-D

This project exposes the APS 2-ID-D MIC acquisition tools through the same
two-process pattern used by `control_suite_mcp_dummy`:

```text
MCP client -> MCP over HTTP -> FastMCP server -> ZMQ -> instrument worker
```

The FastMCP process owns HTTP/MCP request handling. The worker process owns the
beamline control objects from `s2idd_uprobe.startup` and executes Bluesky plans.
ZMQ carries JSON-serializable command requests and responses between those two
processes.

## Setup

Install this repository in an environment that can import the APS 2-ID-D
control packages:

```bash
uv sync
```

At the beamline, make sure the worker environment can import:

- `s2idd_uprobe`
- `mic_common`
- `mic_vis`
- `bluesky`
- `ophyd`

## Run

Start both processes with the launcher:

```bash
uv run control-suite-aps-2idd \
  --worker-endpoint tcp://127.0.0.1:5555 \
  --mcp-host 0.0.0.0 \
  --mcp-port 8050 \
  --mcp-path /mcp
```

Or start them manually.

Worker:

```bash
uv run control-suite-aps-2idd-worker --bind tcp://127.0.0.1:5555
```

MCP server:

```bash
uv run control-suite-aps-2idd-mcp \
  --worker tcp://127.0.0.1:5555 \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

MCP clients should connect to:

```text
http://127.0.0.1:8050/mcp
```

## Tools

- `acquire_image(width, height, x_center, y_center, stepsize_x, stepsize_y)`
- `acquire_line_scan(length, x_center, y_center, stepsize_x)`
- `set_parameters(parameters)`
- `set_config(name, value)`
- `set_attribute(name, value)`
- `get_state()`
- `health()`

This server does not import or depend on EAA packages at runtime. `acquire_image`
returns the acquisition contract keys used by
`MCPAcquireImageProxy`, including `img_path`, `array_path`, and `psize` when the
underlying APS tool successfully acquires and processes an image.
