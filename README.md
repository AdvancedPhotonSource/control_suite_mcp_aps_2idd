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
- `dump_array(buffer_name)`
- `acquire_line_scan(length, x_center, y_center, stepsize_x)`
- `set_parameters(parameters)`
- `get_attribute_payload(name)`

## MCP Client Configuration

Most MCP clients use an `mcpServers` JSON object. For a client that connects to
an already-running HTTP MCP server, use:

```json
{
  "mcpServers": {
    "control-suite-aps-2idd": {
      "url": "http://127.0.0.1:8050/mcp",
      "transport": "http"
    }
  }
}
```

Start the server first with the launcher shown above. Change the host, port, or
path in the JSON if you start the server with different `--mcp-host`,
`--mcp-port`, or `--mcp-path` values.

## Tool Contract

All tool inputs and outputs are JSON-serializable. Numeric scan dimensions,
positions, and step sizes are expressed in microns unless noted otherwise.

### `acquire_image(width, height, x_center, y_center, stepsize_x, stepsize_y)`

Runs a 2D scan centered at `(x_center, y_center)` with the requested width,
height, and x/y step sizes. On a successful acquisition and image export, the
result includes:

```json
{
  "img_path": "/absolute/path/to/exported.png",
  "psize": 0.5
}
```

- `img_path` points to the exported PNG image.
- `psize` is the x step size used as the image pixel size.

The worker also updates the image buffers used by `dump_array`:

- `image_0`: first image acquired in the current worker run
- `image_km1`: image immediately before the current image
- `image_k`: most recent image

If processing or export fails, the result is:

```json
{"result": "Failed to process <file>"}
```

or:

```json
{"result": "Failed to save images for <file>"}
```

### `dump_array(buffer_name)`

Saves a buffered image as a `.npy` file. `buffer_name` must be one of the native
buffer names `image_0`, `image_km1`, or `image_k`. The result is:

```json
{
  "array_path": "/absolute/path/to/image_k_20260101_120000_000.npy"
}
```

The file path is written by the worker and must be readable by the MCP client or
its EAA process when numerical image data is loaded from disk.

### `get_attribute_payload(name)`

Returns a native acquisition or tuning attribute for logic-driven EAA adapter
calls. JSON literal values are returned as JSON. NumPy arrays are encoded as:

```json
{
  "encoding": "numpy_base64",
  "dtype": "float32",
  "shape": [256, 256],
  "data": "<base64-encoded array bytes>"
}
```

### `acquire_line_scan(length, x_center, y_center, stepsize_x)`

Moves the sample y motor to `y_center`, runs a horizontal line scan centered at
`x_center`, and exports a line-scan PNG. On success, the default result is:

```json
{"img_path": "/absolute/path/to/line-scan.png"}
```

If `line_scan_return_gaussian_fit` is enabled, the result also includes Gaussian
fit metadata when available:

```json
{
  "img_path": "/absolute/path/to/line-scan.png",
  "fwhm": 1.5,
  "a": 1.0,
  "mu": 0.0,
  "sigma": 0.6,
  "c": 0.0,
  "normalized_residual": 0.02,
  "x_min": -2.0,
  "x_max": 2.0
}
```

### `set_parameters(parameters)`

Sets beamline tuning parameters. For APS 2-ID-D, `parameters` contains the
zone-plate z target:

```json
{"parameters": [-190.0]}
```

The result is a status string such as:

```text
Moved Zone Plate z position to position: -190.0
```
