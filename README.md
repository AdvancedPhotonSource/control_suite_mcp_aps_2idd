# Control Suite MCP APS 2-ID-D

This project exposes APS 2-ID-D MIC controls through FastMCP over HTTP and
executes beamline actions through Bluesky QueueServer.

## Architecture

```text
MCP client -> FastMCP HTTP server -> REManagerAPI -> Bluesky QueueServer
```

The MCP service is a thin allowlisted control layer over QueueServer. It does
not run a local RunEngine and it does not expose arbitrary QueueServer plans or
functions.

## Setup

Install the package into the target environment:

```bash
pip install -e .
```

The environment needs:

- `fastmcp`
- `bluesky-queueserver-api`

It does not require the old local worker/ZMQ stack.

## Run

Start the MCP server and point it at QueueServer:

```bash
control-suite-aps-2idd-mcp \
  --qserver-control-addr tcp://sec2idd.xray.aps.anl.gov:60615 \
  --qserver-info-addr tcp://sec2idd.xray.aps.anl.gov:60625 \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

If QueueServer is running on the same host as the MCP server, replace
`sec2idd.xray.aps.anl.gov` with `127.0.0.1`.

If line scans need sample-y motion or `set_parameters()` needs zp-z motion,
also configure approved QueueServer helper functions:

```bash
control-suite-aps-2idd-mcp \
  --qserver-control-addr tcp://sec2idd.xray.aps.anl.gov:60615 \
  --qserver-info-addr tcp://sec2idd.xray.aps.anl.gov:60625 \
  --qserver-move-samy-function YOUR_MOVE_SAMY_FUNCTION \
  --qserver-set-zp-z-function YOUR_SET_ZP_Z_FUNCTION \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

## MCP URL

Clients should connect to:

```text
http://127.0.0.1:8050/mcp
```

## Tools

- `health()`
- `get_state()`
- `acquire_image(width, height, x_center, y_center, stepsize_x, stepsize_y)`
- `dump_array(buffer_name)`
- `acquire_line_scan(length, x_center, y_center, stepsize_x)`
- `set_parameters(parameters)`
- `get_attribute_payload(name)`

`dump_array()` intentionally returns an error in this QServer-only design,
because the MCP service does not own in-process image buffers.

## MCP Client Configuration

For an HTTP MCP client:

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

## Tool Contract Notes

- Scan dimensions, positions, and step sizes are in microns unless noted otherwise.
- `set_parameters(parameters)` uses `parameters[0]` as the APS 2-ID-D zp-z target.
- Acquisition results report QueueServer task metadata such as `task_uid`,
  `run_uids`, `scan_ids`, and `save_data_path`.
