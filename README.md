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

## Configuration

A repo-level [config.toml](/home/beams8/USER2IDD/software/control_suite_mcp_aps_2idd/config.toml:1)
provides the default MCP server configuration, including:

- host/port/path
- allowable `x`, `y`, and `z` ranges
- QueueServer addresses
- approved QueueServer plan names for acquisition and motion

CLI flags still override TOML values when needed.

## Run

Start the MCP server with the repo config:

```bash
control-suite-aps-2idd-mcp --config config.toml
```

If QueueServer is running on the same host as the MCP server, keep the default
`127.0.0.1` QueueServer addresses in `config.toml`. Otherwise, update the
`[qserver]` section.

You can still override specific values from the command line, for example:

```bash
control-suite-aps-2idd-mcp   --config config.toml   --qserver-control-addr tcp://[hostname]:60615   --qserver-info-addr tcp://[hostname]:60625   --allowable-x-range 0,50   --allowable-y-range 0,50
```

## MCP URL

Clients should connect to:

```text
http://127.0.0.1:8050/mcp
```

## Tools

- `health()`
- `get_state()`
- `acquire_image(width, height, x_center, y_center, stepsize_x, stepsize_y, dwell_ms=None)`
- `dump_array(buffer_name)`
- `acquire_line_scan(positioner_name, length, stepsize, center=0, sample_x=None, sample_y=None, sample_z=None, energy=None, dwell_ms=None)`
- `move_sample(axis, position)`
- `move_zp_z(position)`
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
- `move_zp_z(position)` drives the zone-plate z positioner, validated against
  `allowable_zp_range` (distinct from the sample z motor's `allowable_z_range`).
  `set_parameters(parameters)` is equivalent, using `parameters[0]` as the zp-z
  target (also validated against `allowable_zp_range`).
- Motion and acquisition tools are QueueServer *plans*, submitted with
  `item_execute`. Plans return an `item_uid` (not a `task_uid`); the service
  waits for the RE manager to return to idle and reads the outcome from
  QueueServer history. `task_uid` is only produced by QueueServer *functions*
  (e.g. `get_save_data_path`).
- `acquire_image` and `acquire_line_scan` stream live scan progress as MCP
  progress notifications, sourced from the QueueServer console (ZMQ info)
  output. Their results report `item_uid`, `run_uids`, `scan_ids`, and
  `save_data_path`.
- `acquire_line_scan` drives the axis named by `positioner_name`
  (`x`, `y`, `z`, or `energy`); `length`, `center`, and `stepsize` are in that
  positioner's units (microns for x/y/z, keV for energy). **`center` is a
  relative offset** from the positioner's position at scan time (e.g. `center=0`
  scans symmetrically around the current position). The `step1d_scanrecord` plan
  moves the sample/energy to the optional `sample_x`/`sample_y`/`sample_z`/
  `energy` positions (current position is kept for any left unset) before
  scanning.
- Dwell time per point: pass `dwell_ms` to override per call; when omitted the
  acquisition uses the configured `dwell_imaging` (images) or `dwell_line_scan`
  (line scans) value.
- Range validation: any explicitly provided absolute position
  (`sample_x`/`sample_y`/`sample_z`/`energy`) is validated against its axis range
  (`allowable_x/y/z_range` in microns, `allowable_energy_range` in keV). The
  relative scan extent (`target + center ± length/2`) is only validated when the
  driven axis's absolute target is supplied; with no target the current position
  is unknown, so the absolute extent cannot be checked.
