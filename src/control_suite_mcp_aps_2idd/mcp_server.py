"""FastMCP HTTP server that forwards APS 2-ID-D tools to ZMQ."""

from __future__ import annotations

from typing import Annotated, Any
import argparse
import asyncio
import logging

from fastmcp import FastMCP

from control_suite_mcp_aps_2idd.zmq_client import WorkerClient

logger = logging.getLogger(__name__)


def create_mcp(worker_endpoint: str, timeout_ms: int = 30_000) -> FastMCP:
    """Create the FastMCP server.

    Parameters
    ----------
    worker_endpoint
        ZMQ endpoint used to reach the instrument worker.
    timeout_ms
        Worker request timeout in milliseconds.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    mcp = FastMCP("Control Suite MCP APS 2-ID-D")
    client = WorkerClient(worker_endpoint, timeout_ms=timeout_ms)

    async def call_worker(method: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(client.call, method, params)

    @mcp.tool()
    async def health() -> dict[str, str]:
        """Check whether the instrument worker is reachable."""
        return await call_worker("health")

    @mcp.tool()
    async def get_state() -> dict[str, Any]:
        """Return APS 2-ID-D acquisition state."""
        return await call_worker("get_state")

    @mcp.tool()
    async def set_config(
        name: Annotated[str, "Writable configuration attribute name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Set a writable backend acquisition configuration value."""
        return await call_worker("set_config", {"name": name, "value": value})

    @mcp.tool()
    async def set_attribute(
        name: Annotated[str, "Writable configuration attribute name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Alias for ``set_config`` used by EAA MCP acquisition proxy."""
        return await call_worker("set_attribute", {"name": name, "value": value})

    @mcp.tool()
    async def acquire_image(
        width: Annotated[float, "The width of the scan area in microns."],
        height: Annotated[float, "The height of the scan area in microns."],
        x_center: Annotated[float, "The scan center x position in microns."],
        y_center: Annotated[float, "The scan center y position in microns."],
        stepsize_x: Annotated[
            float,
            "The scan step size in the x direction in microns.",
        ],
        stepsize_y: Annotated[
            float,
            "The scan step size in the y direction in microns.",
        ],
    ) -> dict[str, Any]:
        """Acquire an image with the APS 2-ID-D MIC instrument."""
        return await call_worker(
            "acquire_image",
            {
                "width": width,
                "height": height,
                "x_center": x_center,
                "y_center": y_center,
                "stepsize_x": stepsize_x,
                "stepsize_y": stepsize_y,
            },
        )

    @mcp.tool()
    async def acquire_line_scan(
        length: Annotated[float, "The length of the line scan in microns."],
        x_center: Annotated[float, "The scan center x position in microns."],
        y_center: Annotated[float, "The scan center y position in microns."],
        stepsize_x: Annotated[float, "The line-scan step size in microns."],
    ) -> dict[str, Any]:
        """Acquire a horizontal line scan with the APS 2-ID-D MIC instrument."""
        return await call_worker(
            "acquire_line_scan",
            {
                "length": length,
                "x_center": x_center,
                "y_center": y_center,
                "stepsize_x": stepsize_x,
            },
        )

    @mcp.tool()
    async def set_parameters(
        parameters: Annotated[
            list[float],
            "Parameter values to set. For APS 2-ID-D this should contain zp-z.",
        ],
    ) -> str:
        """Set beamline tuning parameters."""
        return await call_worker("set_parameters", {"parameters": parameters})

    return mcp


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--path", default="/mcp")
    return parser


def main() -> None:
    """Run the FastMCP HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    logger.info("Starting MCP server, forwarding to worker at %s", args.worker)
    mcp = create_mcp(args.worker, timeout_ms=args.timeout_ms)
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
