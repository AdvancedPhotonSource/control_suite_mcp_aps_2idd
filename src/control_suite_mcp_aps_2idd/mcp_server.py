"""FastMCP HTTP server for APS 2-ID-D tools backed directly by QueueServer."""

from __future__ import annotations

from typing import Annotated, Any
import argparse
import asyncio
import logging

from fastmcp import FastMCP

from control_suite_mcp_aps_2idd.common import APSTwoIDDConfig, parse_range
from control_suite_mcp_aps_2idd.qserver_client import QServerActionConfig, QServerConnectionConfig
from control_suite_mcp_aps_2idd.qserver_instrument import QServerAPSTwoIDDMICInstrument

logger = logging.getLogger(__name__)


def build_service_config(args: argparse.Namespace) -> APSTwoIDDConfig:
    """Build service configuration from CLI args."""
    return APSTwoIDDConfig(
        sample_name=args.sample_name,
        dwell_imaging=args.dwell_imaging,
        dwell_line_scan=args.dwell_line_scan,
        xrf_on=not args.no_xrf,
        preamp1_on=args.preamp1_on,
        using_xrf_maps=args.using_xrf_maps,
        xrf_elms=tuple(args.xrf_elms),
        xrf_roi_num=args.xrf_roi_num,
        allowable_x_range=parse_range(args.allowable_x_range),
        allowable_y_range=parse_range(args.allowable_y_range),
        allowable_z_range=parse_range(args.allowable_z_range),
        plot_image_in_log_scale=args.plot_image_in_log_scale,
        show_colorbar_in_image=args.show_colorbar_in_image,
        line_scan_return_gaussian_fit=args.line_scan_return_gaussian_fit,
        scan_samy=not args.no_scan_samy,
    )


def build_qserver_connection_config(args: argparse.Namespace) -> QServerConnectionConfig:
    """Build QueueServer connection settings from CLI args."""
    return QServerConnectionConfig(
        zmq_control_addr=args.qserver_control_addr,
        zmq_info_addr=args.qserver_info_addr,
        user_group=args.qserver_user_group,
        user=args.qserver_user,
        lock_key=args.qserver_lock_key,
        timeout_s=args.qserver_timeout_s,
        beamline_monitor_manifest_path=args.qserver_beamline_monitor_manifest,
        actions=QServerActionConfig(
            acquire_image_plan=args.qserver_acquire_image_plan,
            acquire_line_scan_plan=args.qserver_acquire_line_scan_plan,
            get_save_data_path_function=args.qserver_get_save_data_path_function,
            move_samy_function=args.qserver_move_samy_function,
            set_zp_z_function=args.qserver_set_zp_z_function,
        ),
    )


def create_mcp(
    *,
    service_config: APSTwoIDDConfig | None = None,
    qserver_connection_config: QServerConnectionConfig | None = None,
) -> FastMCP:
    """Create the FastMCP server."""
    mcp = FastMCP("Control Suite MCP APS 2-ID-D")
    instrument = QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig() if service_config is None else service_config,
        qserver_config=(
            QServerConnectionConfig.from_env()
            if qserver_connection_config is None
            else qserver_connection_config
        ),
    )

    async def call_backend(method: str, params: dict[str, Any] | None = None) -> Any:
        handler = getattr(instrument, method)
        payload = {} if params is None else params
        return await asyncio.to_thread(handler, **payload)

    @mcp.tool()
    async def health() -> dict[str, Any]:
        """Check whether QueueServer is reachable and the MCP backend is healthy."""
        return await call_backend("health")

    @mcp.tool()
    async def get_state() -> dict[str, Any]:
        """Return APS 2-ID-D service and QueueServer state."""
        return await call_backend("get_state")

    async def set_config(
        name: Annotated[str, "Writable configuration attribute name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Set a writable backend acquisition configuration value."""
        return await call_backend("set_config", {"name": name, "value": value})

    async def set_attribute(
        name: Annotated[str, "Writable configuration attribute name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Alias for ``set_config`` used by EAA MCP acquisition proxy."""
        return await call_backend("set_attribute", {"name": name, "value": value})

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
        """Acquire a 2D MIC image in microns centered at ``(x_center, y_center)``.

        The beamline moves during the scan through QueueServer. The result
        contains QueueServer task metadata such as ``task_uid``, ``run_uids``,
        ``scan_ids``, and ``save_data_path``.
        """
        return await call_backend(
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
    async def dump_array(
        buffer_name: Annotated[
            str,
            "Native image buffer name: image_k, image_km1, or image_0.",
        ],
    ) -> dict[str, str]:
        """Raise an error because direct QueueServer mode has no local image buffers."""
        return await call_backend("dump_array", {"buffer_name": buffer_name})

    @mcp.tool()
    async def get_attribute_payload(
        name: Annotated[str, "Native acquisition or parameter tool attribute name."],
    ) -> Any:
        """Return an attribute payload for logic-driven EAA adapter calls.

        NumPy arrays are encoded with dtype, shape, and base64 byte data so the
        adapter can preserve array semantics without importing EAA here.
        """
        return await call_backend("get_attribute_payload", {"name": name})

    @mcp.tool()
    async def acquire_line_scan(
        length: Annotated[float, "The length of the line scan in microns."],
        x_center: Annotated[float, "The scan center x position in microns."],
        y_center: Annotated[float, "The scan center y position in microns."],
        stepsize_x: Annotated[float, "The line-scan step size in microns."],
    ) -> dict[str, Any]:
        """Acquire a horizontal line scan in microns centered at the given position.

        The beamline moves along x at the requested y position through
        QueueServer. The result contains QueueServer task metadata such as
        ``task_uid``, ``run_uids``, ``scan_ids``, and ``save_data_path``.
        """
        return await call_backend(
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
        """Move beamline tuning parameters.

        For APS 2-ID-D ``parameters[0]`` is the zone-plate z position and
        invokes motor motion after worker-side range validation.
        """
        return await call_backend("set_parameters", {"parameters": parameters})

    return mcp


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument("--sample-name", default="smp1")
    parser.add_argument("--dwell-imaging", type=float, default=0.05)
    parser.add_argument("--dwell-line-scan", type=float, default=0.2)
    parser.add_argument("--no-xrf", action="store_true")
    parser.add_argument("--preamp1-on", action="store_true")
    parser.add_argument("--using-xrf-maps", action="store_true")
    parser.add_argument("--xrf-elms", nargs="+", default=["Cr"])
    parser.add_argument("--xrf-roi-num", type=int, default=16)
    parser.add_argument("--allowable-x-range", default=None, help="Comma-separated lower,upper.")
    parser.add_argument("--allowable-y-range", default=None, help="Comma-separated lower,upper.")
    parser.add_argument("--allowable-z-range", default=None, help="Comma-separated lower,upper.")
    parser.add_argument("--plot-image-in-log-scale", action="store_true")
    parser.add_argument("--show-colorbar-in-image", action="store_true")
    parser.add_argument("--line-scan-return-gaussian-fit", action="store_true")
    parser.add_argument("--no-scan-samy", action="store_true")
    parser.add_argument("--qserver-control-addr", default="tcp://127.0.0.1:60615")
    parser.add_argument("--qserver-info-addr", default="tcp://127.0.0.1:60625")
    parser.add_argument("--qserver-user-group", default="root")
    parser.add_argument("--qserver-user", default=None)
    parser.add_argument("--qserver-lock-key", default=None)
    parser.add_argument("--qserver-timeout-s", type=float, default=120.0)
    parser.add_argument("--qserver-beamline-monitor-manifest", default=None)
    parser.add_argument("--qserver-acquire-image-plan", default="fly2d_scanrecord")
    parser.add_argument("--qserver-acquire-line-scan-plan", default="step1d_scanrecord")
    parser.add_argument("--qserver-get-save-data-path-function", default="get_save_data_path")
    parser.add_argument("--qserver-move-samy-function", default=None)
    parser.add_argument("--qserver-set-zp-z-function", default=None)
    return parser


def main() -> None:
    """Run the FastMCP HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    logger.info(
        "Starting MCP server with direct QueueServer backend at %s / %s",
        args.qserver_control_addr,
        args.qserver_info_addr,
    )
    mcp = create_mcp(
        service_config=build_service_config(args),
        qserver_connection_config=build_qserver_connection_config(args),
    )
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
