"""FastMCP HTTP server for APS 2-ID-D tools backed directly by QueueServer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any
import argparse
import asyncio
import logging
import sys
import tomllib

from fastmcp import Context, FastMCP

from control_suite_mcp_aps_2idd.common import APSTwoIDDConfig, parse_range
from control_suite_mcp_aps_2idd.qserver_client import QServerActionConfig, QServerConnectionConfig
from control_suite_mcp_aps_2idd.qserver_instrument import QServerAPSTwoIDDMICInstrument

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"


def _stringify_range(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and len(value) == 2:
        return f"{float(value[0])},{float(value[1])}"
    raise ValueError(f"Invalid range value: {value!r}")


def load_config_file(path: str | Path, *, required: bool = False) -> dict[str, Any]:
    """Load server defaults from a TOML configuration file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        if required:
            raise FileNotFoundError(config_path)
        return {}

    data = tomllib.loads(config_path.read_text())
    qserver = data.get("qserver", {})
    if not isinstance(qserver, Mapping):
        raise ValueError("The [qserver] section must be a TOML table.")

    return {
        "host": data.get("host", "127.0.0.1"),
        "port": data.get("port", 8050),
        "path": data.get("path", "/mcp"),
        "sample_name": data.get("sample_name", "smp1"),
        "dwell_imaging": data.get("dwell_imaging", 0.05),
        "dwell_line_scan": data.get("dwell_line_scan", 0.2),
        "no_xrf": not bool(data.get("xrf_on", True)),
        "preamp1_on": bool(data.get("preamp1_on", False)),
        "using_xrf_maps": bool(data.get("using_xrf_maps", False)),
        "xrf_elms": list(data.get("xrf_elms", ["Cr"])),
        "xrf_roi_num": data.get("xrf_roi_num", 16),
        "allowable_x_range": _stringify_range(data.get("allowable_x_range")),
        "allowable_y_range": _stringify_range(data.get("allowable_y_range")),
        "allowable_z_range": _stringify_range(data.get("allowable_z_range")),
        "allowable_zp_range": _stringify_range(data.get("allowable_zp_range")),
        "allowable_energy_range": _stringify_range(data.get("allowable_energy_range")),
        "plot_image_in_log_scale": bool(data.get("plot_image_in_log_scale", False)),
        "show_colorbar_in_image": bool(data.get("show_colorbar_in_image", False)),
        "line_scan_return_gaussian_fit": bool(data.get("line_scan_return_gaussian_fit", False)),
        "no_scan_samy": not bool(data.get("scan_samy", True)),
        "qserver_control_addr": qserver.get("control_addr", "tcp://127.0.0.1:60615"),
        "qserver_info_addr": qserver.get("info_addr", "tcp://127.0.0.1:60625"),
        "qserver_user_group": qserver.get("user_group", "root"),
        "qserver_user": qserver.get("user"),
        "qserver_lock_key": qserver.get("lock_key"),
        "qserver_timeout_s": qserver.get("timeout_s", 120.0),
        "qserver_beamline_monitor_manifest": qserver.get("beamline_monitor_manifest"),
        "qserver_acquire_image_plan": qserver.get("acquire_image", "fly2d_scanrecord"),
        "qserver_acquire_line_scan_plan": qserver.get("acquire_line_scan", "step1d_scanrecord"),
        "qserver_move_sample_plan": qserver.get("move_sample"),
        "qserver_move_zp_z_plan": qserver.get("move_zp_z"),
        "qserver_get_save_data_path_function": qserver.get("get_save_data_path", "get_save_data_path"),
        "qserver_get_current_mda_file_function": qserver.get("get_current_mda_file", "get_current_mda_file"),
    }


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
        allowable_zp_range=parse_range(args.allowable_zp_range),
        allowable_energy_range=parse_range(args.allowable_energy_range),
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
            acquire_image=args.qserver_acquire_image_plan,
            acquire_line_scan=args.qserver_acquire_line_scan_plan,
            move_sample=args.qserver_move_sample_plan,
            move_zp_z=args.qserver_move_zp_z_plan,
            get_save_data_path=args.qserver_get_save_data_path_function,
            get_current_mda_file=args.qserver_get_current_mda_file_function,
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

    async def run_acquisition_with_progress(
        method: str,
        params: dict[str, Any],
        ctx: Context,
    ) -> Any:
        """Run a long acquisition while streaming QueueServer console output.

        The blocking acquisition runs in a worker thread; QueueServer console
        messages (published on the ZMQ info channel) are forwarded to this
        coroutine through a thread-safe queue and re-emitted as MCP progress
        notifications until the scan finishes.
        """
        handler = getattr(instrument, method)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()

        def on_console(message: Mapping[str, Any]) -> None:
            # Invoked from the worker thread; hop back onto the event loop.
            loop.call_soon_threadsafe(queue.put_nowait, message)

        work = asyncio.ensure_future(
            asyncio.to_thread(handler, on_console=on_console, **params)
        )
        steps = 0
        while not work.done() or not queue.empty():
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            text = message.get("msg") if isinstance(message, Mapping) else None
            text = text.strip() if isinstance(text, str) else None
            steps += 1
            if text:
                await ctx.info(text)
            await ctx.report_progress(progress=steps, message=text or None)
        return await work

    @mcp.tool()
    async def health() -> dict[str, Any]:
        """Check whether QueueServer is reachable and the MCP backend is healthy."""
        return await call_backend("health")

    @mcp.tool()
    async def get_state() -> dict[str, Any]:
        """Return APS 2-ID-D service and QueueServer state."""
        return await call_backend("get_state")

    @mcp.tool()
    async def get_current_mda_file() -> dict[str, Any]:
        """Return the current/next MDA file name from QueueServer.

        Runs the allowlisted ``get_current_mda_file`` QueueServer helper
        function. The result contains ``current_mda_file`` (the savedata
        ``next_file_name``), or null when the savedata device is unavailable.
        """
        return await call_backend("get_current_mda_file")

    @mcp.tool()
    async def get_save_data_path() -> dict[str, Any]:
        """Return the current save data path from QueueServer.

        Runs the allowlisted ``get_save_data_path`` QueueServer helper
        function. The result contains ``save_data_path`` (the savedata
        auto-storage path), or null when the savedata device is unavailable.
        """
        return await call_backend("get_save_data_path")

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
        ctx: Context,
        dwell_ms: Annotated[
            float | None,
            "Dwell time per point in milliseconds; uses the configured "
            "dwell_imaging value when omitted.",
        ] = None,
    ) -> dict[str, Any]:
        """Acquire a 2D MIC image in microns centered at ``(x_center, y_center)``.

        The beamline moves during the scan through QueueServer. Live scan
        progress is streamed as MCP progress notifications from the QueueServer
        console (ZMQ) output. The result contains QueueServer metadata such as
        ``item_uid``, ``run_uids``, ``scan_ids``, ``save_data_path``, and
        ``current_mda_file`` (the MDA file this scan wrote).
        """
        return await run_acquisition_with_progress(
            "acquire_image",
            {
                "width": width,
                "height": height,
                "x_center": x_center,
                "y_center": y_center,
                "stepsize_x": stepsize_x,
                "stepsize_y": stepsize_y,
                "dwell_ms": dwell_ms,
            },
            ctx,
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
        positioner_name: Annotated[
            str,
            "Axis to scan: 'x', 'y', 'z', or 'energy'.",
        ],
        length: Annotated[
            float,
            "Total scan width along the positioner (microns for x/y/z, keV for energy).",
        ],
        stepsize: Annotated[
            float,
            "Step size along the positioner (microns for x/y/z, keV for energy).",
        ],
        ctx: Context,
        center: Annotated[
            float,
            "Scan center as a RELATIVE offset along the positioner (microns for "
            "x/y/z, keV for energy); defaults to 0 (scan around current position).",
        ] = 0.0,
        sample_x: Annotated[
            float | None,
            "Sample x position in microns; current position is kept if omitted.",
        ] = None,
        sample_y: Annotated[
            float | None,
            "Sample y position in microns; current position is kept if omitted.",
        ] = None,
        sample_z: Annotated[
            float | None,
            "Sample z position in microns; current position is kept if omitted.",
        ] = None,
        energy: Annotated[
            float | None,
            "Beam energy in keV; current energy is kept if omitted.",
        ] = None,
        dwell_ms: Annotated[
            float | None,
            "Dwell time per point in milliseconds; uses the configured "
            "dwell_line_scan value when omitted.",
        ] = None,
    ) -> dict[str, Any]:
        """Acquire a 1D line scan by driving the chosen positioner through QueueServer.

        This tool allows you to perform 3 types of operations:
        - Lateral spatial line scan in the x/y plane of the sample
        - Sample z scan along the beam direction (depth scan)
        - Energy scan by tuning the monochromator energy

        To acquire a horizontal spatial line scan, set ``positioner_name``
        to ``x``; for vertical, set it to ``y``. For lateral scans,
        ``sample_x``, ``sample_y`` set the center position of the line scan.
        Do not set ``sample_z`` so that you don't accidentally move the sample
        out of focus.

        To acquire a depth scan, set ``positioner_name`` to ``z``.

        For energy scan, set ``positioner_name`` to ``energy``.
        
        ``length``, ``center``, and ``stepsize`` are expressed in
        that positioner's units (microns for x/y/z, keV for energy). 
        
        Live scan progress is streamed as MCP progress notifications from the
        QueueServer console (ZMQ) output. The result contains QueueServer
        metadata such as ``item_uid``, ``run_uids``, ``scan_ids``,
        ``save_data_path``, and ``current_mda_file`` (the MDA file this
        scan wrote).
        """
        return await run_acquisition_with_progress(
            "acquire_line_scan",
            {
                "positioner_name": positioner_name,
                "length": length,
                "center": center,
                "stepsize": stepsize,
                "sample_x": sample_x,
                "sample_y": sample_y,
                "sample_z": sample_z,
                "energy": energy,
                "dwell_ms": dwell_ms,
            },
            ctx,
        )

    @mcp.tool()
    async def move_sample(
        axis: Annotated[str, "Sample motion axis: x, y, or z."],
        position: Annotated[float, "Target sample position."],
    ) -> dict[str, Any]:
        """Move one sample axis through an allowlisted QueueServer plan."""
        return await call_backend("move_sample", {"axis": axis, "position": position})

    @mcp.tool()
    async def move_zp_z(
        position: Annotated[float, "Zone-plate z (zp-z) target position in microns."],
    ) -> dict[str, Any]:
        """Move the zone-plate z (zp-z) positioner through an allowlisted QueueServer plan.

        zp-z is distinct from the sample z motor; the target is validated against
        ``allowable_zp_range``. The result contains QueueServer metadata such as
        ``item_uid`` and ``exit_status``.
        """
        return await call_backend("move_zp_z", {"position": position})

    @mcp.tool()
    async def set_parameters(
        parameters: Annotated[
            list[float],
            "Parameter values to set. For APS 2-ID-D this should contain zp-z.",
        ],
    ) -> str:
        """Move beamline tuning parameters.

        For APS 2-ID-D ``parameters[0]`` is the zone-plate z position and
        invokes motor motion after worker-side range validation. Equivalent to
        ``move_zp_z(parameters[0])``.
        """
        return await call_backend("set_parameters", {"parameters": parameters})

    return mcp


def build_parser(
    config_defaults: Mapping[str, Any] | None = None,
    *,
    default_config_path: str = DEFAULT_CONFIG_PATH,
) -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    defaults = {
        "host": "127.0.0.1",
        "port": 8050,
        "path": "/mcp",
        "sample_name": "smp1",
        "dwell_imaging": 0.05,
        "dwell_line_scan": 0.2,
        "no_xrf": False,
        "preamp1_on": False,
        "using_xrf_maps": False,
        "xrf_elms": ["Cr"],
        "xrf_roi_num": 16,
        "allowable_x_range": None,
        "allowable_y_range": None,
        "allowable_z_range": None,
        "allowable_zp_range": None,
        "allowable_energy_range": None,
        "plot_image_in_log_scale": False,
        "show_colorbar_in_image": False,
        "line_scan_return_gaussian_fit": False,
        "no_scan_samy": False,
        "qserver_control_addr": "tcp://127.0.0.1:60615",
        "qserver_info_addr": "tcp://127.0.0.1:60625",
        "qserver_user_group": "root",
        "qserver_user": None,
        "qserver_lock_key": None,
        "qserver_timeout_s": 120.0,
        "qserver_beamline_monitor_manifest": None,
        "qserver_acquire_image_plan": "fly2d_scanrecord",
        "qserver_acquire_line_scan_plan": "step1d_scanrecord",
        "qserver_move_sample_plan": None,
        "qserver_move_zp_z_plan": None,
        "qserver_get_save_data_path_function": "get_save_data_path",
        "qserver_get_current_mda_file_function": "get_current_mda_file",
    }
    if config_defaults:
        defaults.update(dict(config_defaults))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config_path, help="Path to a TOML config file.")
    parser.add_argument("--host", default=defaults["host"])
    parser.add_argument("--port", type=int, default=defaults["port"])
    parser.add_argument("--path", default=defaults["path"])
    parser.add_argument("--sample-name", default=defaults["sample_name"])
    parser.add_argument("--dwell-imaging", type=float, default=defaults["dwell_imaging"])
    parser.add_argument("--dwell-line-scan", type=float, default=defaults["dwell_line_scan"])
    parser.add_argument("--no-xrf", action="store_true", default=defaults["no_xrf"])
    parser.add_argument("--preamp1-on", action="store_true", default=defaults["preamp1_on"])
    parser.add_argument("--using-xrf-maps", action="store_true", default=defaults["using_xrf_maps"])
    parser.add_argument("--xrf-elms", nargs="+", default=defaults["xrf_elms"])
    parser.add_argument("--xrf-roi-num", type=int, default=defaults["xrf_roi_num"])
    parser.add_argument("--allowable-x-range", default=defaults["allowable_x_range"], help="Comma-separated lower,upper.")
    parser.add_argument("--allowable-y-range", default=defaults["allowable_y_range"], help="Comma-separated lower,upper.")
    parser.add_argument("--allowable-z-range", default=defaults["allowable_z_range"], help="Comma-separated lower,upper.")
    parser.add_argument("--allowable-zp-range", default=defaults["allowable_zp_range"], help="Comma-separated lower,upper (zp-z).")
    parser.add_argument("--allowable-energy-range", default=defaults["allowable_energy_range"], help="Comma-separated lower,upper (keV).")
    parser.add_argument("--plot-image-in-log-scale", action="store_true", default=defaults["plot_image_in_log_scale"])
    parser.add_argument("--show-colorbar-in-image", action="store_true", default=defaults["show_colorbar_in_image"])
    parser.add_argument("--line-scan-return-gaussian-fit", action="store_true", default=defaults["line_scan_return_gaussian_fit"])
    parser.add_argument("--no-scan-samy", action="store_true", default=defaults["no_scan_samy"])
    parser.add_argument("--qserver-control-addr", default=defaults["qserver_control_addr"])
    parser.add_argument("--qserver-info-addr", default=defaults["qserver_info_addr"])
    parser.add_argument("--qserver-user-group", default=defaults["qserver_user_group"])
    parser.add_argument("--qserver-user", default=defaults["qserver_user"])
    parser.add_argument("--qserver-lock-key", default=defaults["qserver_lock_key"])
    parser.add_argument("--qserver-timeout-s", type=float, default=defaults["qserver_timeout_s"])
    parser.add_argument("--qserver-beamline-monitor-manifest", default=defaults["qserver_beamline_monitor_manifest"])
    parser.add_argument("--qserver-acquire-image-plan", default=defaults["qserver_acquire_image_plan"])
    parser.add_argument("--qserver-acquire-line-scan-plan", default=defaults["qserver_acquire_line_scan_plan"])
    parser.add_argument("--qserver-move-sample-plan", default=defaults["qserver_move_sample_plan"])
    parser.add_argument("--qserver-get-save-data-path-function", default=defaults["qserver_get_save_data_path_function"])
    parser.add_argument("--qserver-get-current-mda-file-function", default=defaults["qserver_get_current_mda_file_function"])
    parser.add_argument("--qserver-move-zp-z-plan", default=defaults["qserver_move_zp_z_plan"])
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with TOML config defaults and CLI overrides."""
    argv = sys.argv[1:] if argv is None else argv
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_ns, _ = config_parser.parse_known_args(argv)
    config_arg_present = "--config" in argv
    try:
        config_defaults = load_config_file(config_ns.config, required=config_arg_present)
    except (FileNotFoundError, ValueError) as exc:
        parser = build_parser(default_config_path=config_ns.config)
        parser.error(str(exc))
    parser = build_parser(config_defaults, default_config_path=config_ns.config)
    return parser.parse_args(argv)


def main() -> None:
    """Run the FastMCP HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
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
