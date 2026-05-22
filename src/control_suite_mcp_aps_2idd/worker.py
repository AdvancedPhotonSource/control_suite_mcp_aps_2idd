"""ZMQ instrument worker for the APS 2-ID-D control suite."""

from __future__ import annotations

from typing import Any, Callable
import argparse
import json
import logging

import zmq

from control_suite_mcp_aps_2idd.instrument import APSTwoIDDConfig, APSTwoIDDMICInstrument
from control_suite_mcp_aps_2idd.protocol import error_response, ok_response

logger = logging.getLogger(__name__)


class InstrumentWorker:
    """Blocking ZMQ worker that owns the APS 2-ID-D instrument adapter.

    Parameters
    ----------
    instrument
        Instrument adapter. This object lives only in the worker process.
    """

    def __init__(self, instrument: APSTwoIDDMICInstrument) -> None:
        self.instrument = instrument
        self.handlers: dict[str, Callable[..., Any]] = {
            "health": self.instrument.health,
            "get_state": self.instrument.get_state,
            "set_config": self.instrument.set_config,
            "set_attribute": self.instrument.set_attribute,
            "acquire_image": self.instrument.acquire_image,
            "dump_array": self.instrument.dump_array,
            "get_attribute_payload": self.instrument.get_attribute_payload,
            "acquire_line_scan": self.instrument.acquire_line_scan,
            "set_parameters": self.instrument.set_parameters,
        }

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Execute one command message and return a response envelope."""
        command_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(method, str) or method not in self.handlers:
            return error_response(command_id, f"Unknown method: {method}")
        if not isinstance(params, dict):
            return error_response(command_id, "Command params must be a JSON object.")
        try:
            result = self.handlers[method](**params)
            return ok_response(command_id, result)
        except Exception as exc:
            logger.exception("Worker command failed: %s", method)
            return error_response(command_id, str(exc))

    def serve(self, bind: str) -> None:
        """Serve worker commands forever on a ZMQ REP socket."""
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(bind)
        logger.info("Instrument worker listening on %s", bind)
        try:
            while True:
                raw_message = socket.recv_string()
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        response = error_response(None, "Command must be a JSON object.")
                    else:
                        response = self.handle_message(message)
                except json.JSONDecodeError as exc:
                    response = error_response(None, f"Invalid JSON command: {exc}")
                socket.send_string(json.dumps(response, allow_nan=True))
        finally:
            socket.close(linger=0)


def parse_range(value: str | None) -> tuple[float, float] | None:
    """Parse a comma-separated numeric range.

    Parameters
    ----------
    value
        Comma-separated range string such as ``"0,100"``.

    Returns
    -------
    tuple[float, float] | None
        Parsed range, or None when no value was supplied.
    """
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Ranges must contain exactly two comma-separated values.")
    return (float(parts[0]), float(parts[1]))


def build_config(args: argparse.Namespace) -> APSTwoIDDConfig:
    """Build the instrument configuration from CLI arguments."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
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
    return parser


def main() -> None:
    """Run the instrument worker process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    worker = InstrumentWorker(APSTwoIDDMICInstrument(build_config(args)))
    worker.serve(args.bind)


if __name__ == "__main__":
    main()
