"""Supervisor CLI for launching the APS 2-ID-D worker and MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import argparse
import logging
import shutil
import signal
import subprocess
import time

from control_suite_mcp_aps_2idd.zmq_client import WorkerClient

logger = logging.getLogger(__name__)


@dataclass
class ManagedProcess:
    """Subprocess metadata managed by the launcher.

    Parameters
    ----------
    name
        Human-readable process name.
    process
        Running subprocess handle.
    """

    name: str
    process: subprocess.Popen[bytes]


def resolve_executable(name: str) -> str:
    """Resolve an executable from PATH."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Could not find executable on PATH: {name}")
    return path


def start_process(name: str, command: Sequence[str]) -> ManagedProcess:
    """Start a managed subprocess."""
    logger.info("Starting %s: %s", name, " ".join(command))
    return ManagedProcess(
        name=name,
        process=subprocess.Popen(command),
    )


def terminate_processes(processes: Sequence[ManagedProcess], timeout_s: float = 5.0) -> None:
    """Terminate all running child processes."""
    for managed in processes:
        if managed.process.poll() is None:
            logger.info("Terminating %s", managed.name)
            managed.process.terminate()
    deadline = time.time() + timeout_s
    for managed in processes:
        remaining = max(0.0, deadline - time.time())
        if managed.process.poll() is None:
            try:
                managed.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Killing %s after graceful shutdown timeout", managed.name)
                managed.process.kill()
    for managed in processes:
        if managed.process.poll() is None:
            managed.process.wait()


def wait_for_worker(endpoint: str, timeout_s: float, request_timeout_ms: int) -> None:
    """Wait until the worker responds to a health command."""
    client = WorkerClient(endpoint, timeout_ms=request_timeout_ms)
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            result = client.call("health")
            if result == {"status": "ok"}:
                return
            last_error = RuntimeError(f"Unexpected health response: {result}")
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"Worker at {endpoint} did not become healthy: {last_error}")


def append_optional_range(command: list[str], flag: str, value: str | None) -> None:
    """Append an optional range CLI argument."""
    if value is not None:
        command.extend([flag, value])


def build_worker_command(args: argparse.Namespace) -> list[str]:
    """Build the worker subprocess command."""
    command = [
        resolve_executable("control-suite-aps-2idd-worker"),
        "--bind",
        args.worker_endpoint,
        "--sample-name",
        args.sample_name,
        "--dwell-imaging",
        str(args.dwell_imaging),
        "--dwell-line-scan",
        str(args.dwell_line_scan),
        "--xrf-roi-num",
        str(args.xrf_roi_num),
    ]
    if args.no_xrf:
        command.append("--no-xrf")
    if args.preamp1_on:
        command.append("--preamp1-on")
    if args.using_xrf_maps:
        command.append("--using-xrf-maps")
    command.append("--xrf-elms")
    command.extend(args.xrf_elms)
    append_optional_range(command, "--allowable-x-range", args.allowable_x_range)
    append_optional_range(command, "--allowable-y-range", args.allowable_y_range)
    append_optional_range(command, "--allowable-z-range", args.allowable_z_range)
    if args.plot_image_in_log_scale:
        command.append("--plot-image-in-log-scale")
    if args.show_colorbar_in_image:
        command.append("--show-colorbar-in-image")
    if args.line_scan_return_gaussian_fit:
        command.append("--line-scan-return-gaussian-fit")
    if args.no_scan_samy:
        command.append("--no-scan-samy")
    return command


def build_mcp_command(args: argparse.Namespace) -> list[str]:
    """Build the MCP server subprocess command."""
    return [
        resolve_executable("control-suite-aps-2idd-mcp"),
        "--worker",
        args.worker_endpoint,
        "--timeout-ms",
        str(args.request_timeout_ms),
        "--host",
        args.mcp_host,
        "--port",
        str(args.mcp_port),
        "--path",
        args.mcp_path,
    ]


def monitor_processes(processes: Sequence[ManagedProcess]) -> int:
    """Block until a child exits and return the launcher exit code."""
    while True:
        for managed in processes:
            return_code = managed.process.poll()
            if return_code is not None:
                if return_code == 0:
                    logger.info("%s exited with code 0", managed.name)
                    return 0
                logger.error("%s exited with code %s", managed.name, return_code)
                return return_code
        time.sleep(0.25)


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """Add worker configuration arguments to the launcher parser."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--worker-startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--request-timeout-ms", type=int, default=30_000)
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8050)
    parser.add_argument("--mcp-path", default="/mcp")
    add_worker_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Launch and supervise worker and MCP server subprocesses."""
    processes: list[ManagedProcess] = []
    shutting_down = False

    def handle_signal(signum: int, _frame) -> None:
        nonlocal shutting_down
        logger.info("Received signal %s, shutting down child processes", signum)
        shutting_down = True
        terminate_processes(processes)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        worker = start_process("worker", build_worker_command(args))
        processes.append(worker)
        wait_for_worker(
            args.worker_endpoint,
            timeout_s=args.worker_startup_timeout_s,
            request_timeout_ms=args.request_timeout_ms,
        )
        logger.info("Worker is healthy at %s", args.worker_endpoint)

        mcp = start_process("mcp", build_mcp_command(args))
        processes.append(mcp)
        logger.info(
            "MCP server starting at http://%s:%s%s",
            args.mcp_host,
            args.mcp_port,
            args.mcp_path,
        )
        return monitor_processes(processes)
    except KeyboardInterrupt:
        shutting_down = True
        return 130
    finally:
        if processes and not shutting_down:
            terminate_processes(processes)


def main() -> None:
    """Run the launcher CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        logger.error("Launcher failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
