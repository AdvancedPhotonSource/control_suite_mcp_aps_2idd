"""Integration tests for the ZMQ worker boundary."""

from __future__ import annotations

from typing import Any
import multiprocessing as mp
import time

from control_suite_mcp_aps_2idd.worker import InstrumentWorker
from control_suite_mcp_aps_2idd.zmq_client import WorkerClient


class FakeInstrument:
    """Small instrument stand-in for testing the ZMQ worker boundary."""

    def __init__(self) -> None:
        self.line_scan_return_gaussian_fit = False
        self.counter_acquire_image = 0

    def health(self) -> dict[str, str]:
        """Return fake worker health."""
        return {"status": "ok"}

    def get_state(self) -> dict[str, Any]:
        """Return fake instrument state."""
        return {
            "counter_acquire_image": self.counter_acquire_image,
            "line_scan_return_gaussian_fit": self.line_scan_return_gaussian_fit,
        }

    def set_config(self, name: str, value: Any) -> dict[str, Any]:
        """Set a fake config value."""
        setattr(self, name, value)
        return {"name": name, "value": value}

    def set_attribute(self, name: str, value: Any) -> dict[str, Any]:
        """Alias for fake config updates."""
        return self.set_config(name=name, value=value)

    def acquire_image(
        self,
        width: float,
        height: float,
        x_center: float,
        y_center: float,
        stepsize_x: float,
        stepsize_y: float,
    ) -> dict[str, Any]:
        """Return an EAA-compatible fake acquisition payload."""
        self.counter_acquire_image += 1
        return {
            "img_path": "/tmp/fake.png",
            "array_path": "/tmp/fake.npy",
            "psize": stepsize_x,
            "width": width,
            "height": height,
            "x_center": x_center,
            "y_center": y_center,
            "stepsize_y": stepsize_y,
        }

    def acquire_line_scan(
        self,
        length: float,
        x_center: float,
        y_center: float,
        stepsize_x: float,
    ) -> dict[str, Any]:
        """Return an EAA-compatible fake line-scan payload."""
        result = {"img_path": "/tmp/fake-line.png"}
        if self.line_scan_return_gaussian_fit:
            result["fwhm"] = 1.5
        return result

    def set_parameters(self, parameters: list[float]) -> str:
        """Return a fake parameter update message."""
        return f"Moved Zone Plate z position to position: {parameters[0]}"


def run_worker(endpoint: str) -> None:
    """Run a worker process for tests."""
    InstrumentWorker(FakeInstrument()).serve(endpoint)


def test_worker_client_exposes_contract_methods() -> None:
    """A ZMQ client can drive worker methods used by the MCP proxy."""
    endpoint = "tcp://127.0.0.1:5577"
    process = mp.Process(target=run_worker, args=(endpoint,), daemon=True)
    process.start()
    client = WorkerClient(endpoint, timeout_ms=1000)

    try:
        deadline = time.time() + 5
        while True:
            try:
                assert client.call("health") == {"status": "ok"}
                break
            except TimeoutError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)

        image_result = client.call(
            "acquire_image",
            {
                "width": 10.0,
                "height": 8.0,
                "x_center": 1.0,
                "y_center": 2.0,
                "stepsize_x": 0.5,
                "stepsize_y": 0.5,
            },
        )
        assert image_result["img_path"] == "/tmp/fake.png"
        assert image_result["array_path"] == "/tmp/fake.npy"
        assert image_result["psize"] == 0.5

        assert client.call(
            "set_attribute",
            {"name": "line_scan_return_gaussian_fit", "value": True},
        ) == {"name": "line_scan_return_gaussian_fit", "value": True}
        line_result = client.call(
            "acquire_line_scan",
            {"length": 4.0, "x_center": 1.0, "y_center": 2.0, "stepsize_x": 0.5},
        )
        assert line_result["img_path"] == "/tmp/fake-line.png"
        assert line_result["fwhm"] == 1.5

        assert client.call("set_parameters", {"parameters": [-190.0]}).startswith(
            "Moved Zone Plate z position"
        )
        state = client.call("get_state")
        assert state["counter_acquire_image"] == 1
    finally:
        process.terminate()
        process.join(timeout=5)
