"""Tests for the FastMCP frontend contract surface."""

from __future__ import annotations

import asyncio
import time

from fastmcp import Client

import control_suite_mcp_aps_2idd.mcp_server as mcp_server


def test_mcp_server_exposes_required_contract_tools() -> None:
    """The direct QueueServer MCP service exposes the expected tools."""

    class FakeInstrument:
        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def get_state(self) -> dict[str, str]:
            return {"mode": "test"}

        def get_current_mda_file(self) -> dict[str, object]:
            return {"current_mda_file": "test.mda"}

        def get_save_data_path(self) -> dict[str, object]:
            return {"save_data_path": "/tmp/test"}

        def get_global_health_snapshot(self) -> dict[str, object]:
            return {"devices": {}}

        def recover_detector(self, device_name: str, retries: int = 1) -> dict[str, object]:
            return {"device": device_name, "success": True}

        def set_config(self, name: str, value: object) -> dict[str, object]:
            return {"name": name, "value": value}

        def set_attribute(self, name: str, value: object) -> dict[str, object]:
            return {"name": name, "value": value}

        def acquire_image(self, **kwargs) -> dict[str, object]:
            return kwargs

        def process_image(self, **kwargs) -> dict[str, object]:
            return kwargs

        def dump_array(self, buffer_name: str) -> dict[str, str]:
            return {"buffer_name": buffer_name}

        def get_attribute_payload(self, name: str) -> str:
            return name

        def acquire_line_scan(self, **kwargs) -> dict[str, object]:
            return kwargs

        def move_sample(self, axis: str, position: float) -> dict[str, object]:
            return {"axis": axis, "position": position}

        def move_zp_z(self, position: float) -> dict[str, object]:
            return {"position": position}

        def set_parameters(self, parameters: list[float]) -> list[float]:
            return parameters

    mcp_server.QServerAPSTwoIDDMICInstrument = lambda *args, **kwargs: FakeInstrument()
    mcp = mcp_server.create_mcp()
    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

    assert tool_names == {
        "aps2idd_control.health",
        "aps2idd_control.get_state",
        "aps2idd_control.get_current_mda_file",
        "aps2idd_control.get_save_data_path",
        "aps2idd_control.get_global_health_snapshot",
        "aps2idd_control.recover_detector",
        "aps2idd_control.acquire_image",
        "aps2idd_control.process_image",
        "aps2idd_control.dump_array",
        "aps2idd_control.get_attribute_payload",
        "aps2idd_control.acquire_line_scan",
        "aps2idd_control.move_sample",
        "aps2idd_control.move_zp_z",
        "aps2idd_control.set_parameters",
    }


def test_acquire_image_streams_console_as_progress(monkeypatch) -> None:
    """acquire_image forwards QueueServer console output as MCP progress notifications."""

    class FakeInstrument:
        def acquire_image(self, on_console=None, **kwargs):
            if on_console is not None:
                on_console({"msg": "scan point 1"})
                on_console({"msg": "scan point 2"})
            # Keep the worker thread alive briefly so the event loop drains the
            # queued console messages before the call returns.
            time.sleep(0.3)
            return {"item_uid": "uid-img", "run_uids": [], "scan_ids": []}

    monkeypatch.setattr(
        mcp_server, "QServerAPSTwoIDDMICInstrument", lambda *a, **k: FakeInstrument()
    )
    mcp = mcp_server.create_mcp()

    progress_messages: list[str | None] = []

    async def progress_handler(progress, total, message):
        progress_messages.append(message)

    async def run():
        async with Client(mcp, progress_handler=progress_handler) as client:
            return await client.call_tool(
                "aps2idd_control.acquire_image",
                {
                    "width": 1.0,
                    "height": 1.0,
                    "x_center": 0.0,
                    "y_center": 0.0,
                    "stepsize_x": 1.0,
                    "stepsize_y": 1.0,
                },
            )

    result = asyncio.run(run())

    assert result.data["item_uid"] == "uid-img"
    assert "scan point 1" in progress_messages
    assert "scan point 2" in progress_messages


def test_parse_args_uses_toml_defaults_and_cli_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
host = "127.0.0.1"
port = 9000
xrf_on = false
allowable_x_range = [1.0, 2.0]

[qserver]
control_addr = "tcp://example:60615"
move_sample = "move_sample"
move_zp_z = "move_zp_z"
""".strip()
    )

    args = mcp_server.parse_args([
        "--config",
        str(config_path),
        "--host",
        "0.0.0.0",
    ])

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.no_xrf is True
    assert args.allowable_x_range == "1.0,2.0"
    assert args.qserver_control_addr == "tcp://example:60615"
    assert args.qserver_move_sample_plan == "move_sample"
    assert args.qserver_move_zp_z_plan == "move_zp_z"
