"""Tests for the FastMCP frontend contract surface."""

from __future__ import annotations

import asyncio

import control_suite_mcp_aps_2idd.mcp_server as mcp_server


def test_mcp_server_exposes_required_contract_tools() -> None:
    """The direct QueueServer MCP service exposes the expected tools."""

    class FakeInstrument:
        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def get_state(self) -> dict[str, str]:
            return {"mode": "test"}

        def set_config(self, name: str, value: object) -> dict[str, object]:
            return {"name": name, "value": value}

        def set_attribute(self, name: str, value: object) -> dict[str, object]:
            return {"name": name, "value": value}

        def acquire_image(self, **kwargs) -> dict[str, object]:
            return kwargs

        def dump_array(self, buffer_name: str) -> dict[str, str]:
            return {"buffer_name": buffer_name}

        def get_attribute_payload(self, name: str) -> str:
            return name

        def acquire_line_scan(self, **kwargs) -> dict[str, object]:
            return kwargs

        def set_parameters(self, parameters: list[float]) -> list[float]:
            return parameters

    mcp_server.QServerAPSTwoIDDMICInstrument = lambda *args, **kwargs: FakeInstrument()
    mcp = mcp_server.create_mcp()
    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

    assert tool_names == {
        "health",
        "get_state",
        "acquire_image",
        "dump_array",
        "get_attribute_payload",
        "acquire_line_scan",
        "set_parameters",
    }
