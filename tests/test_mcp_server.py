"""Tests for the FastMCP frontend contract surface."""

from __future__ import annotations

import asyncio

from control_suite_mcp_aps_2idd.mcp_server import create_mcp


def test_mcp_server_exposes_required_contract_tools() -> None:
    """Only the named EAA contract methods are registered as MCP tools."""
    mcp = create_mcp("tcp://127.0.0.1:5999")
    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

    assert tool_names == {
        "acquire_image",
        "dump_array",
        "get_attribute_payload",
        "acquire_line_scan",
        "set_parameters",
    }
