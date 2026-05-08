"""Tests for launcher command construction."""

from __future__ import annotations

from argparse import Namespace

from control_suite_mcp_aps_2idd.launcher import build_mcp_command, build_worker_command


def test_launcher_builds_separate_process_commands(monkeypatch) -> None:
    """Launcher commands target separate worker and MCP executables."""
    monkeypatch.setattr(
        "control_suite_mcp_aps_2idd.launcher.resolve_executable",
        lambda name: f"/venv/bin/{name}",
    )
    args = Namespace(
        worker_endpoint="tcp://127.0.0.1:5555",
        sample_name="smp1",
        dwell_imaging=0.05,
        dwell_line_scan=0.2,
        no_xrf=False,
        preamp1_on=False,
        using_xrf_maps=False,
        xrf_elms=["Cr"],
        xrf_roi_num=16,
        allowable_x_range=None,
        allowable_y_range=None,
        allowable_z_range=None,
        plot_image_in_log_scale=False,
        show_colorbar_in_image=False,
        line_scan_return_gaussian_fit=False,
        no_scan_samy=False,
        request_timeout_ms=30000,
        mcp_host="127.0.0.1",
        mcp_port=8050,
        mcp_path="/mcp",
    )

    worker_command = build_worker_command(args)
    mcp_command = build_mcp_command(args)

    assert worker_command[0] == "/venv/bin/control-suite-aps-2idd-worker"
    assert mcp_command[0] == "/venv/bin/control-suite-aps-2idd-mcp"
    assert "--bind" in worker_command
    assert "--worker" in mcp_command


def test_launcher_forwards_worker_configuration(monkeypatch) -> None:
    """Launcher forwards beamline configuration to the worker process."""
    monkeypatch.setattr(
        "control_suite_mcp_aps_2idd.launcher.resolve_executable",
        lambda name: f"/venv/bin/{name}",
    )
    args = Namespace(
        worker_endpoint="tcp://127.0.0.1:5555",
        sample_name="sample-a",
        dwell_imaging=0.1,
        dwell_line_scan=0.3,
        no_xrf=True,
        preamp1_on=True,
        using_xrf_maps=True,
        xrf_elms=["Cr", "Fe"],
        xrf_roi_num=8,
        allowable_x_range="0,100",
        allowable_y_range="-50,50",
        allowable_z_range="-200,-180",
        plot_image_in_log_scale=True,
        show_colorbar_in_image=True,
        line_scan_return_gaussian_fit=True,
        no_scan_samy=True,
        request_timeout_ms=30000,
        mcp_host="127.0.0.1",
        mcp_port=8050,
        mcp_path="/mcp",
    )

    worker_command = build_worker_command(args)

    assert "--sample-name" in worker_command
    assert "sample-a" in worker_command
    assert "--no-xrf" in worker_command
    assert "--preamp1-on" in worker_command
    assert "--using-xrf-maps" in worker_command
    assert "--allowable-x-range" in worker_command
    assert "0,100" in worker_command
    assert "--line-scan-return-gaussian-fit" in worker_command
