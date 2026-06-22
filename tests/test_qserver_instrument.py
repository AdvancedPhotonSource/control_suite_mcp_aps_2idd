"""Tests for QueueServer-backed instrument helpers."""

from __future__ import annotations

import pytest

import control_suite_mcp_aps_2idd.qserver_instrument as qserver_instrument
from control_suite_mcp_aps_2idd.common import APSTwoIDDConfig


class DummyQServer:
    def __init__(self) -> None:
        self.move_sample_calls: list[tuple[str, float, float | None]] = []
        self.move_zp_z_calls: list[tuple[float, float | None]] = []
        self.acquire_image_calls: list[tuple[dict[str, object], float | None]] = []
        self.acquire_line_scan_calls: list[tuple[dict[str, object], float | None]] = []

    def move_sample(self, axis: str, position: float, *, timeout: float | None = None) -> dict[str, object]:
        self.move_sample_calls.append((axis, position, timeout))
        return {
            "plan_name": "move_sample",
            "item_uid": "item-1",
            "task_result": {"result": {"return_value": position}, "exit_status": "completed"},
        }

    def move_zp_z(self, position: float, *, timeout: float | None = None) -> dict[str, object]:
        self.move_zp_z_calls.append((position, timeout))
        return {
            "plan_name": "move_zp_z",
            "item_uid": "item-2",
            "task_result": {"result": {"return_value": position}, "exit_status": "completed"},
        }

    def run_acquire_image(
        self,
        kwargs: dict[str, object],
        *,
        timeout: float | None = None,
        on_console=None,
    ) -> dict[str, object]:
        self.acquire_image_calls.append((dict(kwargs), timeout))
        if on_console is not None:
            on_console({"time": 0.0, "msg": "image scan point 1"})
        return {
            "plan_name": "fly2d_scanrecord",
            "item_uid": "item-img",
            "task_result": {
                "result": {"run_uids": ["run-img"], "scan_ids": [42]},
                "exit_status": "completed",
            },
        }

    def run_acquire_line_scan(
        self,
        kwargs: dict[str, object],
        *,
        timeout: float | None = None,
        on_console=None,
    ) -> dict[str, object]:
        self.acquire_line_scan_calls.append((dict(kwargs), timeout))
        if on_console is not None:
            on_console({"time": 0.0, "msg": "line scan point 1"})
        return {
            "plan_name": "step1d_scanrecord",
            "item_uid": "item-line",
            "task_result": {
                "result": {"run_uids": ["run-line"], "scan_ids": [43]},
                "exit_status": "completed",
            },
        }

    def get_save_data_path(self, *, timeout: float | None = None) -> str:
        return "/data/smp1"


def test_move_sample_delegates_to_qserver(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_x_range=(0.0, 10.0))
    )

    result = instrument.move_sample("x", 4.5)

    assert dummy.move_sample_calls == [("x", 4.5, 30.0)]
    assert result["readback"] == 4.5
    assert result["plan_name"] == "move_sample"
    assert result["item_uid"] == "item-1"
    assert result["exit_status"] == "completed"


def test_move_sample_rejects_unknown_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument()

    with pytest.raises(ValueError, match="Axis must be one of 'x', 'y', or 'z'."):
        instrument.move_sample("theta", 1.0)


def test_set_parameters_uses_move_zp_z_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_zp_range=(0.0, 10.0))
    )

    result = instrument.set_parameters([6.25])

    assert dummy.move_zp_z_calls == [(6.25, 30.0)]
    assert result == 6.25


def test_move_zp_z_delegates_and_reports_item_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_zp_range=(-2000.0, 2000.0))
    )

    result = instrument.move_zp_z(150.0)

    assert dummy.move_zp_z_calls == [(150.0, 30.0)]
    assert result["position"] == 150.0
    assert result["item_uid"] == "item-2"
    assert result["plan_name"] == "move_zp_z"


def test_move_zp_z_validates_against_zp_range_not_z_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    # A wide sample-z range must NOT permit a zp-z move outside the zp range.
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_z_range=(-5000.0, 5000.0), allowable_zp_range=(-10.0, 10.0))
    )

    with pytest.raises(ValueError, match="zp-z direction"):
        instrument.move_zp_z(50.0)
    assert dummy.move_zp_z_calls == []


def test_acquire_image_streams_console_and_reports_item_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(
            allowable_x_range=(0.0, 100.0),
            allowable_y_range=(-500.0, 500.0),
        )
    )

    messages: list[dict[str, object]] = []
    result = instrument.acquire_image(
        width=10.0,
        height=10.0,
        x_center=5.0,
        y_center=5.0,
        stepsize_x=1.0,
        stepsize_y=1.0,
        on_console=messages.append,
    )

    assert messages == [{"time": 0.0, "msg": "image scan point 1"}]
    assert result["item_uid"] == "item-img"
    assert result["run_uids"] == ["run-img"]
    assert result["scan_ids"] == [42]
    assert result["save_data_path"] == "/data/smp1"


def test_acquire_line_scan_streams_console_and_passes_positioner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(
            allowable_x_range=(0.0, 100.0),
            allowable_y_range=(-500.0, 500.0),
        )
    )

    messages: list[dict[str, object]] = []
    result = instrument.acquire_line_scan(
        positioner_name="x",
        length=10.0,
        center=5.0,
        stepsize=1.0,
        sample_y=-340.0,
        on_console=messages.append,
    )

    assert messages == [{"time": 0.0, "msg": "line scan point 1"}]
    assert result["item_uid"] == "item-line"
    assert result["run_uids"] == ["run-line"]
    assert result["scan_ids"] == [43]
    # The new step1d_scanrecord request carries the positioner and sample moves;
    # the legacy separate move_sample-y step is gone.
    sent_request, _timeout = dummy.acquire_line_scan_calls[0]
    assert sent_request["positioner_name"] == "x"
    assert sent_request["center"] == 5.0
    assert sent_request["sample_y"] == -340.0
    # Unset positions are omitted (not sent as None) so step1d_scanrecord keeps
    # the current position instead of failing its float validation.
    assert "sample_x" not in sent_request
    assert "sample_z" not in sent_request
    assert "energy" not in sent_request
    assert dummy.move_sample_calls == []


def test_acquire_line_scan_center_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_x_range=(0.0, 100.0))
    )

    # Omitting center scans symmetrically around the current position (center=0).
    instrument.acquire_line_scan(positioner_name="x", length=10.0, stepsize=1.0)

    sent_request, _timeout = dummy.acquire_line_scan_calls[0]
    assert sent_request["center"] == 0.0


def test_acquire_line_scan_dwell_ms_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_x_range=(0.0, 100.0), dwell_line_scan=0.2)
    )

    instrument.acquire_line_scan(
        positioner_name="x", length=5.0, center=0.0, stepsize=1.0, dwell_ms=100.0
    )
    overridden, _ = dummy.acquire_line_scan_calls[0]
    assert overridden["dwell_ms"] == 100.0

    # Omitting dwell_ms falls back to the configured dwell_line_scan (0.2 s).
    instrument.acquire_line_scan(
        positioner_name="x", length=5.0, center=0.0, stepsize=1.0
    )
    default, _ = dummy.acquire_line_scan_calls[1]
    assert default["dwell_ms"] == 200.0


def test_acquire_line_scan_validates_energy_extent_when_target_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_energy_range=(5.0, 30.0))
    )

    # Absolute energy target (6 keV) is in range, but the relative scan extent
    # (6 + 0 +/- 2 = [4, 8] keV) drops below the 5 keV lower bound.
    with pytest.raises(ValueError, match="keV"):
        instrument.acquire_line_scan(
            positioner_name="energy",
            length=4.0,
            center=0.0,
            stepsize=0.5,
            energy=6.0,
        )


def test_acquire_line_scan_validates_absolute_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_y_range=(-500.0, 500.0))
    )

    # An explicit absolute sample position outside its range is rejected.
    with pytest.raises(ValueError, match="y direction"):
        instrument.acquire_line_scan(
            positioner_name="x",
            length=10.0,
            center=0.0,
            stepsize=1.0,
            sample_y=-600.0,
        )


def test_acquire_line_scan_relative_center_skips_absolute_extent_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument(
        APSTwoIDDConfig(allowable_x_range=(0.0, 100.0))
    )

    # center is RELATIVE; with no absolute sample_x target the current position is
    # unknown, so a large relative center must NOT be rejected against the range.
    result = instrument.acquire_line_scan(
        positioner_name="x",
        length=10.0,
        center=5000.0,
        stepsize=1.0,
    )

    assert result["item_uid"] == "item-line"


def test_acquire_line_scan_rejects_unknown_positioner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = DummyQServer()
    monkeypatch.setattr(qserver_instrument, "RestrictedQServerClient", lambda config: dummy)
    instrument = qserver_instrument.QServerAPSTwoIDDMICInstrument()

    with pytest.raises(ValueError, match="positioner_name must be one of"):
        instrument.acquire_line_scan(
            positioner_name="theta",
            length=10.0,
            center=0.0,
            stepsize=1.0,
        )
