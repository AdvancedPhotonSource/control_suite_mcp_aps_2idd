"""Beamline and scan health evaluation for APS 2-ID-D monitoring snapshots.

This module is the source of truth for *how* the health of the beamline and an
in-progress scan is judged from a device snapshot. It is intentionally pure: it
takes a snapshot dictionary and returns verdicts. It does not talk to EPICS, the
QueueServer, or any MCP server, so it is safe to run repeatedly inside a
monitoring loop.

It is invoked server-side by the MCP ``get_global_health_snapshot`` tool
(``qserver_instrument.get_global_health_snapshot``): after QueueServer returns
the raw snapshot, ``evaluate_snapshot`` runs and the verdict is attached to the
result under the ``evaluation`` key.

Snapshot shape expected (as produced by the QueueServer ``get_global_health_snapshot``
helper for APS 2-ID-D)::

    {
        "timestamp": "2026-06-28T21:44:17+00:00",   # ISO-8601, optional
        "devices": {
            "ring":            {"pvs": {"current": {...}, "operating_mode": {...}, "shutter_status": {...}}},
            "sample":          {"pvs": {"x.user_setpoint": {...}, "x.user_readback": {...}, "x.motor_is_moving": {...}, ...}},
            "scanrecord_fly":  {"pvs": {"inner.scan_phase": {...}, "inner.number_points": {...}, "pause_signal": {...}, "wait": {...}, ...}},
            "scanrecord_step": {"pvs": {"inner.scan_phase": {...}, "inner.scan_busy": {...}, ...}},
            "kohzu_mono":      {"pvs": {"user_setpoint": {...}, "user_readback": {...}}},
            "zp_z":            {"pvs": {"user_setpoint": {...}, "user_readback": {...}}},
            ...
        }
    }

Each PV entry is a mapping such as
``{"value": ..., "char_value": ..., "connected": bool, "error": ..., "timestamp": "..."}``.

Detector devices for ``detector_hung`` are listed in ``DETECTOR_DEVICES``
(``xrf``, ``tmm1``, ``sis3820`` for 2-ID-D); ``tmm1`` activates automatically if
it is re-registered and added to the snapshot. ``detector_hung`` fires when the
scan is sitting in ``WAIT:DETCTRS`` with a detector acquiring but the scan has
made no progress for longer than ``max(MIN_DETECTOR_TIMEOUT_S,
DETECTOR_TIMEOUT_FACTOR * inner_points * dwell)``, where dwell comes from the
``fscanh_dwell`` device. The scan-progress signal (not a per-detector capture
timestamp) is used because the scaler exposes no capture timestamp and the area
detector's only updates once per fly line.

CLI usage (for standalone debugging)::

    python health.py --snapshot snapshot.json
    some_tool_that_emits_snapshot_json | python health.py
    python health.py --snapshot snapshot.json --json   # machine-readable report

Exit code reflects the worst severity found: 0 = ok, 1 = warning, 2 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

DETECTOR_TIMEOUT_FACTOR = 3.0
SAMPLE_POSITION_TOLERANCE = 0.1
# Absolute floor for the detector-hang timeout. A fly line's wall time is
# dominated by fixed overhead (arming, settling, return), not just
# points * dwell, so a points*dwell-only timeout false-positives at every line
# boundary. The floor gives margin; the points*dwell term takes over for
# long-dwell scans.
MIN_DETECTOR_TIMEOUT_S = 30.0

DETECTOR_DEVICES = ("xrf", "tmm1", "sis3820")
SCANRECORD_DEVICES = ("scanrecord_fly", "scanrecord_step")
# Devices that report the fly-scan dwell time (value in ms). First match wins.
DWELL_DEVICES = ("fscanh_dwell", "fly_dwell")


def configure_health(
    *,
    detector_timeout_factor: float | None = None,
    sample_position_tolerance: float | None = None,
    min_detector_timeout_s: float | None = None,
) -> None:
    global DETECTOR_TIMEOUT_FACTOR, SAMPLE_POSITION_TOLERANCE, MIN_DETECTOR_TIMEOUT_S
    if detector_timeout_factor is not None:
        DETECTOR_TIMEOUT_FACTOR = float(detector_timeout_factor)
    if sample_position_tolerance is not None:
        SAMPLE_POSITION_TOLERANCE = float(sample_position_tolerance)
    if min_detector_timeout_s is not None:
        MIN_DETECTOR_TIMEOUT_S = float(min_detector_timeout_s)


def _pv_value(pv: Any) -> Any:
    if isinstance(pv, Mapping):
        value = pv.get("char_value")
        if value not in (None, ""):
            return value
        return pv.get("value")
    return None


def _truthy_pv(pv: Any) -> bool:
    value = _pv_value(pv)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "0", "false", "done", "idle", "off", "go"}
    return bool(value)


def _pv_at_least_one(pv: Any) -> bool:
    value = _pv_value(pv)
    try:
        return float(value) >= 1
    except Exception:
        return False


def _pv_float(pv: Any) -> float | None:
    value = _pv_value(pv)
    try:
        return float(value)
    except Exception:
        return None


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _scanrecord_devices(snapshot: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Return (name, pvs) for each scanrecord device present in the snapshot."""
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return []
    found: list[tuple[str, Mapping[str, Any]]] = []
    for name in SCANRECORD_DEVICES:
        device = devices.get(name)
        if not isinstance(device, Mapping):
            continue
        pvs = device.get("pvs")
        if isinstance(pvs, Mapping):
            found.append((name, pvs))
    return found


def _scanrecord_paused(pvs: Mapping[str, Any]) -> bool:
    return (
        _truthy_pv(pvs.get("pause_signal"))
        or _pv_at_least_one(pvs.get("wait"))
        or _pv_at_least_one(pvs.get("inner.wait"))
        or _pv_at_least_one(pvs.get("outer.wait"))
    )


def _sample_hung_axes(snapshot: Mapping[str, Any]) -> list[str]:
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return []
    sample = devices.get("sample")
    if not isinstance(sample, Mapping):
        return []
    pvs = sample.get("pvs")
    if not isinstance(pvs, Mapping):
        return []

    # Legacy device-level busy gate (older schema); if present and busy, skip.
    if _truthy_pv(pvs.get("busy")):
        return []

    axes: list[str] = []
    # Setpoint/readback key pairs per axis: live schema first, legacy fallbacks.
    axis_checks = {
        "x": [
            ("x.user_setpoint", "x.user_readback"),
            ("x.piezo.setpoint", "x.piezo.readback"),
            ("x.stepper.setpoint", "x.stepper.readback"),
        ],
        "y": [
            ("y.user_setpoint", "y.user_readback"),
            ("y.piezo.setpoint", "y.piezo.readback"),
            ("y.stepper.setpoint", "y.stepper.readback"),
        ],
        "z": [
            ("z.user_setpoint", "z.user_readback"),
            ("z.setpoint", "z.readback"),
        ],
        "theta": [
            ("theta.user_setpoint", "theta.user_readback"),
            ("theta.setpoint", "theta.readback"),
        ],
    }
    for axis, pairs in axis_checks.items():
        # If this axis reports it is moving, it is not hung.
        if _truthy_pv(pvs.get(f"{axis}.motor_is_moving")):
            continue
        for setpoint_key, readback_key in pairs:
            setpoint = _pv_float(pvs.get(setpoint_key))
            readback = _pv_float(pvs.get(readback_key))
            if setpoint is None or readback is None:
                continue
            if abs(setpoint - readback) > SAMPLE_POSITION_TOLERANCE:
                axes.append(axis)
            break  # first present pair wins for this axis
    return axes


def _ring_current(snapshot: Mapping[str, Any]) -> float | None:
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return None
    ring = devices.get("ring")
    if not isinstance(ring, Mapping):
        return None
    pvs = ring.get("pvs")
    if not isinstance(pvs, Mapping):
        return None
    try:
        return float(_pv_value(pvs.get("current")))
    except Exception:
        return None


def _ring_mode(snapshot: Mapping[str, Any]) -> str:
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return ""
    ring = devices.get("ring")
    if not isinstance(ring, Mapping):
        return ""
    pvs = ring.get("pvs")
    if not isinstance(pvs, Mapping):
        return ""
    value = _pv_value(pvs.get("operating_mode"))
    return str(value or "")


def _fly_dwell_seconds(snapshot: Mapping[str, Any]) -> float | None:
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return None
    for name in DWELL_DEVICES:
        dwell = devices.get(name)
        if not isinstance(dwell, Mapping):
            continue
        pvs = dwell.get("pvs")
        if not isinstance(pvs, Mapping):
            continue
        try:
            dwell_ms = float(_pv_value(pvs.get("value")))
        except Exception:
            continue
        return max(0.0, dwell_ms / 1000.0)
    return None


def _inner_scan_point_count(snapshot: Mapping[str, Any]) -> float | None:
    for _name, pvs in _scanrecord_devices(snapshot):
        try:
            count = float(_pv_value(pvs.get("inner.number_points")))
        except Exception:
            continue
        if count:
            return count
    return None


def _scan_phase_waiting_for_detectors(snapshot: Mapping[str, Any]) -> bool:
    waiting_phases = {"WAIT:DETCTRS", "WAIT:AFTER_SCAN"}
    for _name, pvs in _scanrecord_devices(snapshot):
        for key in ("inner.scan_phase", "outer.scan_phase"):
            phase = _pv_value(pvs.get(key))
            if isinstance(phase, str) and phase.strip() in waiting_phases:
                return True
    return False


def _scan_progress_age(snapshot: Mapping[str, Any]) -> float | None:
    """Seconds since the scan last advanced.

    Returns the age (now - newest scan-progress PV timestamp) across the
    scanrecord phase/point PVs. A small value means the scan is actively
    advancing; a large value means it is frozen.
    """
    now_ts = _parse_timestamp(snapshot.get("timestamp"))
    if now_ts is None:
        return None
    latest: datetime | None = None
    for _name, pvs in _scanrecord_devices(snapshot):
        for key in (
            "inner.scan_phase",
            "inner.current_point",
            "outer.scan_phase",
            "outer.current_point",
        ):
            pv = pvs.get(key)
            if not isinstance(pv, Mapping):
                continue
            ts = _parse_timestamp(pv.get("timestamp"))
            if ts is not None and (latest is None or ts > latest):
                latest = ts
    if latest is None:
        return None
    return (now_ts - latest).total_seconds()


def _detector_timeout_seconds(snapshot: Mapping[str, Any]) -> float:
    """Allowed seconds without scan progress before a detector is 'hung'."""
    dwell_seconds = _fly_dwell_seconds(snapshot)
    inner_point_count = _inner_scan_point_count(snapshot)
    if dwell_seconds and inner_point_count and dwell_seconds > 0 and inner_point_count > 0:
        return max(MIN_DETECTOR_TIMEOUT_S, DETECTOR_TIMEOUT_FACTOR * inner_point_count * dwell_seconds)
    return MIN_DETECTOR_TIMEOUT_S


def _detector_acquiring(pvs: Mapping[str, Any]) -> bool:
    """Whether a detector reports it is acquiring/counting.

    Handles both area-detector-style devices (``cam.acquire`` /
    ``fileplugin.*``) and scaler-style devices (``acquiring``).
    """
    if "acquiring" in pvs:  # scaler / MCA style (e.g. sis3820)
        return _truthy_pv(pvs.get("acquiring"))
    return (
        _truthy_pv(pvs.get("cam.acquire"))
        or _truthy_pv(pvs.get("fileplugin.capture"))
        or _truthy_pv(pvs.get("fileplugin.write_file"))
    )


def _detector_hung(device_name: str, snapshot: Mapping[str, Any]) -> bool:
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return False
    device = devices.get(device_name)
    if not isinstance(device, Mapping):
        return False
    pvs = device.get("pvs")
    if not isinstance(pvs, Mapping):
        return False

    scanrecords = _scanrecord_devices(snapshot)
    if not scanrecords:
        return False
    # If any scanrecord is paused, a stalled detector is not an anomaly.
    if any(_scanrecord_paused(sr_pvs) for _name, sr_pvs in scanrecords):
        return False
    if not _scan_phase_waiting_for_detectors(snapshot):
        return False

    ring_current = _ring_current(snapshot)
    if ring_current is None or ring_current <= 0:
        return False

    # The detector must be acquiring for its stall to count as a hang.
    if not _detector_acquiring(pvs):
        return False

    # The robust signal: the scan is sitting in WAIT:DETCTRS and has made no
    # progress for longer than the timeout. (Per-detector capture timestamps are
    # unreliable here -- the scaler has none, and the area detector's only
    # updates once per fly line.)
    stall_seconds = _scan_progress_age(snapshot)
    if stall_seconds is None:
        return False
    return stall_seconds > _detector_timeout_seconds(snapshot)


def evaluate_device_health(device_name: str, device: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    pvs = device.get("pvs")
    if not isinstance(pvs, Mapping) or not pvs:
        return "warning"

    if device_name == "ring":
        current = _ring_current(snapshot)
        mode = _ring_mode(snapshot).upper()
        if current is not None and current < 10 and "NO BEAM" in mode:
            return "error"
        if current is not None and current < 100:
            return "warning"

    if device_name in SCANRECORD_DEVICES and _scanrecord_paused(pvs):
        return "warning"

    if device_name == "sample" and _sample_hung_axes(snapshot):
        return "error"

    if device_name in DETECTOR_DEVICES and _detector_hung(device_name, snapshot):
        return "error"

    connected = 0
    disconnected = 0
    degraded = 0
    for pv in pvs.values():
        if not isinstance(pv, Mapping):
            degraded += 1
            continue
        if pv.get("error"):
            degraded += 1
        if pv.get("connected"):
            connected += 1
        else:
            disconnected += 1

    if connected and not disconnected and not degraded:
        return "ok"
    if connected:
        return "warning"
    return "error"


def detector_hung(device_name: str, snapshot: Mapping[str, Any]) -> bool:
    return _detector_hung(device_name, snapshot)


def sample_hung_axes(snapshot: Mapping[str, Any]) -> list[str]:
    return _sample_hung_axes(snapshot)


def evaluate_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a full snapshot and return a structured health report.

    Returns a dict with:
      - ``overall``: worst severity across all devices ("ok" | "warning" | "error")
      - ``devices``: {device_name: severity}
      - ``anomalies``: list of actionable findings, each with ``kind``,
        ``device``/``axes``, ``severity``, and a human-readable ``message``.
      - ``ring``: {"current", "mode"} for quick reference.
    """
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return {
            "overall": "error",
            "devices": {},
            "anomalies": [
                {
                    "kind": "bad_snapshot",
                    "severity": "error",
                    "message": "Snapshot has no 'devices' mapping; cannot evaluate health.",
                }
            ],
            "ring": {"current": None, "mode": ""},
        }

    severities: dict[str, str] = {}
    anomalies: list[dict[str, Any]] = []

    for device_name, device in devices.items():
        if not isinstance(device, Mapping):
            severities[device_name] = "warning"
            continue
        severities[device_name] = evaluate_device_health(device_name, device, snapshot)

    # Surface specific, actionable anomalies (these drive response actions).
    hung_axes = _sample_hung_axes(snapshot)
    if hung_axes:
        anomalies.append(
            {
                "kind": "sample_hung",
                "axes": hung_axes,
                "severity": "error",
                "message": (
                    "Sample appears stuck: axes "
                    f"{', '.join(hung_axes)} not at setpoint while not moving."
                ),
            }
        )

    for det in DETECTOR_DEVICES:
        if det in devices and _detector_hung(det, snapshot):
            anomalies.append(
                {
                    "kind": "detector_hung",
                    "device": det,
                    "severity": "error",
                    "message": (
                        f"Detector '{det}' looks hung: scan is waiting for detectors "
                        "and acquisition has not advanced within the expected timeout. "
                        "Candidate for detector recovery (unhang)."
                    ),
                }
            )

    for name, sr_pvs in _scanrecord_devices(snapshot):
        if _scanrecord_paused(sr_pvs):
            anomalies.append(
                {
                    "kind": "scan_paused",
                    "device": name,
                    "severity": "warning",
                    "message": f"Scan record '{name}' is paused / waiting (pause signal or wait set).",
                }
            )

    ring_current = _ring_current(snapshot)
    ring_mode = _ring_mode(snapshot)
    if ring_current is not None:
        if ring_current < 10 and "NO BEAM" in ring_mode.upper():
            anomalies.append(
                {
                    "kind": "no_beam",
                    "device": "ring",
                    "severity": "error",
                    "message": f"Storage ring has no beam (current={ring_current}, mode='{ring_mode}').",
                }
            )
        elif ring_current < 100:
            anomalies.append(
                {
                    "kind": "low_ring_current",
                    "device": "ring",
                    "severity": "warning",
                    "message": f"Storage ring current is low (current={ring_current} mA).",
                }
            )

    order = {"ok": 0, "warning": 1, "error": 2}
    overall = "ok"
    for sev in severities.values():
        if order.get(sev, 0) > order.get(overall, 0):
            overall = sev

    return {
        "overall": overall,
        "devices": severities,
        "anomalies": anomalies,
        "ring": {"current": ring_current, "mode": ring_mode},
    }


def _format_report(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Overall health: {report.get('overall', 'unknown').upper()}")
    ring = report.get("ring") or {}
    lines.append(f"Ring: current={ring.get('current')} mode='{ring.get('mode')}'")
    lines.append("")
    lines.append("Per-device status:")
    for name, sev in sorted((report.get("devices") or {}).items()):
        lines.append(f"  - {name}: {sev}")
    anomalies = report.get("anomalies") or []
    lines.append("")
    if anomalies:
        lines.append("Anomalies (act on these):")
        for a in anomalies:
            lines.append(f"  [{a.get('severity', '?').upper()}] {a.get('kind')}: {a.get('message')}")
    else:
        lines.append("Anomalies: none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate APS 2-ID-D beamline and scan health from a device snapshot."
    )
    parser.add_argument(
        "--snapshot",
        help="Path to a snapshot JSON file. If omitted, reads JSON from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable report as JSON instead of text.",
    )
    parser.add_argument(
        "--detector-timeout-factor",
        type=float,
        default=None,
        help=f"Override detector hang timeout factor (default {DETECTOR_TIMEOUT_FACTOR}).",
    )
    parser.add_argument(
        "--sample-position-tolerance",
        type=float,
        default=None,
        help=f"Override sample setpoint/readback tolerance (default {SAMPLE_POSITION_TOLERANCE}).",
    )
    parser.add_argument(
        "--min-detector-timeout-s",
        type=float,
        default=None,
        help=f"Override the detector-hang timeout floor in seconds (default {MIN_DETECTOR_TIMEOUT_S}).",
    )
    args = parser.parse_args(argv)

    configure_health(
        detector_timeout_factor=args.detector_timeout_factor,
        sample_position_tolerance=args.sample_position_tolerance,
        min_detector_timeout_s=args.min_detector_timeout_s,
    )

    if args.snapshot:
        with open(args.snapshot, "r") as fh:
            snapshot = json.load(fh)
    else:
        snapshot = json.load(sys.stdin)

    report = evaluate_snapshot(snapshot)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))

    return {"ok": 0, "warning": 1, "error": 2}.get(report.get("overall", "error"), 2)


__all__ = [
    "configure_health",
    "detector_hung",
    "evaluate_device_health",
    "evaluate_snapshot",
    "sample_hung_axes",
]


if __name__ == "__main__":
    raise SystemExit(main())
