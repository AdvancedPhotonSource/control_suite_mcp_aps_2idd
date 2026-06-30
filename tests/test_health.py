"""Tests for the beamline/scan health evaluator (live snapshot schema)."""

from __future__ import annotations

import copy

from control_suite_mcp_aps_2idd.health import evaluate_snapshot


def pv(value, *, connected=True, char_value=None, error=None, timestamp=None):
    return {
        "value": value,
        "char_value": char_value,
        "connected": connected,
        "error": error,
        "timestamp": timestamp,
    }


def healthy_snapshot() -> dict:
    """A live-shaped snapshot: beam delivered, sample idle at setpoint, scan running."""
    return {
        "timestamp": "2026-06-28T21:44:17+00:00",
        "device_names": ["ring", "sample", "scanrecord_fly", "scanrecord_step", "kohzu_mono", "zp_z"],
        "devices": {
            "ring": {
                "pvs": {
                    "current": pv(200.1),
                    "operating_mode": pv("Delivered Beam", char_value="Delivered Beam"),
                    "shutter_status": pv("PERMIT", char_value="PERMIT"),
                }
            },
            "sample": {
                "pvs": {
                    "x.user_setpoint": pv(3531.03),
                    "x.user_readback": pv(3531.03),
                    "x.motor_is_moving": pv(0),
                    "y.user_setpoint": pv(3094.485),
                    "y.user_readback": pv(3094.485),
                    "y.motor_is_moving": pv(0),
                    "z.user_setpoint": pv(-3.5),
                    "z.user_readback": pv(-3.5),
                    "z.motor_is_moving": pv(0),
                }
            },
            "scanrecord_fly": {
                "pvs": {
                    "inner.scan_phase": pv(7, char_value="WAIT:DETCTRS"),
                    "inner.number_points": pv(29),
                    "inner.current_point": pv(0),
                    "outer.scan_phase": pv(7, char_value="WAIT:DETCTRS"),
                    "outer.number_points": pv(31),
                    "pause_signal": pv(0, char_value="GO"),
                    "wait": pv(0),
                }
            },
            "scanrecord_step": {
                "pvs": {
                    "inner.scan_phase": pv(0, char_value="IDLE"),
                    "inner.scan_busy": pv(0),
                }
            },
            "kohzu_mono": {"pvs": {"user_setpoint": pv(12.0001), "user_readback": pv(12.0001)}},
            "zp_z": {"pvs": {"user_setpoint": pv(-140.75), "user_readback": pv(-140.749)}},
        },
    }


def _kinds(report) -> set:
    return {a["kind"] for a in report["anomalies"]}


def test_healthy_snapshot_is_ok():
    report = evaluate_snapshot(healthy_snapshot())
    assert report["overall"] == "ok"
    assert report["anomalies"] == []
    assert report["devices"]["zp_z"] == "ok"
    # No detectors in the snapshot -> detector_hung never fires even mid-scan.
    assert "detector_hung" not in _kinds(report)


def test_sample_hung_fires_when_not_moving_and_off_setpoint():
    snap = healthy_snapshot()
    snap["devices"]["sample"]["pvs"]["x.user_readback"] = pv(3535.0)  # 4 um off, > 0.1
    snap["devices"]["sample"]["pvs"]["x.motor_is_moving"] = pv(0)
    report = evaluate_snapshot(snap)
    assert "sample_hung" in _kinds(report)
    assert report["overall"] == "error"
    assert report["devices"]["sample"] == "error"
    anomaly = next(a for a in report["anomalies"] if a["kind"] == "sample_hung")
    assert "x" in anomaly["axes"]


def test_sample_not_hung_while_moving():
    snap = healthy_snapshot()
    snap["devices"]["sample"]["pvs"]["x.user_readback"] = pv(3535.0)  # off setpoint
    snap["devices"]["sample"]["pvs"]["x.motor_is_moving"] = pv(1)  # but moving
    report = evaluate_snapshot(snap)
    assert "sample_hung" not in _kinds(report)


def test_scan_paused_on_pause_signal():
    snap = healthy_snapshot()
    snap["devices"]["scanrecord_fly"]["pvs"]["pause_signal"] = pv(1, char_value="PAUSE")
    report = evaluate_snapshot(snap)
    assert "scan_paused" in _kinds(report)
    paused = next(a for a in report["anomalies"] if a["kind"] == "scan_paused")
    assert paused["device"] == "scanrecord_fly"
    assert paused["severity"] == "warning"


def test_go_pause_signal_is_not_paused():
    report = evaluate_snapshot(healthy_snapshot())  # pause_signal char_value "GO"
    assert "scan_paused" not in _kinds(report)


def test_low_ring_current_warning():
    snap = healthy_snapshot()
    snap["devices"]["ring"]["pvs"]["current"] = pv(50.0)
    report = evaluate_snapshot(snap)
    assert "low_ring_current" in _kinds(report)
    assert report["devices"]["ring"] == "warning"


def test_no_beam_error():
    snap = healthy_snapshot()
    snap["devices"]["ring"]["pvs"]["current"] = pv(0.2)
    snap["devices"]["ring"]["pvs"]["operating_mode"] = pv("NO BEAM", char_value="NO BEAM")
    report = evaluate_snapshot(snap)
    assert "no_beam" in _kinds(report)
    assert report["overall"] == "error"


def test_overall_ordering_error_beats_warning():
    snap = healthy_snapshot()
    snap["devices"]["ring"]["pvs"]["current"] = pv(50.0)  # warning
    snap["devices"]["sample"]["pvs"]["x.user_readback"] = pv(3535.0)  # error
    report = evaluate_snapshot(snap)
    assert report["overall"] == "error"


def test_bad_snapshot_reports_error():
    report = evaluate_snapshot({"timestamp": "x"})  # no devices mapping
    assert report["overall"] == "error"
    assert "bad_snapshot" in _kinds(report)


def test_disconnected_device_is_error():
    snap = healthy_snapshot()
    for name in ("current", "operating_mode", "shutter_status"):
        snap["devices"]["ring"]["pvs"][name] = pv(None, connected=False)
    report = evaluate_snapshot(snap)
    # ring current unparseable -> falls through to connectivity: none connected = error
    assert report["devices"]["ring"] == "error"


def test_evaluate_does_not_mutate_input():
    snap = healthy_snapshot()
    before = copy.deepcopy(snap)
    evaluate_snapshot(snap)
    assert snap == before


# --- detector_hung -------------------------------------------------------

def _scanning_snapshot(now, phase_ts):
    """A snapshot mid-fly-scan in WAIT:DETCTRS, with xrf + sis3820 acquiring.

    ``phase_ts`` is when the scan last advanced; ``now`` is the snapshot time.
    """
    snap = healthy_snapshot()
    snap["timestamp"] = now
    snap["device_names"] += ["xrf", "sis3820", "fscanh_dwell"]
    fly = snap["devices"]["scanrecord_fly"]["pvs"]
    fly["inner.scan_phase"] = {**pv(7, char_value="WAIT:DETCTRS"), "timestamp": phase_ts}
    fly["inner.current_point"] = {**pv(0), "timestamp": phase_ts}
    snap["devices"]["xrf"] = {
        "pvs": {
            "cam.acquire": pv(1, char_value="Erase"),
            "fileplugin.capture": {**pv(0, char_value="Done"), "timestamp": phase_ts},
            "fileplugin.write_file": pv(0, char_value="Done"),
        }
    }
    snap["devices"]["sis3820"] = {
        "pvs": {"acquiring": pv(1, char_value="Acquiring"), "elapsed_real": pv(120.0)}
    }
    snap["devices"]["fscanh_dwell"] = {"pvs": {"value": pv(50, char_value="50.0")}}  # 50 ms
    return snap


def test_detector_hung_fires_when_scan_frozen():
    # Frozen ~45 s in WAIT:DETCTRS, well past the 30 s floor.
    snap = _scanning_snapshot(
        now="2026-06-28T22:54:22+00:00", phase_ts="2026-06-28T22:53:38+00:00"
    )
    report = evaluate_snapshot(snap)
    kinds = {a["kind"] for a in report["anomalies"]}
    assert "detector_hung" in kinds
    hung_devices = {a["device"] for a in report["anomalies"] if a["kind"] == "detector_hung"}
    assert hung_devices == {"xrf", "sis3820"}
    assert report["overall"] == "error"


def test_detector_not_hung_when_progressing():
    # Only ~4 s since last progress -> normal WAIT:DETCTRS, not a hang.
    snap = _scanning_snapshot(
        now="2026-06-28T22:54:22+00:00", phase_ts="2026-06-28T22:54:18+00:00"
    )
    report = evaluate_snapshot(snap)
    kinds = {a["kind"] for a in report["anomalies"]}
    assert "detector_hung" not in kinds
    assert report["overall"] == "ok"


def test_detector_not_hung_when_paused():
    snap = _scanning_snapshot(
        now="2026-06-28T22:54:22+00:00", phase_ts="2026-06-28T22:53:38+00:00"
    )
    snap["devices"]["scanrecord_fly"]["pvs"]["pause_signal"] = pv(1, char_value="PAUSE")
    report = evaluate_snapshot(snap)
    kinds = {a["kind"] for a in report["anomalies"]}
    assert "detector_hung" not in kinds  # paused, so a stall is expected
    assert "scan_paused" in kinds
