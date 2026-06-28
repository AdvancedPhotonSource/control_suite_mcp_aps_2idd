"""Tests for the QueueServer plan-execution path (item_execute semantics)."""

from __future__ import annotations

import pytest

from bluesky_queueserver_api.comm_base import RequestTimeoutError

from control_suite_mcp_aps_2idd.qserver_client import (
    QServerConnectionConfig,
    RestrictedQServerClient,
    result_run_uids,
    result_scan_ids,
)


class FakeConsoleMonitor:
    """Minimal console monitor that yields a fixed list of messages then times out."""

    def __init__(self, messages: list[dict[str, object]]) -> None:
        self._messages = list(messages)
        self.enabled = False
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def next_msg(self, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        raise RequestTimeoutError("no message", {})


class FakeRM:
    def __init__(
        self,
        *,
        execute_response: dict[str, object],
        history_items: list[dict[str, object]] | None = None,
        manager_state: str = "idle",
        console_monitor: FakeConsoleMonitor | None = None,
    ) -> None:
        self._execute_response = execute_response
        self._history = {"items": list(history_items or [])}
        self._manager_state = manager_state
        self.console_monitor = console_monitor
        self.executed: list[object] = []
        self.wait_for_idle_called = False

    def item_execute(self, item, *, user=None, user_group=None, lock_key=None):
        self.executed.append(item)
        return self._execute_response

    def wait_for_idle(self, timeout=None):
        self.wait_for_idle_called = True

    def status(self):
        return {"manager_state": self._manager_state}

    def history_get(self):
        return self._history


def make_client(rm: FakeRM) -> RestrictedQServerClient:
    """Build a client bound to a fake RE manager without opening sockets."""
    client = object.__new__(RestrictedQServerClient)
    client.config = QServerConnectionConfig(
        zmq_control_addr="tcp://test:1",
        zmq_info_addr="tcp://test:2",
    )
    client._rm = rm
    client._BPlan = lambda name, **kwargs: {"name": name, "kwargs": kwargs}
    client._BFunc = lambda name, **kwargs: {"name": name, "kwargs": kwargs}
    return client


COMPLETED_HISTORY = [
    {
        "item_uid": "uid-xyz",
        "result": {"exit_status": "completed", "run_uids": ["run-1"], "scan_ids": [7]},
    }
]


def test_execute_plan_waits_for_idle_and_reads_history() -> None:
    rm = FakeRM(
        execute_response={"success": True, "item": {"item_uid": "uid-xyz"}},
        history_items=COMPLETED_HISTORY,
    )
    client = make_client(rm)

    out = client._execute_plan("fly2d_scanrecord", {"width": 10})

    assert out["item_uid"] == "uid-xyz"
    assert "task_uid" not in out
    assert rm.wait_for_idle_called is True
    assert result_run_uids(out["task_result"]) == ["run-1"]
    assert result_scan_ids(out["task_result"]) == [7]


def test_execute_plan_streams_console_messages() -> None:
    console = FakeConsoleMonitor([{"msg": "p1"}, {"msg": "p2"}])
    rm = FakeRM(
        execute_response={"success": True, "item": {"item_uid": "uid-xyz"}},
        history_items=COMPLETED_HISTORY,
        console_monitor=console,
    )
    client = make_client(rm)

    seen: list[dict[str, object]] = []
    out = client._execute_plan("fly2d_scanrecord", {}, on_console=seen.append)

    assert [m["msg"] for m in seen] == ["p1", "p2"]
    assert console.cleared is True
    assert console.enabled is False  # disabled after completion
    assert rm.wait_for_idle_called is False  # console path drives the wait
    assert out["item_uid"] == "uid-xyz"


def test_execute_plan_rejects_failed_submission() -> None:
    rm = FakeRM(execute_response={"success": False, "msg": "plan not allowed"})
    client = make_client(rm)

    with pytest.raises(RuntimeError, match="plan not allowed"):
        client._execute_plan("fly2d_scanrecord", {})


def test_execute_plan_requires_item_uid() -> None:
    rm = FakeRM(execute_response={"success": True, "item": {}})
    client = make_client(rm)

    with pytest.raises(RuntimeError, match="did not return an item UID"):
        client._execute_plan("fly2d_scanrecord", {})


def test_execute_plan_raises_on_failed_exit_status() -> None:
    rm = FakeRM(
        execute_response={"success": True, "item": {"item_uid": "uid-xyz"}},
        history_items=[
            {
                "item_uid": "uid-xyz",
                "result": {"exit_status": "failed", "msg": "motor fault", "traceback": ""},
            }
        ],
    )
    client = make_client(rm)

    with pytest.raises(RuntimeError, match="did not complete .* motor fault"):
        client._execute_plan("fly2d_scanrecord", {})


class FakeRecoveryRM:
    """Fake RE manager for exercising the recover_detector pause/resume flow."""

    def __init__(self, *, initial_state: str = "running") -> None:
        self._state = initial_state
        self.pause_calls: list[str | None] = []
        self.resume_calls: int = 0
        self.functions: list[tuple[object, bool]] = []

    def status(self):
        return {"re_state": self._state}

    def re_pause(self, *, option=None):
        self.pause_calls.append(option)
        self._state = "paused"
        return {"success": True}

    def re_resume(self):
        self.resume_calls += 1
        self._state = "running"
        return {"success": True}

    def function_execute(self, item, *, user=None, user_group=None, lock_key=None, run_in_background=False):
        self.functions.append((item, run_in_background))
        return {"success": True, "task_uid": "task-1"}

    def wait_for_completed_task(self, task_uid, timeout=None):
        pass

    def task_result(self, *, task_uid):
        return {"result": {"return_value": {"device": "xmap", "success": True}}}


def test_recover_detector_pauses_resets_and_resumes() -> None:
    rm = FakeRecoveryRM(initial_state="running")
    client = make_client(rm)

    result = client.recover_detector("xmap", retries=2, settle_time_s=0.0)

    # A running scan is paused immediately, then resumed after the reset.
    assert rm.pause_calls == ["immediate"]
    assert rm.resume_calls == 1
    # The allowlisted recover_detector function is run in the background.
    item, run_in_background = rm.functions[0]
    assert item["name"] == "recover_detector"
    assert item["kwargs"] == {"device_name": "xmap", "retries": 2}
    assert run_in_background is True
    assert result["device"] == "xmap"
    assert result["success"] is True
    assert "RE paused" in result["progress"]
    assert "RE resumed" in result["progress"]


def test_recover_detector_leaves_already_paused_re_paused() -> None:
    rm = FakeRecoveryRM(initial_state="paused")
    client = make_client(rm)

    result = client.recover_detector("eiger", settle_time_s=0.0)

    # Already paused on entry: do not pause or resume; leave the RE paused.
    assert rm.pause_calls == []
    assert rm.resume_calls == 0
    assert rm._state == "paused"
    assert result["success"] is True
