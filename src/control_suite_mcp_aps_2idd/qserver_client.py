"""Restricted QueueServer client for the APS 2-ID-D MCP service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
import logging
import os
import time

logger = logging.getLogger(__name__)


def _optional_float_pair(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class QServerActionConfig:
    """Allowlisted QueueServer item linkage for MCP actions.

    Each field maps one MCP-facing action to a named QueueServer item in the
    worker environment. The MCP service never forwards arbitrary plan or
    function names; it only submits the allowlisted names configured here.

    Current linkage model:
    - ``acquire_image`` backs the MCP ``acquire_image`` tool.
    - ``acquire_line_scan`` backs the MCP ``acquire_line_scan`` tool.
    - ``move_sample`` backs the MCP ``move_sample`` tool and is also reused
      internally when line scans need sample-y positioning.
    - ``move_zp_z`` backs the MCP ``set_parameters`` path for zp-z motion.
    - ``get_save_data_path`` fetches current save-path metadata.

    All motion and acquisition actions are expected to be QueueServer plans in
    the current design. ``get_save_data_path`` remains a QueueServer helper
    function because it is a metadata lookup rather than a motion or scan item.
    """

    acquire_image: str = "fly2d_scanrecord"
    acquire_line_scan: str = "step1d_scanrecord"
    move_sample: str | None = None
    move_zp_z: str | None = None
    get_save_data_path: str = "get_save_data_path"


@dataclass(frozen=True)
class QServerConnectionConfig:
    """Connection and execution settings for QueueServer."""

    zmq_control_addr: str
    zmq_info_addr: str
    user_group: str = "root"
    user: str | None = None
    lock_key: str | None = None
    timeout_s: float = 120.0
    beamline_monitor_manifest_path: str | None = None
    actions: QServerActionConfig = field(default_factory=QServerActionConfig)

    @classmethod
    def from_env(
        cls,
        *,
        default_control_addr: str = "tcp://127.0.0.1:60615",
        default_info_addr: str = "tcp://127.0.0.1:60625",
        default_timeout_s: float = 120.0,
    ) -> "QServerConnectionConfig":
        return cls(
            zmq_control_addr=os.getenv("QSERVER_ZMQ_CONTROL_ADDRESS", default_control_addr),
            zmq_info_addr=os.getenv("QSERVER_ZMQ_INFO_ADDRESS", default_info_addr),
            user_group=os.getenv("QSERVER_USER_GROUP", "root"),
            user=os.getenv("QSERVER_USER"),
            lock_key=os.getenv("QSERVER_LOCK_KEY"),
            timeout_s=float(os.getenv("QSERVER_TIMEOUT_S", str(default_timeout_s))),
            beamline_monitor_manifest_path=os.getenv("QSERVER_BEAMLINE_MONITOR_MANIFEST"),
            actions=QServerActionConfig(
                acquire_image=os.getenv("QSERVER_ACQUIRE_IMAGE_PLAN", "fly2d_scanrecord"),
                acquire_line_scan=os.getenv(
                    "QSERVER_ACQUIRE_LINE_SCAN_PLAN",
                    "step1d_scanrecord",
                ),
                move_sample=os.getenv("QSERVER_MOVE_SAMPLE_PLAN") or None,
                move_zp_z=os.getenv("QSERVER_MOVE_ZP_Z_PLAN") or None,
                get_save_data_path=os.getenv(
                    "QSERVER_GET_SAVE_DATA_PATH_FUNCTION",
                    "get_save_data_path",
                ),
            ),
        )


class RestrictedQServerClient:
    """Headless QueueServer client with a fixed allowlist.

    The client intentionally exposes a small, explicit surface. It is not a
    generic proxy for arbitrary plans or functions.
    """

    def __init__(self, config: QServerConnectionConfig) -> None:
        self.config = config
        try:
            from bluesky_queueserver_api import BFunc, BPlan
            from bluesky_queueserver_api.zmq import REManagerAPI
        except ImportError as exc:
            raise ImportError(
                "bluesky_queueserver_api is required for QueueServer-backed MCP support."
            ) from exc
        self._BFunc = BFunc
        self._BPlan = BPlan
        self._rm = REManagerAPI(
            zmq_control_addr=config.zmq_control_addr,
            zmq_info_addr=config.zmq_info_addr,
        )
        if config.user_group:
            self._rm.user_group = config.user_group
        if config.user is not None:
            self._rm.user = config.user
        if config.lock_key is not None:
            self._rm.lock_key = config.lock_key

    def health(self) -> dict[str, Any]:
        """Return QueueServer connectivity and state."""
        status = self._rm.status()
        return {
            "status": "ok",
            "qserver_connected": True,
            "manager_state": status.get("manager_state"),
            "re_state": status.get("re_state"),
            "worker_environment_exists": status.get("worker_environment_exists"),
            "items_in_queue": status.get("items_in_queue"),
            "running_item_uid": status.get("running_item_uid"),
        }

    def status(self) -> dict[str, Any]:
        """Return raw QueueServer status."""
        return dict(self._rm.status())

    def queue_snapshot(self) -> dict[str, Any]:
        """Return raw queue and history data for inspection."""
        return {
            "queue": dict(self._rm.queue_get()),
            "history": dict(self._rm.history_get()),
        }

    def get_save_data_path(self, *, timeout: float | None = None) -> Any:
        """Run the allowlisted QueueServer helper that reports the data path."""
        return self._execute_function(
            self.config.actions.get_save_data_path,
            timeout=timeout,
        )

    def run_acquire_image(
        self,
        kwargs: Mapping[str, Any],
        *,
        timeout: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the allowlisted 2D acquisition plan."""
        return self._execute_plan(
            self.config.actions.acquire_image,
            kwargs,
            timeout=timeout,
            on_console=on_console,
        )

    def run_acquire_line_scan(
        self,
        kwargs: Mapping[str, Any],
        *,
        timeout: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the allowlisted 1D acquisition plan."""
        return self._execute_plan(
            self.config.actions.acquire_line_scan,
            kwargs,
            timeout=timeout,
            on_console=on_console,
        )

    def move_sample(
        self,
        axis: str,
        position: float,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute the allowlisted sample-axis move plan."""
        plan_name = self.config.actions.move_sample
        if not plan_name:
            raise RuntimeError(
                "MCP action 'move_sample' is disabled. Configure QSERVER_MOVE_SAMPLE_PLAN "
                "to an approved QueueServer plan."
            )
        return self._execute_plan(
            plan_name,
            {"axis": str(axis), "position": float(position)},
            timeout=timeout,
        )

    def move_zp_z(self, value: float, *, timeout: float | None = None) -> dict[str, Any]:
        """Execute the allowlisted zp-z move plan."""
        plan_name = self.config.actions.move_zp_z
        if not plan_name:
            raise RuntimeError(
                "MCP action 'move_zp_z' is disabled. Configure QSERVER_MOVE_ZP_Z_PLAN "
                "to an approved QueueServer plan."
            )
        return self._execute_plan(
            plan_name,
            {"position": float(value)},
            timeout=timeout,
        )

    def _execute_plan(
        self,
        plan_name: str,
        kwargs: Mapping[str, Any],
        *,
        timeout: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run an allowlisted QueueServer plan and wait for it to finish.

        Plans are submitted with ``item_execute``, which starts the item
        immediately and returns an ``item_uid`` (not a ``task_uid`` -- that
        field is only produced by ``function_execute``). Completion is detected
        by waiting for the RE manager to return to idle, and the outcome is read
        back from QueueServer history. When ``on_console`` is supplied, console
        output published on the ZMQ info channel is forwarded to the callback
        while the plan runs so callers can surface live progress.
        """
        item = self._BPlan(plan_name, **dict(kwargs))
        logger.info("Submitting QueueServer plan %s with kwargs=%s", plan_name, dict(kwargs))
        response = self._rm.item_execute(
            item,
            user=self.config.user,
            user_group=self.config.user_group,
            lock_key=self.config.lock_key,
        )
        if not response.get("success", False):
            raise RuntimeError(str(response.get("msg", "QueueServer rejected request")))
        item_uid = self._submitted_item_uid(response)
        result = self._wait_for_plan_result(item_uid, timeout=timeout, on_console=on_console)
        return {
            "plan_name": plan_name,
            "item_uid": item_uid,
            "response": dict(response),
            "task_result": result,
        }

    @staticmethod
    def _submitted_item_uid(response: Mapping[str, Any]) -> str:
        item = response.get("item")
        if isinstance(item, Mapping):
            item_uid = item.get("item_uid")
            if isinstance(item_uid, str) and item_uid:
                return item_uid
        raise RuntimeError(f"QueueServer did not return an item UID: {response}")

    def _wait_for_plan_result(
        self,
        item_uid: str,
        *,
        timeout: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        wait_timeout = timeout or self.config.timeout_s
        if on_console is not None:
            self._stream_console_until_idle(on_console, timeout=wait_timeout)
        else:
            self._rm.wait_for_idle(timeout=wait_timeout)
        return self._plan_result_from_history(item_uid)

    def _stream_console_until_idle(
        self,
        on_console: Callable[[Mapping[str, Any]], None],
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        """Forward ZMQ console output to ``on_console`` until the manager is idle."""
        from bluesky_queueserver_api.comm_base import RequestTimeoutError

        def _emit(message: Mapping[str, Any]) -> None:
            try:
                on_console(message)
            except Exception:  # pragma: no cover - callback must not break the wait
                logger.exception("console callback raised; continuing to wait for plan")

        monitor = self._rm.console_monitor
        deadline = time.monotonic() + timeout if timeout else None
        monitor.clear()
        monitor.enable()
        try:
            while True:
                try:
                    message = monitor.next_msg(timeout=poll_interval)
                except RequestTimeoutError:
                    message = None
                if message:
                    _emit(message)
                    continue
                if self._rm.status().get("manager_state") == "idle":
                    break
                if deadline is not None and time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out after {timeout}s waiting for QueueServer plan to finish."
                    )
            # Drain any console output buffered up to the moment of completion.
            while True:
                try:
                    message = monitor.next_msg(timeout=0.05)
                except RequestTimeoutError:
                    break
                if not message:
                    break
                _emit(message)
        finally:
            try:
                monitor.disable()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("failed to disable console monitor")

    def _plan_result_from_history(self, item_uid: str) -> dict[str, Any]:
        history = self._rm.history_get()
        items = history.get("items", []) if isinstance(history, Mapping) else []
        match: Mapping[str, Any] | None = None
        for entry in reversed(items):
            if isinstance(entry, Mapping) and entry.get("item_uid") == item_uid:
                match = entry
                break
        if match is None:
            raise RuntimeError(
                f"QueueServer plan {item_uid} completed but was not found in history."
            )
        result = match.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError(f"QueueServer history item {item_uid} has no result payload.")
        exit_status = result.get("exit_status")
        if exit_status not in (None, "completed"):
            detail = result.get("msg") or result.get("traceback") or exit_status
            raise RuntimeError(
                f"QueueServer plan {item_uid} did not complete ({exit_status}): {detail}"
            )
        # Wrap so result_run_uids/result_scan_ids/result_return_value keep working.
        return {"result": dict(result), "exit_status": exit_status, "item_uid": item_uid}

    def _execute_function(
        self,
        function_name: str,
        *,
        call_kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        run_in_background: bool = False,
    ) -> Any:
        item = self._BFunc(function_name, **dict(call_kwargs or {}))
        logger.info("Submitting QueueServer function %s with kwargs=%s", function_name, dict(call_kwargs or {}))
        response = self._rm.function_execute(
            item,
            user=self.config.user,
            user_group=self.config.user_group,
            lock_key=self.config.lock_key,
            run_in_background=run_in_background,
        )
        result = self._wait_for_task_result(response, timeout=timeout)
        if isinstance(result, Mapping):
            payload = result.get("result")
            if isinstance(payload, Mapping) and "return_value" in payload:
                return payload.get("return_value")
        return result

    def _wait_for_task_result(
        self,
        response: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not response.get("success", False):
            raise RuntimeError(str(response.get("msg", "QueueServer rejected request")))
        task_uid = response.get("task_uid")
        if not isinstance(task_uid, str) or not task_uid:
            raise RuntimeError(f"QueueServer did not return a task UID: {response}")
        self._rm.wait_for_completed_task(task_uid, timeout=timeout or self.config.timeout_s)
        result = self._rm.task_result(task_uid=task_uid)
        if not isinstance(result, Mapping):
            raise RuntimeError(f"QueueServer returned invalid task result payload: {result!r}")
        return dict(result)


def result_run_uids(task_result: Mapping[str, Any]) -> list[str]:
    """Extract run UIDs from a QueueServer task result payload."""
    payload = task_result.get("result")
    if not isinstance(payload, Mapping):
        return []
    run_uids = payload.get("run_uids")
    if isinstance(run_uids, list):
        return [str(uid) for uid in run_uids]
    return []


def result_scan_ids(task_result: Mapping[str, Any]) -> list[int]:
    """Extract scan IDs from a QueueServer task result payload."""
    payload = task_result.get("result")
    if not isinstance(payload, Mapping):
        return []
    scan_ids = payload.get("scan_ids")
    if isinstance(scan_ids, list):
        values: list[int] = []
        for scan_id in scan_ids:
            try:
                values.append(int(scan_id))
            except (TypeError, ValueError):
                continue
        return values
    return []


def normalize_allowable_ranges(
    *,
    allowable_x_range: tuple[float, float] | None,
    allowable_y_range: tuple[float, float] | None,
    allowable_z_range: tuple[float, float] | None,
    allowable_zp_range: tuple[float, float] | None = None,
    allowable_energy_range: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float] | None]:
    """Normalize allowed position ranges for state reporting."""
    return {
        "allowable_x_range": _optional_float_pair(allowable_x_range),
        "allowable_y_range": _optional_float_pair(allowable_y_range),
        "allowable_z_range": _optional_float_pair(allowable_z_range),
        "allowable_zp_range": _optional_float_pair(allowable_zp_range),
        "allowable_energy_range": _optional_float_pair(allowable_energy_range),
    }


def result_return_value(task_result: Mapping[str, Any]) -> Any:
    """Extract the raw return value from a QueueServer task result payload."""
    payload = task_result.get("result")
    if not isinstance(payload, Mapping):
        return None
    return payload.get("return_value")
