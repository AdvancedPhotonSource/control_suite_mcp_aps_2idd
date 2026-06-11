"""Restricted QueueServer client for the APS 2-ID-D MCP service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import logging
import os

logger = logging.getLogger(__name__)


def _optional_float_pair(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class QServerActionConfig:
    """Allowlisted QueueServer actions exposed to the MCP service."""

    acquire_image_plan: str = "fly2d_scanrecord"
    acquire_line_scan_plan: str = "step1d_scanrecord"
    get_save_data_path_function: str = "get_save_data_path"
    move_samy_function: str | None = None
    set_zp_z_function: str | None = None


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
                acquire_image_plan=os.getenv("QSERVER_ACQUIRE_IMAGE_PLAN", "fly2d_scanrecord"),
                acquire_line_scan_plan=os.getenv(
                    "QSERVER_ACQUIRE_LINE_SCAN_PLAN",
                    "step1d_scanrecord",
                ),
                get_save_data_path_function=os.getenv(
                    "QSERVER_GET_SAVE_DATA_PATH_FUNCTION",
                    "get_save_data_path",
                ),
                move_samy_function=os.getenv("QSERVER_MOVE_SAMY_FUNCTION") or None,
                set_zp_z_function=os.getenv("QSERVER_SET_ZP_Z_FUNCTION") or None,
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
            self.config.actions.get_save_data_path_function,
            timeout=timeout,
        )

    def run_acquire_image(self, kwargs: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Execute the allowlisted 2D acquisition plan."""
        return self._execute_plan(self.config.actions.acquire_image_plan, kwargs, timeout=timeout)

    def run_acquire_line_scan(
        self,
        kwargs: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute the allowlisted 1D acquisition plan."""
        return self._execute_plan(self.config.actions.acquire_line_scan_plan, kwargs, timeout=timeout)

    def move_samy(self, value: float, *, timeout: float | None = None) -> Any:
        """Execute the allowlisted sample-y helper function."""
        function_name = self.config.actions.move_samy_function
        if not function_name:
            raise RuntimeError(
                "MCP action 'move_samy' is disabled. Configure QSERVER_MOVE_SAMY_FUNCTION "
                "to an approved QueueServer helper function."
            )
        return self._execute_function(function_name, call_kwargs={"value": float(value)}, timeout=timeout)

    def set_zp_z(self, value: float, *, timeout: float | None = None) -> Any:
        """Execute the allowlisted zp-z helper function."""
        function_name = self.config.actions.set_zp_z_function
        if not function_name:
            raise RuntimeError(
                "MCP action 'set_zp_z' is disabled. Configure QSERVER_SET_ZP_Z_FUNCTION "
                "to an approved QueueServer helper function."
            )
        return self._execute_function(function_name, call_kwargs={"value": float(value)}, timeout=timeout)

    def _execute_plan(
        self,
        plan_name: str,
        kwargs: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        item = self._BPlan(plan_name, **dict(kwargs))
        logger.info("Submitting QueueServer plan %s with kwargs=%s", plan_name, dict(kwargs))
        response = self._rm.item_execute(
            item,
            user=self.config.user,
            user_group=self.config.user_group,
            lock_key=self.config.lock_key,
        )
        result = self._wait_for_task_result(response, timeout=timeout)
        return {
            "plan_name": plan_name,
            "task_uid": response.get("task_uid"),
            "response": dict(response),
            "task_result": result,
        }

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
) -> dict[str, tuple[float, float] | None]:
    """Normalize allowed position ranges for state reporting."""
    return {
        "allowable_x_range": _optional_float_pair(allowable_x_range),
        "allowable_y_range": _optional_float_pair(allowable_y_range),
        "allowable_z_range": _optional_float_pair(allowable_z_range),
    }
