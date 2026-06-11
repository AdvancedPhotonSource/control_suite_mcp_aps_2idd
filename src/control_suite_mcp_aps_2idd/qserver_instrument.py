"""QueueServer-backed APS 2-ID-D instrument adapter."""

from __future__ import annotations

from typing import Any

from control_suite_mcp_aps_2idd.common import APSTwoIDDConfig, json_safe, validate_position_in_range
from control_suite_mcp_aps_2idd.qserver_client import (
    QServerConnectionConfig,
    RestrictedQServerClient,
    normalize_allowable_ranges,
    result_run_uids,
    result_scan_ids,
)


class QServerAPSTwoIDDMICInstrument:
    """Instrument adapter that delegates approved actions to Bluesky QueueServer."""

    writable_config_names = {
        "sample_name",
        "dwell_imaging",
        "dwell_line_scan",
        "xrf_on",
        "preamp1_on",
        "using_xrf_maps",
        "xrf_elms",
        "xrf_roi_num",
        "allowable_x_range",
        "allowable_y_range",
        "allowable_z_range",
        "plot_image_in_log_scale",
        "show_colorbar_in_image",
        "line_scan_return_gaussian_fit",
        "scan_samy",
    }

    def __init__(
        self,
        config: APSTwoIDDConfig | None = None,
        *,
        qserver_config: QServerConnectionConfig | None = None,
    ) -> None:
        self.config = APSTwoIDDConfig() if config is None else config
        self.qserver = RestrictedQServerClient(
            QServerConnectionConfig.from_env() if qserver_config is None else qserver_config
        )
        self.image_acquisition_call_history: list[dict[str, Any]] = []
        self.line_scan_call_history: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return self.qserver.health()

    def get_state(self) -> dict[str, Any]:
        return json_safe(
            {
                "sample_name": self.config.sample_name,
                "dwell_imaging": self.config.dwell_imaging,
                "dwell_line_scan": self.config.dwell_line_scan,
                "xrf_on": self.config.xrf_on,
                "preamp1_on": self.config.preamp1_on,
                "using_xrf_maps": self.config.using_xrf_maps,
                "xrf_elms": self.config.xrf_elms,
                "xrf_roi_num": self.config.xrf_roi_num,
                "plot_image_in_log_scale": self.config.plot_image_in_log_scale,
                "show_colorbar_in_image": self.config.show_colorbar_in_image,
                "line_scan_return_gaussian_fit": self.config.line_scan_return_gaussian_fit,
                "scan_samy": self.config.scan_samy,
                "image_acquisition_call_history": self.image_acquisition_call_history,
                "line_scan_call_history": self.line_scan_call_history,
                "qserver": self.qserver.status(),
                **normalize_allowable_ranges(
                    allowable_x_range=self.config.allowable_x_range,
                    allowable_y_range=self.config.allowable_y_range,
                    allowable_z_range=self.config.allowable_z_range,
                ),
            }
        )

    def set_config(self, name: str, value: Any) -> dict[str, Any]:
        if name not in self.writable_config_names:
            raise ValueError(f"Unsupported configuration attribute: {name}")
        if name == "xrf_elms":
            value = tuple(value)
        if name.endswith("_range") and value is not None:
            value = tuple(float(item) for item in value)
        if name == "line_scan_return_gaussian_fit":
            value = bool(value)
        object.__setattr__(self.config, name, value)
        return {"name": name, "value": json_safe(value)}

    def set_attribute(self, name: str, value: Any) -> dict[str, Any]:
        return self.set_config(name=name, value=value)

    def dump_array(self, buffer_name: str) -> dict[str, str]:
        raise RuntimeError(
            "QueueServer-backed MCP service does not own in-process image buffers. "
            f"dump_array('{buffer_name}') is unavailable in this mode."
        )

    def get_attribute_payload(self, name: str) -> Any:
        state = self.get_state()
        if name not in state:
            raise AttributeError(name)
        return state[name]

    def acquire_image(
        self,
        width: float,
        height: float,
        x_center: float,
        y_center: float,
        stepsize_x: float,
        stepsize_y: float,
    ) -> dict[str, Any]:
        validate_position_in_range(x_center, self.config.allowable_x_range, "x")
        validate_position_in_range(y_center, self.config.allowable_y_range, "y")
        request = {
            "samplename": self.config.sample_name,
            "width": width,
            "x_center": x_center,
            "stepsize_x": stepsize_x,
            "height": height,
            "y_center": y_center,
            "stepsize_y": stepsize_y,
            "dwell_ms": self.config.dwell_imaging * 1000,
            "xrf_on": self.config.xrf_on,
            "preamp1_on": self.config.preamp1_on,
        }
        self.image_acquisition_call_history.append(
            {
                "x_center": x_center,
                "y_center": y_center,
                "size_x": width,
                "size_y": height,
                "psize_x": stepsize_x,
                "psize_y": stepsize_y,
            }
        )
        execution = self.qserver.run_acquire_image(request)
        task_result = execution["task_result"]
        return json_safe(
            {
                "plan_name": execution["plan_name"],
                "task_uid": execution["task_uid"],
                "run_uids": result_run_uids(task_result),
                "scan_ids": result_scan_ids(task_result),
                "save_data_path": self.qserver.get_save_data_path(timeout=10.0),
                "raw_task_result": task_result,
            }
        )

    def acquire_line_scan(
        self,
        length: float,
        x_center: float,
        y_center: float,
        stepsize_x: float,
    ) -> dict[str, Any]:
        start_x = x_center - length / 2
        end_x = x_center + length / 2
        validate_position_in_range(start_x, self.config.allowable_x_range, "x")
        validate_position_in_range(end_x, self.config.allowable_x_range, "x")
        validate_position_in_range(y_center, self.config.allowable_y_range, "y")
        if self.config.scan_samy:
            self.qserver.move_samy(y_center, timeout=30.0)
        request = {
            "samplename": self.config.sample_name,
            "width": length,
            "center": x_center,
            "stepsize_x": stepsize_x,
            "dwell_ms": self.config.dwell_line_scan * 1000,
            "xrf_on": self.config.xrf_on,
            "preamp1_on": self.config.preamp1_on,
        }
        self.line_scan_call_history.append(
            {
                "step": stepsize_x,
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "angle": 0.0,
            }
        )
        execution = self.qserver.run_acquire_line_scan(request)
        task_result = execution["task_result"]
        return json_safe(
            {
                "plan_name": execution["plan_name"],
                "task_uid": execution["task_uid"],
                "run_uids": result_run_uids(task_result),
                "scan_ids": result_scan_ids(task_result),
                "save_data_path": self.qserver.get_save_data_path(timeout=10.0),
                "raw_task_result": task_result,
            }
        )

    def set_parameters(self, parameters: list[float]) -> Any:
        if not parameters:
            raise ValueError("The 'parameters' list must contain at least one value.")
        value = float(parameters[0])
        validate_position_in_range(value, self.config.allowable_z_range, "z")
        return self.qserver.set_zp_z(value, timeout=30.0)
