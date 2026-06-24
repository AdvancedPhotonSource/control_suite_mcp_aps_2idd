"""QueueServer-backed APS 2-ID-D instrument adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from control_suite_mcp_aps_2idd.acquisition_processing import APSMICPostProcessor
from control_suite_mcp_aps_2idd.common import (
    APSTwoIDDConfig,
    json_safe,
    validate_position_in_range,
)
from control_suite_mcp_aps_2idd.qserver_client import (
    QServerConnectionConfig,
    RestrictedQServerClient,
    normalize_allowable_ranges,
    result_return_value,
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
        "allowable_zp_range",
        "allowable_energy_range",
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
        self.postprocessor = APSMICPostProcessor()
        self.image_acquisition_call_history: list[dict[str, Any]] = []
        self.line_scan_call_history: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return self.qserver.health()

    def get_current_mda_file(self) -> dict[str, Any]:
        return json_safe(
            {"current_mda_file": self.qserver.get_current_mda_file(timeout=10.0)}
        )

    def get_save_data_path(self) -> dict[str, Any]:
        return json_safe(
            {"save_data_path": self.qserver.get_save_data_path(timeout=10.0)}
        )

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
                    allowable_zp_range=self.config.allowable_zp_range,
                    allowable_energy_range=self.config.allowable_energy_range,
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
        dwell_ms: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        validate_position_in_range(x_center, self.config.allowable_x_range, "x")
        validate_position_in_range(y_center, self.config.allowable_y_range, "y")
        dwell = self.config.dwell_imaging * 1000 if dwell_ms is None else dwell_ms
        request = {
            "samplename": self.config.sample_name,
            "width": width,
            "x_center": x_center,
            "stepsize_x": stepsize_x,
            "height": height,
            "y_center": y_center,
            "stepsize_y": stepsize_y,
            "dwell_ms": dwell,
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
        # Captured before the scan: next_file_name auto-increments once the plan
        # runs, so reading it here yields the file this scan actually writes.
        current_mda_file = self.qserver.get_current_mda_file(timeout=10.0)
        execution = self.qserver.run_acquire_image(request, on_console=on_console)
        task_result = execution["task_result"]
        save_data_path = self.qserver.get_save_data_path(timeout=10.0)
        postprocessed = self.postprocessor.process_image(
            save_data_path=save_data_path,
            current_mda_file=current_mda_file,
            channels=self.config.xrf_elms,
            using_xrf_maps=self.config.using_xrf_maps,
            plot_in_log_scale=self.config.plot_image_in_log_scale,
            show_colorbar=self.config.show_colorbar_in_image,
        )
        return json_safe(
            {
                "plan_name": execution["plan_name"],
                "item_uid": execution["item_uid"],
                "run_uids": result_run_uids(task_result),
                "scan_ids": result_scan_ids(task_result),
                "save_data_path": save_data_path,
                "current_mda_file": current_mda_file,
                **postprocessed,
                "raw_task_result": task_result,
            }
        )

    def process_image(
        self,
        current_mda_file: str,
        save_data_path: str | None = None,
        channels: list[str] | None = None,
        using_xrf_maps: bool | None = None,
        plot_in_log_scale: bool | None = None,
        show_colorbar: bool | None = None,
    ) -> dict[str, Any]:
        """Post-process an already-acquired MDA file into PNG/NPY artifacts.

        Reuses the same postprocessor as ``acquire_image`` but on existing
        data, so an image can be (re)visualized without running a new scan.
        ``save_data_path`` defaults to the current QueueServer save path; the
        visualization options default to the service configuration.
        """
        resolved_path = (
            self.qserver.get_save_data_path(timeout=10.0)
            if save_data_path is None
            else save_data_path
        )
        postprocessed = self.postprocessor.process_image(
            save_data_path=resolved_path,
            current_mda_file=current_mda_file,
            channels=(
                self.config.xrf_elms if channels is None else tuple(channels)
            ),
            using_xrf_maps=(
                self.config.using_xrf_maps
                if using_xrf_maps is None
                else using_xrf_maps
            ),
            plot_in_log_scale=(
                self.config.plot_image_in_log_scale
                if plot_in_log_scale is None
                else plot_in_log_scale
            ),
            show_colorbar=(
                self.config.show_colorbar_in_image
                if show_colorbar is None
                else show_colorbar
            ),
        )
        return json_safe(
            {
                "save_data_path": resolved_path,
                "current_mda_file": current_mda_file,
                **postprocessed,
            }
        )

    # Positioner axis -> (allowable range, unit) used for scan-range validation.
    _line_scan_positioners = ("x", "y", "z", "energy")

    def acquire_line_scan(
        self,
        positioner_name: str,
        length: float,
        stepsize: float,
        center: float = 0.0,
        sample_x: float | None = None,
        sample_y: float | None = None,
        sample_z: float | None = None,
        energy: float | None = None,
        dwell_ms: float | None = None,
        on_console: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        axis = positioner_name.strip().lower()
        # Absolute position the plan moves each axis to before scanning, with the
        # range/unit used to validate it. ``center`` is a RELATIVE offset applied
        # to the driven positioner from its position at scan time.
        absolute_targets = {
            "x": (sample_x, self.config.allowable_x_range, "um"),
            "y": (sample_y, self.config.allowable_y_range, "um"),
            "z": (sample_z, self.config.allowable_z_range, "um"),
            "energy": (energy, self.config.allowable_energy_range, "keV"),
        }
        if axis not in absolute_targets:
            raise ValueError(
                "positioner_name must be one of 'x', 'y', 'z', or 'energy'."
            )
        # Validate every explicitly requested absolute sample/energy position.
        # step1d_scanrecord moves to these (keeping the current position when None).
        for name, (value, value_range, value_unit) in absolute_targets.items():
            if value is not None:
                validate_position_in_range(value, value_range, name, unit=value_unit)
        # The scan sweeps ``center +/- length/2`` relative to the driven positioner.
        # The absolute extent can only be bounded when that axis's target position
        # is given; otherwise the current (unknown) position is used, so an absolute
        # range check is not possible here.
        scan_target, scan_range, scan_unit = absolute_targets[axis]
        if scan_target is not None:
            scan_center = scan_target + center
            validate_position_in_range(
                scan_center - length / 2, scan_range, axis, unit=scan_unit
            )
            validate_position_in_range(
                scan_center + length / 2, scan_range, axis, unit=scan_unit
            )
        dwell = self.config.dwell_line_scan * 1000 if dwell_ms is None else dwell_ms
        request = {
            "samplename": self.config.sample_name,
            "positioner_name": axis,
            "width": length,
            "center": center,
            "stepsize_x": stepsize,
            "dwell_ms": dwell,
            "xrf_on": self.config.xrf_on,
            "preamp1_on": self.config.preamp1_on,
        }
        # Only forward sample/energy positions that were explicitly requested.
        # step1d_scanrecord types these as ``float`` (not Optional), so sending an
        # explicit None fails QueueServer plan validation; omitting them lets the
        # plan fall back to the current position.
        for key, value in (
            ("sample_x", sample_x),
            ("sample_y", sample_y),
            ("sample_z", sample_z),
            ("energy", energy),
        ):
            if value is not None:
                request[key] = value
        self.line_scan_call_history.append(
            {
                "positioner_name": axis,
                "center": center,
                "length": length,
                "step": stepsize,
                "sample_x": sample_x,
                "sample_y": sample_y,
                "sample_z": sample_z,
                "energy": energy,
            }
        )
        # Captured before the scan: next_file_name auto-increments once the plan
        # runs, so reading it here yields the file this scan actually writes.
        current_mda_file = self.qserver.get_current_mda_file(timeout=10.0)
        execution = self.qserver.run_acquire_line_scan(request, on_console=on_console)
        task_result = execution["task_result"]
        save_data_path = self.qserver.get_save_data_path(timeout=10.0)
        postprocessed = self.postprocessor.process_line_scan(
            save_data_path=save_data_path,
            current_mda_file=current_mda_file,
            channels=self.config.xrf_elms,
            roi_num=self.config.xrf_roi_num,
            using_xrf_maps=self.config.using_xrf_maps,
            scan_samy=self.config.scan_samy,
        )
        return json_safe(
            {
                "plan_name": execution["plan_name"],
                "item_uid": execution["item_uid"],
                "run_uids": result_run_uids(task_result),
                "scan_ids": result_scan_ids(task_result),
                "save_data_path": save_data_path,
                "current_mda_file": current_mda_file,
                **postprocessed,
                "raw_task_result": task_result,
            }
        )

    def move_sample(self, axis: str, position: float) -> dict[str, Any]:
        axis_name = axis.strip().lower()
        allowable_ranges = {
            "x": self.config.allowable_x_range,
            "y": self.config.allowable_y_range,
            "z": self.config.allowable_z_range,
        }
        if axis_name not in allowable_ranges:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")
        position_value = float(position)
        validate_position_in_range(position_value, allowable_ranges[axis_name], axis_name)
        execution = self.qserver.move_sample(axis_name, position_value, timeout=30.0)
        task_result = execution["task_result"]
        return json_safe(
            {
                "axis": axis_name,
                "position": position_value,
                "readback": result_return_value(task_result),
                "plan_name": execution["plan_name"],
                "item_uid": execution["item_uid"],
                "exit_status": task_result.get("exit_status"),
                "raw_task_result": task_result,
            }
        )

    def move_zp_z(self, position: float) -> dict[str, Any]:
        # zp-z (zone-plate z) has its own travel limits, distinct from the
        # sample z motor's allowable_z_range.
        value = float(position)
        validate_position_in_range(value, self.config.allowable_zp_range, "zp-z")
        execution = self.qserver.move_zp_z(value, timeout=30.0)
        task_result = execution["task_result"]
        return json_safe(
            {
                "position": value,
                "readback": result_return_value(task_result),
                "plan_name": execution["plan_name"],
                "item_uid": execution["item_uid"],
                "exit_status": task_result.get("exit_status"),
                "raw_task_result": task_result,
            }
        )

    def set_parameters(self, parameters: list[float]) -> Any:
        if not parameters:
            raise ValueError("The 'parameters' list must contain at least one value.")
        result = self.move_zp_z(parameters[0])
        return result_return_value(result["raw_task_result"])
