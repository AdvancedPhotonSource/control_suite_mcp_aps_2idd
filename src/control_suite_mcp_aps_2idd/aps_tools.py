"""Local APS 2-ID-D acquisition and parameter tools."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable
import base64
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from control_suite_mcp_aps_2idd.util import (
    get_timestamp,
    process_xrfdata,
    save_xrf_line_scan,
    save_xrfdata,
    validate_position_in_range,
    wait_for_file,
)

logger = logging.getLogger(__name__)


class AcquisitionBuffers:
    """Image and call-history buffers compatible with the EAA MCP proxy contract."""

    def __init__(self) -> None:
        """Initialize empty image buffers and acquisition histories."""
        self.image_0: np.ndarray | None = None
        self.image_km1: np.ndarray | None = None
        self.image_k: np.ndarray | None = None
        self.psize_0: float | None = None
        self.psize_km1: float | None = None
        self.psize_k: float | None = None
        self.image_array_artifact_dir = Path(".tmp") / "acquisition_arrays"
        self.image_acquisition_call_history: list[dict[str, Any]] = []
        self.line_scan_call_history: list[dict[str, Any]] = []

    @property
    def counter_acquire_image(self) -> int:
        """Return the number of image acquisitions recorded by this worker."""
        return len(self.image_acquisition_call_history)

    def update_image_acquisition_call_history(
        self,
        x_center: float,
        y_center: float,
        size_x: float,
        size_y: float,
        psize_x: float,
        psize_y: float,
    ) -> None:
        """Record an image acquisition request."""
        self.image_acquisition_call_history.append(
            {
                "x_center": x_center,
                "y_center": y_center,
                "size_x": size_x,
                "size_y": size_y,
                "psize_x": psize_x,
                "psize_y": psize_y,
            }
        )

    def update_line_scan_call_history(
        self,
        step: float,
        x_center: float,
        y_center: float,
        length: float,
        angle: float,
    ) -> None:
        """Record a line-scan request."""
        self.line_scan_call_history.append(
            {
                "step": step,
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "angle": angle,
            }
        )

    def get_image_buffer_info_by_name(self, buffer_name: str) -> dict[str, Any]:
        """Return pixel size and shape metadata for an image buffer."""
        image = getattr(self, buffer_name)
        psize = getattr(self, f"psize_{buffer_name.split('_', 1)[1]}")
        return {
            "buffer_name": buffer_name,
            "psize": psize,
            "shape": None if image is None else list(image.shape),
            "dtype": None if image is None else str(image.dtype),
        }

    def get_current_image_info(self) -> dict[str, Any]:
        """Return metadata for the current image buffer."""
        return self.get_image_buffer_info_by_name("image_k")

    def get_previous_image_info(self) -> dict[str, Any]:
        """Return metadata for the previous image buffer."""
        return self.get_image_buffer_info_by_name("image_km1")

    def get_initial_image_info(self) -> dict[str, Any]:
        """Return metadata for the initial image buffer."""
        return self.get_image_buffer_info_by_name("image_0")

    def resolve_image_buffer_name(self, buffer_name: str) -> str:
        """Resolve public EAA buffer aliases to internal image buffer names."""
        aliases = {
            "current": "image_k",
            "previous": "image_km1",
            "initial": "image_0",
            "image_k": "image_k",
            "image_km1": "image_km1",
            "image_0": "image_0",
        }
        try:
            return aliases[buffer_name]
        except KeyError as exc:
            raise ValueError(
                "buffer_name must be one of current, previous, initial, "
                "image_k, image_km1, or image_0."
            ) from exc

    def get_image_array_payload(self, buffer_name: str) -> dict[str, Any]:
        """Return a base64-encoded NumPy payload for a buffered image array."""
        resolved_name = self.resolve_image_buffer_name(buffer_name)
        image = getattr(self, resolved_name)
        if image is None:
            raise ValueError(f"Image buffer is empty: {buffer_name}")
        contiguous = np.ascontiguousarray(image)
        return {
            "encoding": "numpy_base64",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }

    def dump_array(self, buffer_name: str) -> dict[str, str]:
        """Save a buffered image as a NumPy array artifact."""
        resolved_name = self.resolve_image_buffer_name(buffer_name)
        image = getattr(self, resolved_name)
        if image is None:
            raise ValueError(f"Image buffer is empty: {buffer_name}")
        artifact_dir = self.image_array_artifact_dir.expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        array_path = artifact_dir / f"{resolved_name}_{get_timestamp()}.npy"
        np.save(array_path, image)
        return {"array_path": str(array_path)}

    def update_image_buffers(self, new_image: np.ndarray, psize: float = 1) -> None:
        """Update initial, previous, and current image buffers."""
        if self.image_0 is None:
            self.image_0 = new_image
            self.psize_0 = psize
        self.image_km1 = self.image_k
        self.psize_km1 = self.psize_k
        self.image_k = new_image
        self.psize_k = psize


class APSTwoIDDAcquireImage(AcquisitionBuffers):
    """APS 2-ID-D MIC image and line-scan acquisition routines."""

    def __init__(
        self,
        sample_name: str = "smp1",
        dwell_imaging: float = 0.05,
        dwell_line_scan: float = 0.2,
        xrf_on: bool = True,
        preamp1_on: bool = False,
        using_xrf_maps: bool = False,
        xrf_elms: tuple[str, ...] = ("Cr",),
        xrf_roi_num: int = 16,
        allowable_x_range: tuple[float, float] | None = None,
        allowable_y_range: tuple[float, float] | None = None,
        allowable_z_range: tuple[float, float] | None = None,
        plot_image_in_log_scale: bool = False,
        show_colorbar_in_image: bool = False,
        line_scan_return_gaussian_fit: bool = False,
        scan_samy: bool = False,
    ) -> None:
        """Initialize local acquisition settings and empty beamline handles."""
        super().__init__()
        self.sample_name = sample_name
        self.dwell_imaging = dwell_imaging
        self.dwell_line_scan = dwell_line_scan
        self.xrf_on = xrf_on
        self.preamp1_on = preamp1_on
        self.using_xrf_maps = using_xrf_maps
        self.xrf_elms = xrf_elms
        self.xrf_roi_num = xrf_roi_num
        self.allowable_x_range = allowable_x_range
        self.allowable_y_range = allowable_y_range
        self.allowable_z_range = allowable_z_range
        self.plot_image_in_log_scale = plot_image_in_log_scale
        self.show_colorbar_in_image = show_colorbar_in_image
        self.line_scan_return_gaussian_fit = line_scan_return_gaussian_fit
        self.scan_samy = scan_samy
        self.RE: Callable | None = None
        self.savedata: Any = None
        self.scan2d_plan: Callable | None = None
        self.scan1d_plan: Callable | None = None
        self.samy_motor: Any = None
        self.bps: Any = None

    def acquire_image(
        self,
        width: float = 0,
        height: float = 0,
        x_center: float | None = None,
        y_center: float | None = None,
        stepsize_x: float = 0,
        stepsize_y: float = 0,
    ) -> dict[str, Any]:
        """Acquire an image of a scan area with the scanning x-ray microscope."""
        self.update_image_acquisition_call_history(
            x_center,
            y_center,
            width,
            height,
            stepsize_x,
            stepsize_y,
        )
        validate_position_in_range(x_center, self.allowable_x_range, "x")
        validate_position_in_range(y_center, self.allowable_y_range, "y")
        logger.info(
            "Acquiring image of size %s um x %s um at %s um, %s um.",
            width,
            height,
            x_center,
            y_center,
        )
        if self.RE is None or self.scan2d_plan is None:
            raise ValueError("RunEngine or 2D scan plan is not initialized.")
        self.RE(
            self.scan2d_plan(
                samplename=self.sample_name,
                width=width,
                x_center=x_center,
                stepsize_x=stepsize_x,
                height=height,
                y_center=y_center,
                stepsize_y=stepsize_y,
                dwell_ms=self.dwell_imaging * 1000,
                xrf_on=self.xrf_on,
                preamp1_on=self.preamp1_on,
            )
        )

        mda_path = self.savedata.full_path_name.get()
        mda_dir = mda_path.replace("data1", "mnt/micdata1")
        parent_dir = os.path.dirname(os.path.dirname(mda_dir))
        png_output_dir = os.path.join(parent_dir, "png_output")
        current_mda_file = self.savedata.next_file_name

        logger.info("About to process data: %s", current_mda_file)
        if self.using_xrf_maps:
            process_code = process_xrfdata(parent_dir, current_mda_file)
        else:
            img_h5_path = os.path.join(parent_dir, "img.dat", f"{current_mda_file}.h50")
            logger.info("Expected .h5 file path is %s", img_h5_path)
            process_code = wait_for_file(img_h5_path, duration=30)

        if not process_code:
            logger.error("Failed to process %s", current_mda_file)
            return {"result": f"Failed to process {current_mda_file}"}

        os.makedirs(png_output_dir, exist_ok=True)
        img_h5_path = os.path.join(parent_dir, "img.dat", f"{current_mda_file}.h50")
        img_path, img_arr = save_xrfdata(
            img_h5_path,
            png_output_dir,
            elms=self.xrf_elms,
            return_image_array=True,
            plot_in_log_scale=self.plot_image_in_log_scale,
            show_colorbar_in_image=self.show_colorbar_in_image,
        )
        if img_path is not None:
            wait_for_file(img_path, duration=5)
        if img_path and img_arr is not None:
            self.update_image_buffers(img_arr, psize=stepsize_x)
            return {"img_path": img_path, "psize": stepsize_x}
        logger.error("Failed to save images for %s", current_mda_file)
        return {"result": f"Failed to save images for {current_mda_file}"}

    def acquire_line_scan(
        self,
        length: float = 0,
        x_center: float | None = None,
        y_center: float | None = None,
        stepsize_x: float = 0,
    ) -> dict[str, Any]:
        """Acquire a horizontal line scan at a center position."""
        self.set_motor_y(y_center)
        start_x = x_center - length / 2
        end_x = x_center + length / 2
        self.update_line_scan_call_history(
            step=stepsize_x,
            x_center=x_center,
            y_center=y_center,
            length=length,
            angle=0.0,
        )
        validate_position_in_range(start_x, self.allowable_x_range, "x")
        validate_position_in_range(end_x, self.allowable_x_range, "x")
        logger.info("Acquiring line scan of width %s um at x=%s um.", length, x_center)
        if self.RE is None or self.scan1d_plan is None:
            raise ValueError("RunEngine or 1D scan plan is not initialized.")
        self.RE(
            self.scan1d_plan(
                samplename=self.sample_name,
                width=length,
                x_center=x_center,
                stepsize_x=stepsize_x,
                dwell_ms=self.dwell_line_scan * 1000,
                xrf_on=self.xrf_on,
                preamp1_on=self.preamp1_on,
            )
        )

        mda_path = self.savedata.full_path_name.get()
        mda_dir = mda_path.replace("data1", "mnt/micdata1")
        parent_dir = os.path.dirname(os.path.dirname(mda_dir))
        png_output_dir = os.path.join(parent_dir, "png_output")
        current_mda_file = self.savedata.next_file_name
        mda_file_path = os.path.join(mda_dir, current_mda_file)

        process_code = wait_for_file(mda_file_path, duration=20)
        if not process_code:
            logger.error("Failed to process %s", current_mda_file)
            return {"result": f"Failed to process {current_mda_file}"}

        os.makedirs(png_output_dir, exist_ok=True)
        line_result = save_xrf_line_scan(
            mda_file_path,
            png_output_dir,
            roi_num=self.xrf_roi_num,
            return_line_array=True,
            scan_samy=self.scan_samy,
        )
        if line_result is None:
            logger.error("Failed to save images for %s", current_mda_file)
            return {"result": f"Failed to save images for {current_mda_file}"}
        img_path, fit_payload = line_result
        wait_for_file(img_path, duration=5)
        if np.isnan(fit_payload["fwhm"]):
            logger.warning("Gaussian fit returned NaN for line-scan FWHM.")
        self.append_line_scan_overlays(img_path)
        if self.line_scan_return_gaussian_fit:
            return {
                "img_path": img_path,
                "fwhm": fit_payload["fwhm"],
                "a": fit_payload.get("a"),
                "mu": fit_payload.get("mu"),
                "sigma": fit_payload.get("sigma"),
                "c": fit_payload.get("c"),
                "normalized_residual": fit_payload.get("normalized_residual"),
                "x_min": fit_payload.get("x_min"),
                "x_max": fit_payload.get("x_max"),
            }
        return {"img_path": img_path}

    def append_line_scan_overlays(self, img_path: str) -> None:
        """Append current and reference line-scan overlays to a line-scan image."""
        line_scan_image = Image.open(img_path).convert("RGB")
        overlays: list[Image.Image] = []
        if self.image_k is not None and len(self.image_acquisition_call_history) > 0:
            overlays.append(
                self.render_scan_overlay(
                    image_array=self.image_k,
                    image_info=self.image_acquisition_call_history[-1],
                    line_info=self.line_scan_call_history[-1],
                    line_color="red",
                    title="Line scan position",
                )
            )
        if (
            self.image_0 is not None
            and len(self.image_acquisition_call_history) > 1
            and len(self.line_scan_call_history) > 1
        ):
            first_line_info = self.line_scan_call_history[0]
            if first_line_info["y_center"] is not None:
                overlays.append(
                    self.render_scan_overlay(
                        image_array=self.image_0,
                        image_info=self.image_acquisition_call_history[0],
                        line_info=first_line_info,
                        line_color="blue",
                        title="Reference line scan position",
                    )
                )
        if not overlays:
            return

        resized_overlays = []
        for overlay_image in overlays:
            if overlay_image.height != line_scan_image.height:
                new_width = int(overlay_image.width * line_scan_image.height / overlay_image.height)
                overlay_image = overlay_image.resize((new_width, line_scan_image.height))
            resized_overlays.append(overlay_image)
        total_overlay_width = sum(overlay_image.width for overlay_image in resized_overlays)
        stitched_image = Image.new(
            "RGB",
            (line_scan_image.width + total_overlay_width, line_scan_image.height),
            "white",
        )
        stitched_image.paste(line_scan_image, (0, 0))
        x_offset = line_scan_image.width
        for overlay_image in resized_overlays:
            stitched_image.paste(overlay_image, (x_offset, 0))
            x_offset += overlay_image.width
        stitched_image.save(img_path)

    def render_scan_overlay(
        self,
        image_array: np.ndarray,
        image_info: dict[str, Any],
        line_info: dict[str, Any],
        line_color: str,
        title: str,
    ) -> Image.Image:
        """Render a line position overlay on top of an acquired image."""
        image_to_plot = image_array
        if self.plot_image_in_log_scale:
            image_to_plot = np.log10(image_to_plot + 1)
        x_min = image_info["x_center"] - image_info["size_x"] / 2
        x_max = image_info["x_center"] + image_info["size_x"] / 2
        y_min = image_info["y_center"] - image_info["size_y"] / 2
        y_max = image_info["y_center"] + image_info["size_y"] / 2
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(
            image_to_plot,
            cmap="inferno",
            origin="upper",
            extent=[x_min, x_max, y_max, y_min],
        )
        half = line_info["length"] / 2
        angle_rad = np.radians(line_info["angle"])
        ax.plot(
            [
                line_info["x_center"] - half * np.cos(angle_rad),
                line_info["x_center"] + half * np.cos(angle_rad),
            ],
            [
                line_info["y_center"] - half * np.sin(angle_rad),
                line_info["y_center"] + half * np.sin(angle_rad),
            ],
            color=line_color,
            linewidth=2,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def set_motor_y(self, value: float) -> str:
        """Set the sample y motor position."""
        if self.RE is None:
            raise ValueError("RunEngine is not set")
        if self.samy_motor is None:
            raise ValueError("samy_motor is not set")
        if self.bps is None:
            raise ValueError("Bluesky plan stubs are not set")
        validate_position_in_range(value, self.allowable_y_range, "y")
        self.RE(self.bps.mv(self.samy_motor, value))
        message = f"Move sample y motor to position: {value}"
        logger.info(message)
        return message


class APSSetParameters:
    """Local parameter tuning routine for APS 2-ID-D zp-z."""

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        parameter_ranges: list[tuple[float, ...]] | None = None,
    ) -> None:
        """Initialize local parameter metadata and empty beamline handles."""
        self.parameter_names = ["zp-z"] if parameter_names is None else parameter_names
        self.parameter_ranges = [(-200.0,), (-180.0,)] if parameter_ranges is None else parameter_ranges
        self.allowable_z_range: tuple[float, float] | None = None
        self.RE: Callable | None = None
        self.zp_z_motor: Any = None
        self.bps: Any = None
        self.parameter_history: list[list[float]] = []

    def update_parameter_history(self, parameters: list[float]) -> None:
        """Record a parameter update."""
        self.parameter_history.append(parameters)

    def set_parameters(self, parameters: list[float]) -> str:
        """Set the zone-plate z motor position."""
        if self.RE is None:
            raise ValueError("RunEngine is not set")
        if self.zp_z_motor is None:
            raise ValueError("zp_z_motor is not set")
        if self.bps is None:
            raise ValueError("Bluesky plan stubs are not set")
        if self.parameter_ranges is None:
            raise ValueError("parameter_ranges is not set")
        validate_position_in_range(
            parameters[0],
            (self.parameter_ranges[0][0], self.parameter_ranges[1][0]),
            "z",
        )
        self.RE(self.bps.mv(self.zp_z_motor, parameters[0]))
        message = f"Moved Zone Plate z position to position: {parameters[0]}"
        logger.info(message)
        self.update_parameter_history(parameters)
        return message
