"""Post-processing helpers for APS 2-ID-D MIC acquisition artifacts."""

from __future__ import annotations

import logging
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.optimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquisitionArtifacts:
    parent_dir: Path
    mda_dir: Path
    mda_path: Path
    h5_path: Path
    png_output_dir: Path
    raw_output_dir: Path


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def gaussian_1d(x: np.ndarray, a: float, mu: float, sigma: float, c: float = 0) -> np.ndarray:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c


def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    y_threshold: float = 0,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit a 1D Gaussian, matching the EAA fitting behavior without importing EAA."""
    x_data = np.array(x, dtype=float)
    y_data = np.array(y, dtype=float)
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite_mask]
    y_data = y_data[finite_mask]
    if x_data.size < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]
    x_min = float(np.min(x_data))
    x_max_input = float(np.max(x_data))
    x_span = x_max_input - x_min
    if not np.isfinite(x_span) or x_span <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_max, y_min = np.max(y_data), np.min(y_data)
    y_range = float(y_max - y_min)
    if not np.isfinite(y_range) or np.isclose(y_range, 0.0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    smooth_sigma = max(1.0, x_data.size * 0.02)
    y_smooth = scipy.ndimage.gaussian_filter1d(y_data, sigma=smooth_sigma, mode="nearest")
    y_smooth_max = float(np.max(y_smooth))
    y_smooth_min = float(np.min(y_smooth))
    y_smooth_range = y_smooth_max - y_smooth_min
    if not np.isfinite(y_smooth_range) or np.isclose(y_smooth_range, 0.0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    x_peak = float(x_data[np.argmax(y_smooth)])
    offset = x_peak
    x = x_data - offset
    fit_threshold = y_smooth_min + y_threshold * y_smooth_range
    mask = y_smooth >= fit_threshold
    if int(np.count_nonzero(mask)) < 5:
        mask = np.ones_like(y_data, dtype=bool)

    positive_weight = np.clip(y_smooth - y_smooth_min, a_min=0.0, a_max=None)
    if np.sum(positive_weight) > 0:
        mu_guess = float(np.sum(x * positive_weight) / np.sum(positive_weight))
    else:
        mu_guess = 0.0

    width_mask = y_smooth >= (y_smooth_min + 0.5 * y_smooth_range)
    x_above_half = x[width_mask]
    if x_above_half.size >= 2:
        sigma_guess = float((x_above_half.max() - x_above_half.min()) / 2.355)
    else:
        sigma_guess = x_span / 6.0
    sigma_guess = float(np.clip(sigma_guess, x_span / 100.0, x_span))

    a_guess = max(y_smooth_range, np.finfo(float).eps)
    c_guess = float(np.median(y_data[~mask])) if np.any(~mask) else float(y_smooth_min)
    p0 = [a_guess, mu_guess, sigma_guess, c_guess]
    lower_bounds = [0.0, float(np.min(x)), max(x_span / 1000.0, 1e-12), float(y_min - y_range)]
    upper_bounds = [
        float(2 * y_range + abs(y_max)),
        float(np.max(x)),
        float(2 * x_span),
        float(y_max + y_range),
    ]

    def run_fit(current_mask: np.ndarray, current_p0: list[float]) -> np.ndarray | None:
        if int(np.count_nonzero(current_mask)) < 5:
            return None
        try:
            popt, _ = scipy.optimize.curve_fit(
                gaussian_1d,
                x[current_mask],
                y_data[current_mask],
                p0=current_p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
            return popt
        except (RuntimeError, ValueError):
            return None

    popt = run_fit(mask, p0)
    if popt is None:
        y_smooth_retry = scipy.ndimage.gaussian_filter1d(
            y_data,
            sigma=max(2.0, x_data.size * 0.05),
            mode="nearest",
        )
        retry_weight = np.clip(y_smooth_retry - np.min(y_smooth_retry), a_min=0.0, a_max=None)
        if np.sum(retry_weight) > 0:
            mu_retry = float(np.sum(x * retry_weight) / np.sum(retry_weight))
        else:
            mu_retry = 0.0
        retry_mask = y_smooth_retry >= (
            np.min(y_smooth_retry)
            + max(0.1, y_threshold) * (np.max(y_smooth_retry) - np.min(y_smooth_retry))
        )
        retry_p0 = [a_guess, mu_retry, sigma_guess, c_guess]
        popt = run_fit(retry_mask, retry_p0)

    if popt is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_fit = gaussian_1d(x, *popt)
    amplitude = float(popt[0])
    sigma = float(popt[2])
    mu = float(popt[1])
    if sigma <= 0 or not (float(np.min(x)) <= mu <= float(np.max(x))):
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input
    if np.isclose(amplitude, 0.0):
        normalized_residual = np.nan
    else:
        normalized_residual = float(np.mean(((y_data - y_fit) / amplitude) ** 2))
    popt[1] += offset
    return (
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        float(popt[3]),
        normalized_residual,
        x_min,
        x_max_input,
    )


def wait_for_stable_file(
    file_path: str | Path,
    *,
    stable_s: float = 30.0,
    timeout_s: float | None = 300.0,
    poll_s: float = 1.0,
) -> bool:
    """Wait until a file exists and its modification time is stable."""
    path = Path(file_path)
    start = time.monotonic()
    stable_since: float | None = None
    last_mtime: float | None = None
    while True:
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            return False
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stable_since = time.monotonic()
            elif stable_since is not None and time.monotonic() - stable_since >= stable_s:
                return True
        time.sleep(poll_s)


def run_xrfmaps_exe(
    exe_path: str, args: list[str] | None = None
) -> subprocess.CompletedProcess[str] | None:
    command = [exe_path]
    if args:
        command.extend(args)
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Error executing %s: %s", exe_path, exc)
        return None


def process_xrfdata(
    parent_dir: str | Path,
    scan_num_mda: str,
    det_range: str = "0:0",
    quantify_with: str = "maps_standardinfo.txt",
    fitting_type: str = "roi, nnls",
    num_iter: int = 20000,
    exe_path: str = "/mnt/micdata1/XRF-Maps/bin/xrf_maps.exe",
) -> bool:
    parent = Path(parent_dir)
    mda_dir = parent / "mda"
    mda_path = mda_dir / scan_num_mda
    if not mda_path.exists():
        logger.error("MDA file %s not found", mda_path)
        return False
    args = [
        "--dir",
        str(parent),
        "--files",
        scan_num_mda,
        "--detector-range",
        det_range,
        "--export-csv",
        "",
        "--quantify-with",
        quantify_with,
        "--fit",
        fitting_type,
        "--optimizer-num-iter",
        str(num_iter),
        "--generate-avg-h5",
        "",
    ]
    result = run_xrfmaps_exe(exe_path, args)
    return result is not None and result.returncode == 0


def artifact_paths(save_data_path: str | Path, current_mda_file: str) -> AcquisitionArtifacts:
    # base = Path(str(save_data_path).replace("mda", "img.dat")).expanduser().resolve()
    # if base.suffix == ".mda":
    #     mda_path = base
    #     mda_dir = base.parent
    #     parent_dir = mda_dir.parent if mda_dir.name == "mda" else mda_dir
    # elif base.name == "mda":
    #     mda_dir = base
    #     parent_dir = base.parent
    #     mda_path = mda_dir / current_mda_file
    # else:
    #     parent_dir = base
    #     mda_dir = parent_dir / "mda"
    #     mda_path = mda_dir / current_mda_file

    tool_output_dir = Path(save_data_path) / "tool_output"
    if not tool_output_dir.exists():
        tool_output_dir.mkdir(parents=True, exist_ok=True)

    return AcquisitionArtifacts(
        parent_dir=save_data_path,
        mda_dir=Path(save_data_path) / "mda",
        mda_path=Path(save_data_path) / "mda" / current_mda_file,
        h5_path=Path(save_data_path) / "img.dat" / f"{current_mda_file}.h50",
        png_output_dir=Path(tool_output_dir) / "png_output",
        raw_output_dir=Path(tool_output_dir) / "npy_output",
    )


def _decode_names(values: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode(errors="replace"))
        else:
            names.append(str(value))
    return names


def load_h5(
    img_h5_path: str | Path, fit_types: tuple[str, ...] = ("NNLS", "ROI")
) -> dict[str, Any]:
    path = Path(img_h5_path)
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"The XRF h5 file {path} was not found or is empty.")
    with h5py.File(path, "r") as h5:
        data: dict[str, Any] = {
            "scan": path.name,
            "x_axis": h5["MAPS/Scan/x_axis"][:],
            "y_axis": h5["MAPS/Scan/y_axis"][:],
        }
        for fit_type in fit_types:
            group = h5[f"MAPS/XRF_Analyzed/{fit_type}"]
            data[f"{fit_type}_arr"] = group["Counts_Per_Sec"][:]
            data[f"{fit_type}_ch"] = _decode_names(group["Channel_Names"][:])
        return data


def select_channel(
    data: dict[str, Any],
    *,
    channels: tuple[str, ...] | list[str] | None = None,
    roi_num: int | None = None,
    fit_type: str = "ROI",
) -> tuple[str, np.ndarray]:
    data_arr = np.asarray(data[f"{fit_type}_arr"])
    data_ch = list(data[f"{fit_type}_ch"])
    if channels:
        for channel in channels:
            if channel in data_ch:
                return channel, np.asarray(data_arr[data_ch.index(channel)])
        raise ValueError(f"None of the requested channels {channels!r} are available: {data_ch!r}")
    if roi_num is not None and 0 <= roi_num < len(data_ch):
        return data_ch[roi_num], np.asarray(data_arr[roi_num])
    if not data_ch:
        raise ValueError(f"No channels found for fit type {fit_type}.")
    return data_ch[0], np.asarray(data_arr[0])


def render_xrf_image(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    scan_name: str,
    channel: str,
    output_path: str | Path,
    cmap: str = "inferno",
    vmax_percentile: float = 99.0,
    vmin: float = 0.0,
    plot_in_log_scale: bool = False,
    show_colorbar: bool = False,
) -> str:
    plot_array = np.asarray(image, dtype=float)
    vmax = float(np.nanpercentile(plot_array, vmax_percentile))
    if plot_in_log_scale:
        plot_array = np.log10(plot_array + 1)
        vmax = float(np.log10(vmax + 1))
        vmin = float(np.log10(vmin + 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(plot_array, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(f"{scan_name} {channel}")
    if show_colorbar:
        cbar = fig.colorbar(im)
        cbar.set_label("Intensity")
    xticks = np.linspace(0, len(x_axis) - 1, min(5, len(x_axis)), dtype=int)
    yticks = np.linspace(0, len(y_axis) - 1, min(5, len(y_axis)), dtype=int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([np.round(x_axis[i], 2) for i in xticks])
    ax.set_yticklabels([np.round(y_axis[i], 2) for i in yticks])
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path.resolve())


def line_profile(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    scan_samy: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.squeeze(np.asarray(image, dtype=float))
    if values.ndim == 0:
        raise ValueError("Line-scan channel data is scalar; expected a 1D profile.")
    if values.ndim > 1:
        keep_axis = int(np.argmax(values.shape))
        reduce_axes = tuple(axis for axis in range(values.ndim) if axis != keep_axis)
        values = np.nanmean(values, axis=reduce_axes)

    axis_candidates = (
        [
            (np.asarray(y_axis, dtype=float), "Y-axis Position"),
            (np.asarray(x_axis, dtype=float), "X-axis Position"),
        ]
        if scan_samy
        else [
            (np.asarray(x_axis, dtype=float), "X-axis Position"),
            (np.asarray(y_axis, dtype=float), "Y-axis Position"),
        ]
    )
    for axis_values, label in axis_candidates:
        axis_values = np.squeeze(axis_values)
        if axis_values.ndim == 1 and axis_values.size == values.size:
            order = np.argsort(axis_values)
            return axis_values[order], values[order], label
    x = np.arange(values.size, dtype=float)
    return x, values, "Point"


def render_line_scan(
    x: np.ndarray,
    y: np.ndarray,
    val_gauss: np.ndarray | None,
    fwhm: float | None,
    *,
    scan_name: str,
    channel: str,
    axis_label: str,
    output_path: str | Path,
) -> str:
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.plot(x, y, label="data")
    if val_gauss is not None:
        ax.plot(x, val_gauss, linestyle="--", color="red", label="fit")
    fwhm_text = "NaN" if fwhm is None else f"{fwhm:.2f}"
    ax.text(
        0.05,
        0.95,
        f"FWHM = {fwhm_text}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax.legend()
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Intensity")
    ax.set_title(f"{scan_name}-{channel}")
    ax.grid(True)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path.resolve())


class APSMICPostProcessor:
    """Process MDA/HDF5 artifacts into EAA-readable PNG and NPY files."""

    def __init__(
        self,
        *,
        h5_stable_s: float = 30.0,
        output_stable_s: float = 5.0,
        wait_timeout_s: float = 300.0,
        poll_s: float = 1.0,
    ) -> None:
        self.h5_stable_s = h5_stable_s
        self.output_stable_s = output_stable_s
        self.wait_timeout_s = wait_timeout_s
        self.poll_s = poll_s
        self.last_image: np.ndarray | None = None

    def prepare_artifacts(
        self,
        *,
        save_data_path: str,
        current_mda_file: str,
        using_xrf_maps: bool,
    ) -> AcquisitionArtifacts:
        artifacts = artifact_paths(save_data_path, current_mda_file)
        logger.info(
            "Preparing acquisition artifacts for %s: mda=%s h5=%s png_dir=%s raw_dir=%s",
            current_mda_file,
            artifacts.mda_path,
            artifacts.h5_path,
            artifacts.png_output_dir,
            artifacts.raw_output_dir,
        )
        if using_xrf_maps:
            logger.info("Processing %s with XRF-Maps in %s", current_mda_file, artifacts.parent_dir)
            if not process_xrfdata(artifacts.parent_dir, current_mda_file):
                raise RuntimeError(f"Failed to process {current_mda_file} with XRF-Maps.")
        else:
            logger.info(
                "Waiting for HDF5 file %s to exist and remain stable for %.1f seconds",
                artifacts.h5_path,
                self.h5_stable_s,
            )
            if not wait_for_stable_file(
                artifacts.h5_path,
                stable_s=self.h5_stable_s,
                timeout_s=self.wait_timeout_s,
                poll_s=self.poll_s,
            ):
                raise TimeoutError(f"Timed out waiting for HDF5 file {artifacts.h5_path}.")
        logger.info(
            "Verifying HDF5 file %s remains stable for %.1f seconds",
            artifacts.h5_path,
            self.h5_stable_s,
        )
        if not wait_for_stable_file(
            artifacts.h5_path,
            stable_s=self.h5_stable_s,
            timeout_s=self.wait_timeout_s,
            poll_s=self.poll_s,
        ):
            raise TimeoutError(f"Timed out waiting for HDF5 file {artifacts.h5_path}.")
        artifacts.png_output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.raw_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Artifact directories are ready for %s", current_mda_file)
        return artifacts

    def process_image(
        self,
        *,
        save_data_path: str,
        current_mda_file: str,
        channels: tuple[str, ...],
        using_xrf_maps: bool,
        plot_in_log_scale: bool,
        show_colorbar: bool,
    ) -> dict[str, Any]:
        artifacts = self.prepare_artifacts(
            save_data_path=save_data_path,
            current_mda_file=current_mda_file,
            using_xrf_maps=using_xrf_maps,
        )
        logger.info("Parsing image HDF5 data from %s", artifacts.h5_path)
        data = load_h5(artifacts.h5_path)
        channel, image = select_channel(data, channels=channels, fit_type="ROI")
        logger.info(
            "Selected image channel %s from %s with array shape %s",
            channel,
            artifacts.h5_path,
            image.shape,
        )
        self.last_image = np.asarray(image).copy()
        raw_path = (artifacts.raw_output_dir / f"{current_mda_file}_{channel}.npy").resolve()
        np.save(raw_path, np.asarray(image))
        logger.info("Saved raw image array to %s", raw_path)
        img_path = render_xrf_image(
            image,
            np.asarray(data["x_axis"]),
            np.asarray(data["y_axis"]),
            scan_name=str(data["scan"]),
            channel=channel,
            output_path=artifacts.png_output_dir / f"{data['scan']}_{channel}.png",
            plot_in_log_scale=plot_in_log_scale,
            show_colorbar=show_colorbar,
        )
        logger.info("Saved rendered image PNG to %s", img_path)
        wait_for_stable_file(
            img_path,
            stable_s=self.output_stable_s,
            timeout_s=self.wait_timeout_s,
            poll_s=self.poll_s,
        )
        return {
            "img_path": img_path,
            "raw_data_path": str(raw_path),
            "channel": channel,
            "h5_path": str(artifacts.h5_path),
            "mda_path": str(artifacts.mda_path),
        }

    def process_line_scan(
        self,
        *,
        save_data_path: str,
        current_mda_file: str,
        channels: tuple[str, ...],
        roi_num: int,
        using_xrf_maps: bool,
        scan_samy: bool,
    ) -> dict[str, Any]:
        artifacts = self.prepare_artifacts(
            save_data_path=save_data_path,
            current_mda_file=current_mda_file,
            using_xrf_maps=using_xrf_maps,
        )
        logger.info("Parsing line-scan HDF5 data from %s", artifacts.h5_path)
        data = load_h5(artifacts.h5_path)
        channel, image = select_channel(data, channels=channels, roi_num=roi_num, fit_type="ROI")
        logger.info(
            "Selected line-scan channel %s from %s with array shape %s",
            channel,
            artifacts.h5_path,
            image.shape,
        )
        self.last_image = np.asarray(image).copy()
        x, y, axis_label = line_profile(
            image,
            np.asarray(data["x_axis"]),
            np.asarray(data["y_axis"]),
            scan_samy=scan_samy,
        )
        logger.info("Fitting Gaussian to line profile with %d points", x.size)
        a, mu, sigma, c, normalized_residual, x_min, x_max = fit_gaussian_1d(x, y)
        if np.any(np.isnan([a, mu, sigma, c])):
            val_gauss = None
            fwhm = None
            logger.info("Gaussian fit did not produce finite primary parameters")
        else:
            val_gauss = gaussian_1d(x, a, mu, sigma, c)
            fwhm = float(2.35 * abs(sigma))
            logger.info(
                "Gaussian fit completed: fwhm=%s a=%s mu=%s sigma=%s c=%s",
                fwhm,
                a,
                mu,
                sigma,
                c,
            )

        raw_path = (artifacts.raw_output_dir / f"{current_mda_file}_{channel}_line.npy").resolve()
        np.save(raw_path, np.column_stack((x, y)))
        logger.info("Saved raw line-scan array to %s", raw_path)
        img_path = render_line_scan(
            x,
            y,
            val_gauss,
            fwhm,
            scan_name=str(data["scan"]),
            channel=channel,
            axis_label=axis_label,
            output_path=artifacts.png_output_dir / f"{data['scan']}_{channel}_line.png",
        )
        logger.info("Saved rendered line-scan PNG to %s", img_path)
        wait_for_stable_file(
            img_path,
            stable_s=self.output_stable_s,
            timeout_s=self.wait_timeout_s,
            poll_s=self.poll_s,
        )
        return {
            "img_path": img_path,
            "raw_data_path": str(raw_path),
            "channel": channel,
            "h5_path": str(artifacts.h5_path),
            "mda_path": str(artifacts.mda_path),
            "gaussian_fit_params": {
                "fwhm": _json_number(fwhm) if fwhm is not None else None,
                "a": _json_number(a),
                "mu": _json_number(mu),
                "sigma": _json_number(sigma),
                "c": _json_number(c),
                "normalized_residual": _json_number(normalized_residual),
                "x_min": _json_number(x_min),
                "x_max": _json_number(x_max),
            },
        }
