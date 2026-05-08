"""Local utility routines copied from the APS MIC EAA tool dependencies."""

from __future__ import annotations

from math import inf
from pathlib import Path
from typing import Any
import datetime
import logging
import os
import subprocess
import time

import h5py
import matplotlib
import numpy as np
import scipy.ndimage
import scipy.optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def get_timestamp(as_int: bool = False) -> str | int:
    """Return the current timestamp.

    Parameters
    ----------
    as_int
        When True, return the timestamp as an integer.

    Returns
    -------
    str | int
        Formatted timestamp.
    """
    if as_int:
        return int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def wait_for_file(file_path: str, duration: int = 30, timeout: int | float | None = inf) -> bool:
    """Wait for a file to exist and stop changing.

    Parameters
    ----------
    file_path
        Path to watch.
    duration
        Stable period in seconds required before returning success.
    timeout
        Maximum wait time in seconds. None disables the timeout.

    Returns
    -------
    bool
        True when the file exists and remains unchanged for ``duration`` seconds.
    """
    time_diff = 0.0
    time_mod = 0.0
    while any([time_diff < duration, not os.path.exists(file_path)]):
        if timeout is not None and time_diff > timeout:
            return False
        time.sleep(1)
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) != time_mod:
                time_mod = os.path.getmtime(file_path)
            time_diff = time.time() - time_mod
            logger.info("File %s exists.", file_path)
            logger.info(
                "Watching file until it remains unchanged for %s seconds.",
                duration,
            )
        else:
            logger.info("File %s does not exist. Waiting %s seconds.", file_path, duration)
            time.sleep(duration)
    return True


def validate_position_in_range(
    center: float | None,
    allowable_range: tuple[float, float] | None,
    axis_label: str,
) -> None:
    """Validate that a center position lies within an allowable range."""
    if allowable_range is None:
        return
    if len(allowable_range) != 2:
        raise ValueError(
            f"The allowable range for the {axis_label} direction must contain exactly two values."
        )
    lower, upper = allowable_range
    if lower > upper:
        raise ValueError(
            f"The allowable range for the {axis_label} direction "
            f"({allowable_range}) has the lower bound greater than the upper bound."
        )
    if center is None:
        raise ValueError(
            f"The scan center position in the {axis_label} direction must be provided "
            "when an allowable range is set."
        )
    if not lower <= center <= upper:
        raise ValueError(
            f"The scan center position in the {axis_label} direction {center} um is out "
            f"of the allowable range {allowable_range} um."
        )


def run_xrfmaps_exe(exe_path: str, args: list[str] | None = None) -> subprocess.CompletedProcess | None:
    """Run the XRF-Maps executable with optional arguments."""
    command = [exe_path]
    if args:
        command.extend(args)
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Error executing %s: %s", exe_path, exc)
        return None


def process_xrfdata(
    parent_dir: str,
    scan_num_mda: str,
    det_range: str = "0:0",
    quantify_with: str = "maps_standardinfo.txt",
    fitting_type: str = "roi, nnls",
    num_iter: int = 20000,
    exe_path: str = "/mnt/micdata1/XRF-Maps/bin/xrf_maps.exe",
) -> int | None:
    """Process an MDA file with the XRF-Maps executable."""
    mda_dir = os.path.join(parent_dir, "mda")
    mda_files = [filename for filename in os.listdir(mda_dir) if filename.endswith(".mda")]
    if scan_num_mda not in mda_files:
        logger.error("MDA file %s not found in %s", scan_num_mda, mda_dir)
        return None
    args = [
        "--dir",
        parent_dir,
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
    logger.info("Fitting %s with XRF-Maps.", scan_num_mda)
    result = run_xrfmaps_exe(exe_path, args)
    if result is None:
        return None
    logger.info("Fitting %s completed.", scan_num_mda)
    return result.returncode


def gaussian_1d(x: np.ndarray, a: float, mu: float, sigma: float, c: float = 0) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c


def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    y_threshold: float = 0,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit a one-dimensional Gaussian to finite data points."""
    x_data = np.array(x, dtype=float)
    y_data = np.array(y, dtype=float)
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite_mask]
    y_data = y_data[finite_mask]
    if x_data.size < 5:
        logger.error("Too few finite data points for Gaussian fitting.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]
    x_min = float(np.min(x_data))
    x_max_input = float(np.max(x_data))
    x_span = x_max_input - x_min
    if not np.isfinite(x_span) or x_span <= 0:
        logger.error("Invalid x range for Gaussian fitting.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_max = float(np.max(y_data))
    y_min = float(np.min(y_data))
    y_range = y_max - y_min
    if not np.isfinite(y_range) or np.isclose(y_range, 0.0):
        logger.error("Input data are too flat for Gaussian fitting.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_smooth = scipy.ndimage.gaussian_filter1d(
        y_data,
        sigma=max(1.0, x_data.size * 0.02),
        mode="nearest",
    )
    y_smooth_min = float(np.min(y_smooth))
    y_smooth_range = float(np.max(y_smooth) - y_smooth_min)
    if not np.isfinite(y_smooth_range) or np.isclose(y_smooth_range, 0.0):
        logger.error("Smoothed data are too flat for Gaussian fitting.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    x_peak = float(x_data[np.argmax(y_smooth)])
    offset = x_peak
    x_offset = x_data - offset
    fit_threshold = y_smooth_min + y_threshold * y_smooth_range
    mask = y_smooth >= fit_threshold
    if int(np.count_nonzero(mask)) < 5:
        mask = np.ones_like(y_data, dtype=bool)

    positive_weight = np.clip(y_smooth - y_smooth_min, a_min=0.0, a_max=None)
    mu_guess = 0.0
    if np.sum(positive_weight) > 0:
        mu_guess = float(np.sum(x_offset * positive_weight) / np.sum(positive_weight))

    width_mask = y_smooth >= (y_smooth_min + 0.5 * y_smooth_range)
    x_above_half = x_offset[width_mask]
    if x_above_half.size >= 2:
        sigma_guess = float((x_above_half.max() - x_above_half.min()) / 2.355)
    else:
        sigma_guess = x_span / 6.0
    sigma_guess = float(np.clip(sigma_guess, x_span / 100.0, x_span))

    a_guess = max(y_smooth_range, np.finfo(float).eps)
    c_guess = float(np.median(y_data[~mask])) if np.any(~mask) else y_smooth_min
    lower_bounds = [0.0, float(np.min(x_offset)), max(x_span / 1000.0, 1e-12), y_min - y_range]
    upper_bounds = [float(2 * y_range + abs(y_max)), float(np.max(x_offset)), float(2 * x_span), y_max + y_range]

    def run_fit(current_mask: np.ndarray, current_p0: list[float]) -> np.ndarray | None:
        if int(np.count_nonzero(current_mask)) < 5:
            return None
        try:
            popt, _ = scipy.optimize.curve_fit(
                gaussian_1d,
                x_offset[current_mask],
                y_data[current_mask],
                p0=current_p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
            return popt
        except (RuntimeError, ValueError):
            return None

    popt = run_fit(mask, [a_guess, mu_guess, sigma_guess, c_guess])
    if popt is None:
        y_smooth_retry = scipy.ndimage.gaussian_filter1d(
            y_data,
            sigma=max(2.0, x_data.size * 0.05),
            mode="nearest",
        )
        retry_weight = np.clip(y_smooth_retry - np.min(y_smooth_retry), a_min=0.0, a_max=None)
        mu_retry = 0.0
        if np.sum(retry_weight) > 0:
            mu_retry = float(np.sum(x_offset * retry_weight) / np.sum(retry_weight))
        retry_range = np.max(y_smooth_retry) - np.min(y_smooth_retry)
        retry_mask = y_smooth_retry >= (
            np.min(y_smooth_retry) + max(0.1, y_threshold) * retry_range
        )
        popt = run_fit(retry_mask, [a_guess, mu_retry, sigma_guess, c_guess])

    if popt is None:
        logger.error("Failed to fit Gaussian to data.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_fit = gaussian_1d(x_offset, *popt)
    amplitude = float(popt[0])
    sigma = float(popt[2])
    mu = float(popt[1])
    if sigma <= 0 or not (float(np.min(x_offset)) <= mu <= float(np.max(x_offset))):
        logger.error("Gaussian fit parameters are invalid.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input
    normalized_residual = np.nan
    if not np.isclose(amplitude, 0.0):
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


def plot_xrf_line_scan(
    x: np.ndarray,
    y: np.ndarray,
    val_gauss: np.ndarray | None,
    fwhm: float,
    scan_name: str,
    roi_num: int,
    scan_samy: bool = False,
) -> plt.Figure:
    """Plot XRF line-scan data and an optional Gaussian fit."""
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.plot(x, y, label="data")
    if val_gauss is not None:
        ax.plot(x, val_gauss, linestyle="--", color="red", label="fit")
    ax.text(
        0.05,
        0.95,
        f"FWHM = {fwhm:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax.legend()
    ax.set_xlabel("X-axis Position" if not scan_samy else "Y-axis Position")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{scan_name}-{roi_num}")
    ax.grid(True)
    plt.tight_layout()
    return fig


def save_xrf_line_scan(
    mda_path: str,
    output_dir: str,
    roi_num: int,
    y_threshold: float = 0.0,
    return_line_array: bool = False,
    scan_samy: bool = False,
) -> str | tuple[str, dict[str, Any]] | None:
    """Save XRF line-scan data as a PNG and optionally return fit metadata."""
    try:
        from mic_vis.s2idd.mda import get_roi_from_mda
    except ImportError as exc:
        raise ImportError("mic_vis is required for APS 2-ID-D line-scan processing.") from exc

    try:
        roi_data, position_data = get_roi_from_mda(mda_path, roi_num)
    except Exception as exc:
        logger.error("Failed to get ROI or position data from %s: %s", mda_path, exc)
        return None

    if any(data is None for data in [roi_data, position_data]):
        logger.error("Failed to get ROI or position data from %s", mda_path)
        return None

    order = np.argsort(position_data)
    x = position_data[order]
    y = roi_data[order]
    try:
        a, mu, sigma, c, normalized_residual, x_min, x_max = fit_gaussian_1d(
            x,
            y,
            y_threshold=y_threshold,
        )
        if np.any(np.isnan([a, mu, sigma, c])):
            val_gauss = None
            fwhm = np.nan
        else:
            val_gauss = gaussian_1d(x, a, mu, sigma, c)
            fwhm = 2.35 * abs(sigma)
    except RuntimeError as exc:
        logger.error("Failed to fit Gaussian to data: %s", exc)
        a = mu = sigma = c = normalized_residual = x_min = x_max = np.nan
        val_gauss = None
        fwhm = np.nan

    scan_name = os.path.basename(mda_path)
    fig = plot_xrf_line_scan(x, y, val_gauss, fwhm, scan_name, roi_num, scan_samy=scan_samy)
    output_path = Path(output_dir) / f"{scan_name.replace('.mda', '')}_ROI{roi_num}.png"
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Image saved to %s", output_path)
    if return_line_array:
        return str(output_path), {
            "x": x,
            "y": y,
            "val_gauss": val_gauss,
            "fwhm": fwhm,
            "a": a,
            "mu": mu,
            "sigma": sigma,
            "c": c,
            "normalized_residual": normalized_residual,
            "x_min": x_min,
            "x_max": x_max,
        }
    return str(output_path)


def load_h5(img_h5_path: str, fit_type: list[str] | None = None, fsizelim: float = 1e3) -> dict[str, Any] | None:
    """Load XRF data from an HDF5 file."""
    if fit_type is None:
        fit_type = ["NNLS", "ROI"]
    data: dict[str, Any] = {}
    if os.path.getsize(img_h5_path) <= fsizelim:
        logger.error("The XRF h5 file %s not found or too small.", img_h5_path)
        return None
    with h5py.File(img_h5_path, "r") as handle:
        data.update({"scan": os.path.basename(img_h5_path)})
        data.update({"x_axis": handle["MAPS/Scan/x_axis"][:]})
        data.update({"y_axis": handle["MAPS/Scan/y_axis"][:]})
        for fit_name in fit_type:
            counts = handle[f"MAPS/XRF_Analyzed/{fit_name}/Counts_Per_Sec"][:]
            channels = handle[f"MAPS/XRF_Analyzed/{fit_name}/Channel_Names"][:].astype(str).tolist()
            scaler_names = handle["MAPS/Scalers/Names"][:].astype(str).tolist()
            scaler_values = handle["MAPS/Scalers/Values"][:]
            data.update({f"{fit_name}_arr": counts})
            data.update({f"{fit_name}_ch": channels})
            data.update({f"{fit_name}_scaler_names": scaler_names})
            data.update({f"{fit_name}_scaler_values": scaler_values})
    return data


def plot_xrfdata(
    plotarr: np.ndarray,
    xaxis: np.ndarray,
    yaxis: np.ndarray,
    scan_name: str,
    elm_name: str,
    cmap: str,
    vmax: float,
    vmin: float,
    plot_in_log_scale: bool = False,
    show_colorbar: bool = False,
) -> plt.Figure:
    """Plot XRF image data."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if plot_in_log_scale:
        plotarr = np.log10(plotarr + 1)
        vmax = np.log10(vmax + 1)
        vmin = np.log10(vmin + 1)
    image = ax.imshow(plotarr, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(f"{scan_name} {elm_name}")
    if show_colorbar:
        colorbar = fig.colorbar(image)
        colorbar.set_label("Intensity")
    xticks = np.linspace(0, len(xaxis) - 1, 5, dtype=int)
    yticks = np.linspace(0, len(yaxis) - 1, 5, dtype=int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([np.round(xaxis[index], 2) for index in xticks])
    ax.set_yticklabels([np.round(yaxis[index], 2) for index in yticks])
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    return fig


def save_xrfdata(
    img_h5_path: str,
    output_dir: str,
    cmap: str = "inferno",
    elms: list[str] | tuple[str, ...] | None = None,
    vmax_th: float = 99,
    vmin: float = 0,
    plot_in_log_scale: bool = False,
    return_image_array: bool = False,
    show_colorbar_in_image: bool = False,
) -> str | tuple[str | None, np.ndarray | None] | None:
    """Save XRF image data as PNG."""
    data = load_h5(img_h5_path)
    if not data:
        logger.error("The XRF h5 file %s not found.", img_h5_path)
        if return_image_array:
            return None, None
        return None

    data_arr = data["ROI_arr"]
    data_ch = data["ROI_ch"]
    xaxis = data["x_axis"]
    yaxis = data["y_axis"]
    plot_elms = elms if elms else data_ch
    for element in plot_elms:
        plotarr = data_arr[data_ch.index(element)]
        vmax = np.nanpercentile(plotarr, vmax_th)
        fig = plot_xrfdata(
            plotarr,
            xaxis,
            yaxis,
            data["scan"],
            element,
            cmap,
            vmax,
            vmin,
            plot_in_log_scale=plot_in_log_scale,
            show_colorbar=show_colorbar_in_image,
        )
        output_path = Path(output_dir) / f"{data['scan']}_{element}.png"
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("Image saved to %s", output_path)
        if return_image_array:
            return str(output_path), plotarr
        return str(output_path)
    if return_image_array:
        return None, None
    return None
