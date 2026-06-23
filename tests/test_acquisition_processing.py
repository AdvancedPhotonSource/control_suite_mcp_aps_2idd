from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from control_suite_mcp_aps_2idd.acquisition_processing import APSMICPostProcessor


def _write_maps_h5(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("MAPS/Scan/x_axis", data=np.linspace(-2.0, 2.0, data.shape[-1]))
        h5.create_dataset("MAPS/Scan/y_axis", data=np.linspace(-1.0, 1.0, data.shape[-2]))
        group = h5.create_group("MAPS/XRF_Analyzed/ROI")
        group.create_dataset("Counts_Per_Sec", data=data[np.newaxis, ...])
        group.create_dataset("Channel_Names", data=np.array([b"Cr"]))
        nnls = h5.create_group("MAPS/XRF_Analyzed/NNLS")
        nnls.create_dataset("Counts_Per_Sec", data=data[np.newaxis, ...])
        nnls.create_dataset("Channel_Names", data=np.array([b"Cr"]))


def test_process_image_writes_absolute_png_and_npy(tmp_path: Path) -> None:
    current_mda_file = "2idd_0001.mda"
    _write_maps_h5(tmp_path / "img.dat" / f"{current_mda_file}.h50", np.arange(12).reshape(3, 4))
    processor = APSMICPostProcessor(h5_stable_s=0, output_stable_s=0, poll_s=0)

    result = processor.process_image(
        save_data_path=str(tmp_path),
        current_mda_file=current_mda_file,
        channels=("Cr",),
        using_xrf_maps=False,
        plot_in_log_scale=False,
        show_colorbar=False,
    )

    assert Path(result["img_path"]).is_absolute()
    assert Path(result["img_path"]).exists()
    assert Path(result["raw_data_path"]).is_absolute()
    np.testing.assert_array_equal(np.load(result["raw_data_path"]), np.arange(12).reshape(3, 4))


def test_process_line_scan_writes_fit_plot_raw_data_and_parameters(tmp_path: Path) -> None:
    current_mda_file = "2idd_0002.mda"
    x = np.linspace(-5.0, 5.0, 41)
    y = 12.0 * np.exp(-((x - 1.0) ** 2) / (2 * 1.2**2)) + 0.5
    _write_maps_h5(tmp_path / "img.dat" / f"{current_mda_file}.h50", y[np.newaxis, :])
    processor = APSMICPostProcessor(h5_stable_s=0, output_stable_s=0, poll_s=0)

    result = processor.process_line_scan(
        save_data_path=str(tmp_path),
        current_mda_file=current_mda_file,
        channels=("Cr",),
        roi_num=16,
        using_xrf_maps=False,
        scan_samy=False,
    )

    assert Path(result["img_path"]).is_absolute()
    assert Path(result["img_path"]).exists()
    assert Path(result["raw_data_path"]).is_absolute()
    raw = np.load(result["raw_data_path"])
    assert raw.shape == (41, 2)
    assert result["gaussian_fit_params"]["fwhm"] is not None
    assert result["gaussian_fit_params"]["a"] is not None
    assert result["gaussian_fit_params"]["mu"] is not None
    assert result["gaussian_fit_params"]["sigma"] is not None
