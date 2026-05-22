"""Tests for EAA image buffer payload compatibility."""

from __future__ import annotations

import base64

import numpy as np

from control_suite_mcp_aps_2idd.aps_tools import AcquisitionBuffers


def decode_payload(payload: dict) -> np.ndarray:
    """Decode the dependency-light EAA image array payload."""
    return np.frombuffer(
        base64.b64decode(payload["data"]),
        dtype=np.dtype(payload["dtype"]),
    ).reshape(payload["shape"])


def test_image_buffers_dump_array_artifacts_and_payloads(tmp_path) -> None:
    """Acquisition buffers expose contract array artifacts and array payloads."""
    buffers = AcquisitionBuffers()
    buffers.image_array_artifact_dir = tmp_path
    first = np.array([[1, 2], [3, 4]], dtype=np.float32)
    second = np.array([[5, 6], [7, 8]], dtype=np.float32)

    buffers.update_image_buffers(first, psize=0.5)
    buffers.update_image_buffers(second, psize=0.25)

    current_info = buffers.get_current_image_info()
    assert "array_path" not in current_info
    assert current_info["psize"] == 0.25
    assert current_info["shape"] == [2, 2]
    assert current_info["dtype"] == "float32"

    current_payload = buffers.get_image_array_payload("current")
    previous_payload = buffers.get_image_array_payload("previous")
    initial_payload = buffers.get_image_array_payload("initial")

    assert current_payload["encoding"] == "numpy_base64"
    assert np.array_equal(decode_payload(current_payload), second)
    assert np.array_equal(decode_payload(previous_payload), first)
    assert np.array_equal(decode_payload(initial_payload), first)

    dumped = buffers.dump_array("image_k")
    assert np.array_equal(np.load(dumped["array_path"]), second)
