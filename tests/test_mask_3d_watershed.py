"""Regression tests for the watershed shared by masks and localizations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from imageProcessing import segmentMasks, segmentMasks3D


def joined_objects():
    z, x, y = np.indices((40, 64, 48))
    first = (z - 20) ** 2 + (x - 18) ** 2 + (y - 24) ** 2 <= 6 ** 2
    second = (z - 20) ** 2 + (x - 40) ** 2 + (y - 24) ** 2 <= 6 ** 2
    foreground = first | second
    foreground[20, 18:41, 24] = True
    return foreground


def test_watershed_separates_joined_objects():
    foreground = joined_objects()
    labels = segmentMasks._deblend_3d_segmentation_advanced(foreground)
    assert labels[20, 18, 24] > 0
    assert labels[20, 40, 24] > 0
    assert labels[20, 18, 24] != labels[20, 40, 24]
    np.testing.assert_array_equal(labels > 0, foreground)


def test_empty_foreground():
    labels = segmentMasks._deblend_3d_segmentation_advanced(np.zeros((8, 8, 8)))
    assert not labels.any()


def test_unseeded_border_component_is_preserved():
    foreground = joined_objects()
    foreground[0, 0, 0] = True
    labels = segmentMasks._deblend_3d_segmentation_advanced(foreground)
    assert labels[0, 0, 0] > 0
    np.testing.assert_array_equal(labels > 0, foreground)


def test_only_elongated_objects_without_regular_reference():
    foreground = np.zeros((40, 64, 48), dtype=bool)
    foreground[18:23, 8:55, 22:27] = True
    labels = segmentMasks._deblend_3d_segmentation_advanced(foreground)
    np.testing.assert_array_equal(labels > 0, foreground)


def test_mask_file_saves_watershed_labels_without_merging(tmp_path, monkeypatch):
    foreground = joined_objects()
    params = SimpleNamespace(
        decode_file_parts=lambda filename: {"cycle": "mask0"},
    )
    processor = segmentMasks3D.Mask3D(params)
    processor.dict_shifts = {"ROI:1": {"mask0": [0, 0]}}
    monkeypatch.setattr(segmentMasks3D.io, "imread", lambda filename: foreground)
    monkeypatch.setattr(
        processor, "_segment_3d_volumes",
        lambda *args: (foreground, foreground.astype(np.int32)),
    )
    monkeypatch.setattr(segmentMasks3D, "plot_image_3d", Mock(return_value=Mock()))
    (tmp_path / "mask_3d" / "data").mkdir(parents=True)
    seg_params = SimpleNamespace(mask_2d_folder="mask_2d", mask_3d_folder="mask_3d")
    processor.segment_masks_3d_file(
        "test_mask0.tif", str(tmp_path), seg_params, "1",
        SimpleNamespace(zBinning=1), "mask0",
    )
    saved = np.load(tmp_path / "mask_3d" / "data" / "test_mask0_3Dmasks.npy")
    assert saved[20, 18, 24] != saved[20, 40, 24]
    np.testing.assert_array_equal(saved > 0, foreground)
    projection = np.load(tmp_path / "mask_2d" / "data" / "test_mask0_Masks.npy")
    np.testing.assert_array_equal(projection, saved.max(axis=0))
