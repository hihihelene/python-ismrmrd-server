"""Preprocessing ablation / identity control.

Runs the preprocessing and postprocessing chain of the learned registration
methods *without* applying any network. The result quantifies how much the
image changes through resampling, cropping, normalisation and interpolation
alone.
"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

try:
    from .VoxelMorph_Registration import (
        estimate_crop_center,
        extract_frame,
        join_series,
        load_voxelmorph_config,
        prepare_frame,
        restore_frame,
    )
except ImportError:
    from VoxelMorph_Registration import (
        estimate_crop_center,
        extract_frame,
        join_series,
        load_voxelmorph_config,
        prepare_frame,
        restore_frame,
    )

# ablation_mode -> (resample, crop, normalise)
ABLATION_MODES: dict[str, tuple[bool, bool, bool]] = {
    "original": (False, False, False),
    "resample": (True, False, False),
    "resample_crop": (True, True, False),
    "full": (True, True, True),
}

# Method suffix used by the pipeline ("ablation_resample_linear", ...)
INTERPOLATORS = {
    "resample_linear": sitk.sitkLinear,
    "resample_bspline": sitk.sitkBSpline,
    "resample_nearest": sitk.sitkNearestNeighbor,
}


class PreprocessingAblation:
    def __init__(self, config_path: str | Path):
        _model_cfg, preprocessing_cfg = load_voxelmorph_config(config_path)
        self.config_path = Path(config_path)
        self.target_spacing = tuple(float(v) for v in preprocessing_cfg.target_spacing)
        self.target_size = tuple(int(v) for v in preprocessing_cfg.target_size)
        self.fixed_index: int | None = None
        self.last_preprocessing_metadata: dict = {}

    def register(
        self,
        moving_series: sitk.Image,
        fixed_index: int,
        ablation_mode: str = "resample",
        resample_interpolator=None,
    ) -> sitk.Image:
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(
                f"Unknown ablation_mode='{ablation_mode}'. "
                f"Expected one of {sorted(ABLATION_MODES)}."
            )
        if moving_series.GetDimension() != 3:
            raise ValueError("Expected a 3D image with time as third dimension.")
        depth = moving_series.GetSize()[2]
        if fixed_index is None or not 0 <= int(fixed_index) < depth:
            raise ValueError(f"fixed_index={fixed_index} is outside [0, {depth - 1}].")
        self.fixed_index = int(fixed_index)

        apply_resample, apply_crop, apply_normalize = ABLATION_MODES[ablation_mode]
        crop_center = estimate_crop_center(extract_frame(moving_series, fixed_index))

        prep = {
            "crop_center": crop_center,
            "target_spacing": self.target_spacing,
            "target_size": self.target_size,
            "apply_resample": apply_resample,
            "apply_crop": apply_crop,
            "apply_normalize": apply_normalize,
            "resample_interpolator": resample_interpolator,
        }

        frames = []
        for t in range(depth):
            moving_frame = extract_frame(moving_series, t)
            array, processed, low, high = prepare_frame(moving_frame, **prep)
            # Identity control: the network output is the network input.
            frames.append(
                restore_frame(
                    array,
                    processed,
                    moving_frame,
                    low,
                    high,
                    regrid=(ablation_mode != "original"),
                )
            )

        self.last_preprocessing_metadata = {
            "fixed_index": self.fixed_index,
            "ablation_mode": ablation_mode,
            "crop_center_physical": tuple(float(v) for v in crop_center),
        }
        return join_series(frames, moving_series)
