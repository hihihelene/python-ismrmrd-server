"""VoxelMorph inference for 2D MRI time series."""

from __future__ import annotations

import json
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


@contextmanager
def _legacy_checkpoint_paths():
    """Allow checkpoints saved on POSIX systems to load on Windows."""
    if type(Path()) is not pathlib.WindowsPath:
        yield
        return

    local_path_module = pathlib._local
    original_posix_path = pathlib.PosixPath
    original_local_posix_path = local_path_module.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    local_path_module.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path
        local_path_module.PosixPath = original_local_posix_path


def _load_checkpoint(torch, path: Path, device):
    with _legacy_checkpoint_paths():
        return torch.load(path, map_location=device, weights_only=False)


@dataclass
class _ModelArchConfig:
    ndim: int = 2
    source_channels: int = 1
    target_channels: int = 1
    nb_features: list[int] = field(default_factory=lambda: [16, 32, 32, 32, 16])
    integration_steps: int = 0
    image_size: tuple[int, int] = (256, 256)


@dataclass
class _PreprocessingArchConfig:
    target_spacing: tuple[float, float] = (1.25, 1.25)
    target_size: tuple[int, int] = (256, 256)
    normalize: bool = True


def _load_config_mapping(config_path: str | Path) -> dict[str, Any]:
    """Read a training config file (YAML or JSON) into a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text) or {}
    import yaml  # local import: only needed when a YAML config is read

    return yaml.safe_load(text) or {}


def _filtered(cls, data: dict[str, Any] | None):
    """Build ``cls`` from ``data``, dropping keys the dataclass doesn't have.

    Mirrors config.py's own forward-compatible filtering, so unknown keys in
    a newer config file don't break inference.
    """
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in valid_keys})


def load_voxelmorph_config(
    config_path: str | Path,
) -> tuple[_ModelArchConfig, _PreprocessingArchConfig]:
    """Read the ``model`` and ``dataset.preprocessing`` sections used for
    VoxelMorph inference directly from a training config file.
    """
    data = _load_config_mapping(config_path)
    model_cfg = _filtered(_ModelArchConfig, data.get("model"))
    preprocessing_data = (data.get("dataset") or {}).get("preprocessing")
    preprocessing_cfg = _filtered(_PreprocessingArchConfig, preprocessing_data)
    return model_cfg, preprocessing_cfg


# Preprocessing utility
def extract_frame(series: sitk.Image, index: int) -> sitk.Image:
    """Return frame ``index`` of a 3D (x, y, t) series as 2D image."""
    size = series.GetSize()
    return sitk.Extract(series, size=[size[0], size[1], 0], index=[0, 0, int(index)])


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float],
    interpolator=None,
) -> sitk.Image:
    """Resample a 2D image to ``target_spacing`` (same physical extent)."""
    spacing = image.GetSpacing()
    size = image.GetSize()
    target_size = [
        max(1, round(size[i] * spacing[i] / target_spacing[i])) for i in range(2)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(
        interpolator if interpolator is not None else sitk.sitkBSpline
    )
    return resampler.Execute(image)


def crop_pad(
    image: sitk.Image,
    center_physical: tuple[float, float],
    target_size: tuple[int, int],
) -> sitk.Image:
    """Crop/zero-pad ``image`` to ``target_size`` around a physical centre."""
    tx, ty = target_size
    cx, cy = image.TransformPhysicalPointToContinuousIndex(center_physical)
    start = (round(cx - tx / 2.0), round(cy - ty / 2.0))

    template = sitk.Image(int(tx), int(ty), image.GetPixelID())
    template.SetSpacing(image.GetSpacing())
    template.SetDirection(image.GetDirection())
    template.SetOrigin(image.TransformIndexToPhysicalPoint(start))

    return sitk.Resample(
        image,
        template,
        sitk.Transform(),
        sitk.sitkBSpline,
        0,
        image.GetPixelID(),
    )


def estimate_crop_center(
    frame: sitk.Image,
    smoothing_mm: float = 3.0,
    min_fraction: float = 0.05,
    max_fraction: float = 0.95,
) -> tuple[float, float]:
    """Estimate the crop centre (physical coordinates) from one 2D frame.

    Intensity-only estimate of the body centroid:
    Gaussian smoothing -> Otsu threshold -> largest connected component
    -> hole filling -> centroid.

    The centroid of the filled body is used instead of the (dark) lung
    itself: it is stable against vessels/heart and needs no segmentation.
    Falls back to the field-of-view centre if the foreground is implausible
    (empty, or covering almost the whole image).
    """
    fov_center_index = tuple((s - 1) / 2.0 for s in frame.GetSize())
    fallback = frame.TransformContinuousIndexToPhysicalPoint(fov_center_index)

    image = sitk.Cast(frame, sitk.sitkFloat32)
    arr = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(arr).all() or float(arr.max()) <= float(arr.min()):
        return fallback

    smoothed = sitk.SmoothingRecursiveGaussian(image, float(smoothing_mm))

    otsu = sitk.OtsuThresholdImageFilter()
    otsu.Execute(smoothed)
    foreground = sitk.Cast(smoothed > otsu.GetThreshold(), sitk.sitkUInt8)

    components = sitk.RelabelComponent(sitk.ConnectedComponent(foreground))
    body = sitk.BinaryFillhole(components == 1)

    body_np = sitk.GetArrayFromImage(body) > 0
    fraction = float(body_np.mean())
    if not (min_fraction <= fraction <= max_fraction):
        return fallback

    ys, xs = np.nonzero(body_np)
    return frame.TransformContinuousIndexToPhysicalPoint(
        (float(xs.mean()), float(ys.mean()))
    )


def normalize_percentile(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust [0, 1] normalisation (1st/99th percentile).

    Returns ``(normalised, low, high)``. ``low == high == 0`` signals that
    the percentile range was degenerate and no rescaling must be undone.
    """
    array = array.astype(np.float32)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high > low:
        out = (np.clip(array, low, high) - low) / (high - low)
    else:
        out = array - array.mean()
        std = float(out.std())
        if std > 1e-6:
            out = out / std
        low = high = 0.0
    return out.astype(np.float32), low, high


def prepare_frame(
    frame: sitk.Image,
    crop_center: tuple[float, float],
    target_spacing: tuple[float, float],
    target_size: tuple[int, int],
    apply_resample: bool = True,
    apply_crop: bool = True,
    apply_normalize: bool = True,
    resample_interpolator=None,
) -> tuple[np.ndarray, sitk.Image, float, float]:
    """Resample -> crop/pad -> normalise one frame.

    Returns ``(network_array, processed_sitk, low, high)``.
    """
    processed = frame
    if apply_resample:
        processed = resample_to_spacing(
            processed, target_spacing, resample_interpolator
        )
    if apply_crop:
        processed = crop_pad(processed, crop_center, target_size)

    if apply_normalize:
        array, low, high = normalize_percentile(sitk.GetArrayFromImage(processed))
    else:
        array = sitk.GetArrayFromImage(processed).astype(np.float32)
        low = high = 0.0
    return array, processed, low, high


def restore_frame(
    network_array: np.ndarray,
    processed_reference: sitk.Image,
    target_frame: sitk.Image,
    low: float,
    high: float,
    regrid: bool = True,
) -> sitk.Image:
    """Undo normalisation and map a network-space frame onto ``target_frame``.

    ``processed_reference`` supplies the geometry of the network space.
    The resampling only changes the sampling grid; it does not invert the
    learned deformation.
    """
    array = network_array.astype(np.float32)
    if high > low:
        array = array * (high - low) + low

    image = sitk.GetImageFromArray(array.astype(np.float32))
    if not regrid:
        image.CopyInformation(target_frame)
        return image

    image.CopyInformation(processed_reference)
    restored = sitk.Resample(
        image,
        target_frame,
        sitk.Transform(),
        sitk.sitkBSpline,
        0.0,
        sitk.sitkFloat32,
    )
    restored.CopyInformation(target_frame)
    return restored


def join_series(frames: list[sitk.Image], template: sitk.Image) -> sitk.Image:
    """Stack 2D frames and restore the stack geometry of ``template``."""
    stack = sitk.JoinSeries(frames)
    stack.SetSpacing(template.GetSpacing())
    stack.SetOrigin(template.GetOrigin())
    stack.SetDirection(template.GetDirection())
    return stack


class VoxelMorphRegistration:
    """Inference wrapper for the trained 2D VoxelMorph model.
        register(moving_series, fixed_index) -> sitk.Image

    * ``moving_series``: 3D image (x, y, t), warm-up frames already removed.
    * ``fixed_index``: reference frame within ``moving_series``.
    * returns a 3D image in the original geometry, one frame per input frame.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = "cuda",
    ):
        import torch
        import voxelmorph as vxm
        import os

        self._torch = torch
        current_dir = Path.cwd()
        self.checkpoint_path = Path(os.path.join(current_dir, "custom", str(checkpoint_path)))
        self.config_path = Path(os.path.join(current_dir, "custom", str(config_path)))
        print(f"Current directory: {Path.cwd()}")
        print(f"Checkpoint path: {self.checkpoint_path}")
        print(f"Config path: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"VoxelMorph checkpoint not found: {self.checkpoint_path}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        model_cfg, preprocessing_cfg = load_voxelmorph_config(self.config_path)
        self.target_spacing = tuple(float(v) for v in preprocessing_cfg.target_spacing)
        self.target_size = tuple(int(v) for v in preprocessing_cfg.target_size)

        self.model = vxm.nn.models.VxmPairwise(
            ndim=model_cfg.ndim,
            source_channels=model_cfg.source_channels,
            target_channels=model_cfg.target_channels,
            nb_features=model_cfg.nb_features,
            integration_steps=model_cfg.integration_steps,
        )
        checkpoint = _load_checkpoint(torch, self.checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.fixed_index: int | None = None
        self.last_preprocessing_metadata: dict = {}

    def _register_frame(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            moving_t = torch.from_numpy(moving).float()[None, None].to(self.device)
            fixed_t = torch.from_numpy(fixed).float()[None, None].to(self.device)
            _, warped = self.model(
                moving_t,
                fixed_t,
                return_warped_source=True,
                return_field_type="displacement",
            )
        return warped[0, 0].cpu().numpy().astype(np.float32)

    def register(
        self,
        moving_series: sitk.Image,
        fixed_index: int,
    ) -> sitk.Image:
        if moving_series.GetDimension() != 3:
            raise ValueError(
                "VoxelMorph registration expects a 3D image "
                "with time as the third dimension."
            )
        depth = moving_series.GetSize()[2]
        if depth < 2:
            raise ValueError("At least two frames are required for registration.")
        if fixed_index is None or not 0 <= int(fixed_index) < depth:
            raise ValueError(f"fixed_index={fixed_index} is outside [0, {depth - 1}].")
        fixed_index = int(fixed_index)
        self.fixed_index = fixed_index

        fixed_frame = extract_frame(moving_series, fixed_index)
        crop_center = estimate_crop_center(fixed_frame)

        prep = {
            "crop_center": crop_center,
            "target_spacing": self.target_spacing,
            "target_size": self.target_size,
        }
        fixed_np, fixed_processed, _, _ = prepare_frame(fixed_frame, **prep)

        registered_frames = []
        for t in range(depth):
            moving_frame = extract_frame(moving_series, t)
            moving_np, _, low, high = prepare_frame(moving_frame, **prep)
            warped_np = self._register_frame(moving_np, fixed_np)
            registered_frames.append(
                restore_frame(warped_np, fixed_processed, moving_frame, low, high)
            )

        self.last_preprocessing_metadata = {
            "fixed_index": fixed_index,
            "target_spacing": self.target_spacing,
            "target_size": self.target_size,
            "crop_center_physical": tuple(float(v) for v in crop_center),
            "num_frames": int(depth),
        }
        return join_series(registered_frames, moving_series)
