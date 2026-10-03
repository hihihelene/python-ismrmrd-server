"""HyperMorph inference for 2D MRI time series."""

from __future__ import annotations

import json
import pathlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

try:
    from Scripts.torch.networks import HyperVxmPairwise
except ImportError:
    from .Scripts.torch.networks import HyperVxmPairwise

# Semi-supervised HyperMorph takes exactly [lambda_reg, gamma_seg].
_NB_HYPERPARAMETERS = 2

# The network output may be a dict; these are the keys we look for.
_WARPED_KEYS = ("warped", "warped_source", "moved", "moved_source")
_DISPLACEMENT_KEYS = ("field", "displacement", "displacement_field", "flow")

ConfigT = TypeVar("ConfigT")


class RegistrationCancelled(Exception):
    """Raised when a running register() call is cancelled via ``cancel``."""


@dataclass
class ModelConfig:
    """Architecture parameters for the 2D VoxelMorph network."""

    ndim: int = 2
    source_channels: int = 1
    target_channels: int = 1
    nb_features: list[int] = field(default_factory=lambda: [16, 32, 32, 32, 16])
    integration_steps: int = 0
    image_size: tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class HyperMorphConfig:
    """Configuration for paper-style HyperMorph.

    ``lambda_reg`` weights the deformation regularization, ``gamma_seg`` the
    segmentation term. During training, ``lambda_reg`` is sampled with endpoint
    oversampling (80 % Uniform(0, 1), 20 % from {0, 1}) and ``gamma_seg`` from
    Uniform(0, 1).
    """

    enabled: bool = True

    # HyperMorph architecture
    nb_hyp_params: int = 2
    nb_hyp_layers: int = 2
    nb_hyp_units: int = 64

    # Hyperparameter indices
    lambda_reg_index: int = 0
    gamma_seg_index: int = 1

    # Hyperparameter sampling
    endpoint_fraction: float = 0.20
    min_value: float = 0.0
    max_value: float = 1.0

    # Validation-time hyperparameter optimization
    tune_epochs: int = 20
    tune_lr: float = 0.05
    tune_optimizer: str = "sgd"
    tune_init: tuple[float, float] = (0.5, 0.5)


@dataclass
class PreprocessingConfig:
    """Preprocessing parameters used for HyperMorph inference."""

    target_spacing: tuple[float, float] = (1.25, 1.25)
    target_size: tuple[int, int] = (256, 256)
    normalize: bool = True


class HyperMorph(nn.Module):
    """HyperMorph on top of VoxelMorph's ``HyperVxmPairwise``.

    (VoxelMorph 0.3.3 dev branch, PyTorch.) A hypernetwork maps
    ``[lambda_reg, gamma_seg]`` to the parameters of the registration network.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hyper_config: HyperMorphConfig,
    ):
        super().__init__()

        if hyper_config.nb_hyp_params != _NB_HYPERPARAMETERS:
            raise ValueError(
                "Paper-style semi-supervised HyperMorph requires "
                "exactly two hyperparameters: "
                "[lambda_reg, gamma_seg]."
            )

        self.hyper_config = hyper_config
        self.net = HyperVxmPairwise(
            ndim=model_config.ndim,
            source_channels=model_config.source_channels,
            target_channels=model_config.target_channels,
            nb_features=tuple(model_config.nb_features),
            nb_hyp_params=hyper_config.nb_hyp_params,
            nb_hyp_layers=hyper_config.nb_hyp_layers,
            nb_hyp_units=hyper_config.nb_hyp_units,
            integration_steps=model_config.integration_steps,
        )

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        hyperparameters: torch.Tensor,
        return_warped_source: bool = True,
        return_field_type: str = "displacement",
    ):
        if hyperparameters.ndim != 2 or hyperparameters.shape[1] != _NB_HYPERPARAMETERS:
            raise ValueError(
                f"hyperparameters must have shape [batch_size, {_NB_HYPERPARAMETERS}] "
                f"([lambda_reg, gamma_seg]), got {tuple(hyperparameters.shape)}"
            )

        if hyperparameters.shape[0] != moving.shape[0]:
            raise ValueError(
                "The hyperparameter batch dimension must match "
                "the image batch dimension. "
                f"Got {hyperparameters.shape[0]} vs "
                f"{moving.shape[0]}."
            )

        return self.net(
            moving,
            fixed,
            hyperparameters=hyperparameters,
            return_warped_source=return_warped_source,
            return_field_type=return_field_type,
        )


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


def _dataclass_from_mapping(cls: type[ConfigT], data: dict[str, Any] | None) -> ConfigT:
    """Build ``cls`` from ``data``, ignoring keys the dataclass doesn't define."""
    valid_names = {f.name for f in fields(cls)}
    known_items = {k: v for k, v in (data or {}).items() if k in valid_names}
    return cls(**known_items)


def load_hypermorph_config(
    config_path: str | Path,
) -> tuple[ModelConfig, HyperMorphConfig, PreprocessingConfig]:
    """Read the ``model``, ``hypermorph`` and ``dataset.preprocessing`` sections
    used for HyperMorph inference from a training config file.
    """
    data = _load_config_mapping(config_path)
    model_config = _dataclass_from_mapping(ModelConfig, data.get("model"))
    hyper_config = _dataclass_from_mapping(HyperMorphConfig, data.get("hypermorph"))
    preprocessing_data = (data.get("dataset") or {}).get("preprocessing")
    preprocessing_config = _dataclass_from_mapping(
        PreprocessingConfig, preprocessing_data
    )
    return model_config, hyper_config, preprocessing_config


# Preprocessing utility
def extract_frame(series: sitk.Image, index: int) -> sitk.Image:
    size = series.GetSize()
    return sitk.Extract(series, size=[size[0], size[1], 0], index=[0, 0, int(index)])


def resample_to_spacing(
    image: sitk.Image, target_spacing: tuple[float, float]
) -> sitk.Image:
    output_size = [
        max(1, round(size * spacing / target))
        for size, spacing, target in zip(
            image.GetSize(), image.GetSpacing(), target_spacing
        )
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(image)


def crop_pad(
    image: sitk.Image,
    center_physical: tuple[float, float],
    target_size: tuple[int, int],
) -> sitk.Image:
    """Crop or zero-pad ``image`` to ``target_size`` around ``center_physical``."""
    width, height = target_size
    center_x, center_y = image.TransformPhysicalPointToContinuousIndex(center_physical)
    start = (round(center_x - width / 2.0), round(center_y - height / 2.0))

    template = sitk.Image(int(width), int(height), image.GetPixelID())
    template.SetSpacing(image.GetSpacing())
    template.SetDirection(image.GetDirection())
    template.SetOrigin(image.TransformIndexToPhysicalPoint(start))
    return sitk.Resample(
        image, template, sitk.Transform(), sitk.sitkBSpline, 0, image.GetPixelID()
    )


def estimate_crop_center(
    frame: sitk.Image,
    smoothing_mm: float = 3.0,
    min_fraction: float = 0.05,
    max_fraction: float = 0.95,
) -> tuple[float, float]:
    """Intensity-only body centroid of a 2D frame (physical coordinates).

    Gaussian smoothing -> Otsu -> largest component -> hole filling ->
    centroid. Falls back to the field-of-view centre if implausible.
    """
    fov_center_index = tuple((s - 1) / 2.0 for s in frame.GetSize())
    fallback = frame.TransformContinuousIndexToPhysicalPoint(fov_center_index)

    image = sitk.Cast(frame, sitk.sitkFloat32)
    pixels = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(pixels).all() or pixels.max() <= pixels.min():
        return fallback

    smoothed = sitk.SmoothingRecursiveGaussian(image, smoothing_mm)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.Execute(smoothed)
    foreground = sitk.Cast(smoothed > otsu.GetThreshold(), sitk.sitkUInt8)

    labels = sitk.RelabelComponent(sitk.ConnectedComponent(foreground))
    body = sitk.BinaryFillhole(labels == 1)  # label 1 = largest component

    body_mask = sitk.GetArrayFromImage(body) > 0
    if not min_fraction <= body_mask.mean() <= max_fraction:
        return fallback

    ys, xs = np.nonzero(body_mask)
    return frame.TransformContinuousIndexToPhysicalPoint(
        (float(xs.mean()), float(ys.mean()))
    )


def normalize_percentile(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale ``array`` to [0, 1] between its 1st and 99th percentile.

    Returns the normalized array and the ``(low, high)`` percentiles needed to
    undo the scaling. If both percentiles coincide, the array is z-scored
    instead and ``(0.0, 0.0)`` is returned, meaning "nothing to undo".
    """
    array = array.astype(np.float32)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high > low:
        normalized = (np.clip(array, low, high) - low) / (high - low)
    else:
        normalized = array - array.mean()
        std = float(normalized.std())
        if std > 1e-6:
            normalized = normalized / std
        low = high = 0.0
    return normalized.astype(np.float32), low, high


# Model utility
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    with _legacy_checkpoint_paths():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("Invalid HyperMorph checkpoint format.")
    return checkpoint


def make_model(
    model_cfg: ModelConfig,
    hyper_cfg: HyperMorphConfig,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> HyperMorph:
    model = HyperMorph(model_config=model_cfg, hyper_config=hyper_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _first_tensor_by_key(
    output: dict[str, Any], keys: tuple[str, ...]
) -> torch.Tensor | None:
    for key in keys:
        if torch.is_tensor(output.get(key)):
            return output[key]
    return None


def extract_warped_output(output: Any, reference: torch.Tensor) -> torch.Tensor:
    """Find the warped image in a model output (tensor, dict or sequence)."""
    if torch.is_tensor(output):
        warped = output
    elif isinstance(output, dict):
        warped = _first_tensor_by_key(output, _WARPED_KEYS)
    elif isinstance(output, (tuple, list)):
        warped = next(
            (v for v in output if torch.is_tensor(v) and v.shape == reference.shape),
            None,
        )
    else:
        warped = None
    if warped is None:
        raise RuntimeError("Could not extract warped image from HyperMorph output.")
    return warped[:, 0] if warped.ndim == 4 else warped


def extract_displacement_field(output: Any) -> torch.Tensor | None:
    if isinstance(output, dict):
        return _first_tensor_by_key(output, _DISPLACEMENT_KEYS)
    if isinstance(output, (tuple, list)):
        return next(
            (
                v
                for v in output
                if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] in (2, 3)
            ),
            None,
        )
    return None


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


@torch.inference_mode()
def predict(
    trained_model: nn.Module,
    moving_np: np.ndarray,
    fixed_np: np.ndarray,
    hyperparameters: tuple[float, float],
    device: torch.device,
) -> dict[str, np.ndarray | None]:
    """Register one 2D frame pair.

    Returns ``{"moved": (H, W) array, "field": (2, H, W) array or None}``.
    """
    moving = torch.from_numpy(np.asarray(moving_np, dtype=np.float32))[None, None]
    fixed = torch.from_numpy(np.asarray(fixed_np, dtype=np.float32))[None, None]
    hyper = torch.tensor([list(map(float, hyperparameters))], dtype=torch.float32)
    output = trained_model(
        moving.to(device),
        fixed.to(device),
        hyperparameters=hyper.to(device),
        return_warped_source=True,
        return_field_type="displacement",
    )
    warped = extract_warped_output(output, moving)
    displacement = extract_displacement_field(output)
    return {
        "moved": _to_numpy(warped[0]),
        "field": _to_numpy(displacement[0]) if displacement is not None else None,
    }


class HyperMorphRegistration:
    """Pipeline-compatible HyperMorph registration.

    Parameters
    ----------
    checkpoint_path:
        Checkpoint containing ``model_state_dict``.
    config_path:
        Training config file (YAML or JSON). Read internally for the
        network architecture (``model``/``hypermorph`` sections) and the
        network-space geometry (``dataset.preprocessing.target_spacing`` /
        ``target_size``); no separate geometry arguments are needed.
    device:
        ``"cuda"`` (falls back to CPU) or ``"cpu"``.

    After a ``register()`` call, ``fixed_index``, ``lambda_reg``, ``gamma_seg``
    and ``last_disp_fields`` (shape ``(T, H, W, 2)`` in network space) are
    available as attributes.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = "cuda",
    ):
        import os
        current_dir = Path(__file__).parent
        self.checkpoint_path = Path(os.path.join(current_dir, checkpoint_path))
        self.config_path = Path(os.path.join(current_dir, config_path))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"HyperMorph checkpoint not found: {self.checkpoint_path}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        model_config, hyper_config, preprocessing_config = load_hypermorph_config(
            self.config_path
        )
        self.target_spacing = tuple(
            float(v) for v in preprocessing_config.target_spacing
        )
        self.target_size = tuple(int(v) for v in preprocessing_config.target_size)

        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        self.model = make_model(model_config, hyper_config, checkpoint, self.device)

        self.fixed_index: int | None = None
        self.lambda_reg: float | None = None
        self.gamma_seg: float | None = None
        self.last_disp_fields: np.ndarray | None = None
        self.last_preprocessing_metadata: dict[str, Any] = {}

    def _preprocess_frame(
        self, frame: sitk.Image, crop_center: tuple[float, float]
    ) -> tuple[np.ndarray, sitk.Image, float, float]:
        """Return the normalized array, the processed image and its (low, high)."""
        processed = resample_to_spacing(frame, self.target_spacing)
        processed = crop_pad(processed, crop_center, self.target_size)
        array, low, high = normalize_percentile(sitk.GetArrayFromImage(processed))
        return array, processed, low, high

    @staticmethod
    def _restore_frame(
        array: np.ndarray,
        processed_reference: sitk.Image,
        original_frame: sitk.Image,
        low: float,
        high: float,
    ) -> sitk.Image:
        """Undo the normalization and resample back onto ``original_frame``."""
        if high > low:
            array = array * (high - low) + low
        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        image.CopyInformation(processed_reference)
        return sitk.Resample(
            image,
            original_frame,
            sitk.Transform(),
            sitk.sitkBSpline,
            0.0,
            sitk.sitkFloat32,
        )

    def register(
        self,
        moving_series: sitk.Image,
        fixed_index: int,
        lambda_reg: float | None = None,
        gamma_seg: float | None = None,
        cancel: threading.Event | None = None,
    ) -> sitk.Image:
        """Register every frame of ``moving_series`` to frame ``fixed_index``.

        Parameters
        ----------
        moving_series:
            3D image (x, y, t), warm-up frames already removed.
        fixed_index:
            Reference frame within ``moving_series``.
        lambda_reg, gamma_seg:
            HyperMorph hyperparameters; both are required. ``gamma_seg`` stays
            a network input even though no segmentation is provided.
        cancel:
            If set during registration, ``RegistrationCancelled`` is raised
            before the next frame is processed.

        Returns
        -------
        A 3D image in the original geometry, one frame per input frame.
        """
        if lambda_reg is None or gamma_seg is None:
            raise ValueError("register() requires lambda_reg and gamma_seg.")
        if moving_series.GetDimension() != 3:
            raise ValueError(
                "HyperMorph registration expects a 3D image "
                "with time as the third dimension."
            )
        num_frames = moving_series.GetSize()[2]
        if num_frames < 2:
            raise ValueError("At least two frames are required for registration.")
        if fixed_index is None or not 0 <= int(fixed_index) < num_frames:
            raise ValueError(
                f"fixed_index={fixed_index} is outside [0, {num_frames - 1}]."
            )

        self.fixed_index = int(fixed_index)
        self.lambda_reg = float(lambda_reg)
        self.gamma_seg = float(gamma_seg)
        hyperparameters = (self.lambda_reg, self.gamma_seg)

        fixed_frame = extract_frame(moving_series, self.fixed_index)
        crop_center = estimate_crop_center(fixed_frame)
        fixed_array, fixed_image, _, _ = self._preprocess_frame(
            fixed_frame, crop_center
        )
        zero_field = np.zeros((2, *fixed_array.shape), dtype=np.float32)

        registered_frames: list[sitk.Image] = []
        displacement_fields: list[np.ndarray] = []
        for t in range(num_frames):
            if cancel is not None and cancel.is_set():
                raise RegistrationCancelled

            moving_frame = extract_frame(moving_series, t)
            moving_array, _, low, high = self._preprocess_frame(
                moving_frame, crop_center
            )

            if t == self.fixed_index:
                moved, displacement = fixed_array, zero_field
            else:
                result = predict(
                    self.model, moving_array, fixed_array, hyperparameters, self.device
                )
                moved = result["moved"]
                displacement = result["field"]
                if displacement is None:
                    displacement = zero_field

            displacement_fields.append(displacement)
            registered_frames.append(
                self._restore_frame(moved, fixed_image, moving_frame, low, high)
            )

        registered_stack = sitk.JoinSeries(registered_frames)
        registered_stack.CopyInformation(moving_series)

        self.last_disp_fields = np.stack(
            [np.moveaxis(f, 0, -1) for f in displacement_fields], axis=0
        )
        self.last_preprocessing_metadata = {
            "fixed_index": self.fixed_index,
            "lambda_reg": self.lambda_reg,
            "gamma_seg": self.gamma_seg,
            "target_spacing": self.target_spacing,
            "target_size": self.target_size,
            "crop_center_physical": tuple(float(v) for v in crop_center),
            "num_frames": num_frames,
        }
        return registered_stack
