"""nnU-Net based lung segmentation helper."""

from __future__ import annotations

import contextlib
import io
import os
import cv2
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

DEFAULT_MODEL_FOLDER = Path(__file__).resolve().parent / "models" / "nnunet_model_v001"
DEFAULT_CHECKPOINT_NAME = "checkpoint_best.pth"


def _resolve_model_folder(model_folder: Optional[Union[str, Path]]) -> Path:
    resolved = Path(model_folder) if model_folder is not None else DEFAULT_MODEL_FOLDER
    if not resolved.exists():
        raise FileNotFoundError(f"nnU-Net model folder not found: {resolved}")
    return resolved


def _image_to_nnunet_input(
    image: Union[np.ndarray, sitk.Image],
    spacing: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, dict]:
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        if spacing is None:
            spacing = tuple(float(value) for value in image.GetSpacing()[::-1])
    else:
        image_array = np.asarray(image, dtype=np.float32)

    if image_array.ndim == 2:
        image_array = image_array[None, None, ...]
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        else:
            spacing = (1.0, *tuple(float(value) for value in spacing))
    elif image_array.ndim != 3:
        raise ValueError(f"Expected a 2D image or a single-channel 3D array, got shape {image_array.shape}.")

    if spacing is None:
        spacing = tuple(1.0 for _ in range(image_array.ndim))

    image_properties = {"spacing": tuple(float(value) for value in spacing)}
    return image_array, image_properties

def select_largest_connected_components(segmentation: np.ndarray, k: int = 2) -> np.ndarray:
    # find contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # How many contours were found
    print(f"Found {len(contours)} contours in the segmentation mask.")

    # sort contours by area and select the k largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:k]


    # create a mask for the selected contours
    k_largest_components = np.zeros_like(segmentation, dtype=np.uint8)
    for contour in contours:
        cv2.fillPoly(k_largest_components, [contour], 1)

    # convert to boolean mask
    k_largest_components = k_largest_components.astype(bool)
    return k_largest_components

def segment_nnunet(
    image: Union[np.ndarray, sitk.Image],
    model_folder: Optional[Union[str, Path]] = None,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    use_folds: Tuple[Union[int, str], ...] = (0,),
    device: str = "cpu",
    spacing: Optional[Sequence[float]] = None,
    **kwargs,
) -> np.ndarray:
    """Run nnU-Net inference on a single image and return a binary mask.

    The helper intentionally keeps imports local so the rest of the pipeline can still
    import even when nnU-Net is not available in the active Python environment.
    """

    resolved_model_folder = _resolve_model_folder(model_folder)
    os.environ.setdefault("nnUNet_raw", str(resolved_model_folder.parent))
    os.environ.setdefault("nnUNet_preprocessed", str(resolved_model_folder.parent))
    os.environ.setdefault("nnUNet_results", str(resolved_model_folder.parent))


    import torch
    image_array, image_properties = _image_to_nnunet_input(image, spacing=spacing)

    startup_sink = io.StringIO()
    with contextlib.redirect_stdout(startup_sink), contextlib.redirect_stderr(startup_sink):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(resolved_model_folder),
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
        )

    segmentation = predictor.predict_single_npy_array(
        image_array,
        image_properties,
        None,
        None,
        False,
    )

    segmentation_array = np.asarray(segmentation)
    if segmentation_array.ndim == 3 and segmentation_array.shape[0] == 1:
        segmentation_array = np.squeeze(segmentation_array, axis=0)

    # Checking only for the two largest connected regions 
    # currently necessary due to nnUnet training (maybe irrelevant for better model in the future)
    
    segmentation_array = select_largest_connected_components(segmentation_array, k=2)
    
    return segmentation_array
