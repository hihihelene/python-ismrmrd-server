"""Group-Oriented Registration (GOREG) of 2D time series with Elastix."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try: 
    from Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        omit_first_frames,
    )
    from registration_context import RegistrationResult
except ImportError:
    from .Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        omit_first_frames,
    )
    from .registration_context import RegistrationResult


def get_respiratory_surrogate(series: sitk.Image) -> np.ndarray:
    """Return one total-intensity value per frame."""
    return sitk.GetArrayFromImage(series).sum(axis=(1, 2))


def _to_float32_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.float32, copy=False)


def _require_2d(image: sitk.Image, name: str) -> None:
    if image.GetDimension() != 2:
        raise ValueError(f"{name} must be 2D.")


def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def _empty_deformation_field(reference: sitk.Image) -> sitk.Image:
    field = sitk.Image(reference.GetSize(), sitk.sitkVectorFloat64, 2)
    field.CopyInformation(reference)
    return field


def register_2d_pair(
    fixed: sitk.Image,
    moving: sitk.Image,
    parameter_file: str,
) -> list[sitk.ParameterMap]:
    """Register ``moving`` to ``fixed`` with Elastix.

    Returns the complete multi-stage Elastix transformation. The image result
    lives on the fixed grid; the transformation itself maps fixed-space points
    to moving-space coordinates for resampling.
    """
    _require_2d(fixed, "fixed")
    _require_2d(moving, "moving")

    elastix = sitk.ElastixImageFilter()
    elastix.LogToFileOff()
    elastix.LogToConsoleOff()

    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    elastix.SetParameterMap(sitk.ReadParameterFile(parameter_file))
    elastix.Execute()

    transform_maps = list(elastix.GetTransformParameterMap())
    if not transform_maps:
        raise RuntimeError(
            "Elastix completed but returned no transform parameter maps."
        )
    return transform_maps


def _write_composite_transform(
    registration_chains: list[list[sitk.ParameterMap]],
    output_directory: Path,
) -> str:
    """Write all stages as one hierarchy of Transformix parameter files.

    ``registration_chains`` follows the spatial path ``moving -> ... ->
    reference`` and must contain only non-empty chains. Transformix evaluates
    points from the reference grid back to the moving image, so the outermost
    transform must belong to the first registration (A in ``A(B(C(x)))``).
    Each stage therefore names the stage written before it as its
    ``InitialTransformParameterFileName``: the registrations are written in
    reverse order, the stages within one registration in their original order.

    Returns the path of the outermost parameter file.
    """
    previous_file = None

    for chain_index in reversed(range(len(registration_chains))):
        for stage_index, stage in enumerate(registration_chains[chain_index]):
            path = (
                output_directory / f"registration_{chain_index}_stage_{stage_index}.txt"
            )

            stage = sitk.ParameterMap(stage)  # copy: keep the stored maps unchanged
            if previous_file is not None:
                stage["InitialTransformParameterFileName"] = [previous_file]
            sitk.WriteParameterFile(stage, str(path))

            previous_file = str(path)

    return previous_file


def apply_composite_transforms(
    moving: sitk.Image,
    registration_chains: list[list[sitk.ParameterMap]],
    reference: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    """Apply a complete GOREG transformation to ``moving``.

    ``registration_chains`` describes the spatial path
    ``moving -> representative -> ... -> reference``. The image is resampled
    only once, with Transformix.

    Returns
    -------
    registered_image:
        Moving image represented on the reference grid.
    deformation_field:
        Transformix deformation field represented on the reference grid.
    """
    _require_2d(moving, "moving")
    _require_2d(reference, "reference")

    chains = [chain for chain in registration_chains if chain]

    if not chains:
        registered = sitk.Resample(
            moving,
            reference,
            sitk.Transform(2, sitk.sitkIdentity),
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        return registered, _empty_deformation_field(reference)

    with tempfile.TemporaryDirectory(prefix="goreg_") as temp_dir:
        transform_file = _write_composite_transform(chains, Path(temp_dir))

        transformix = sitk.TransformixImageFilter()
        transformix.LogToFileOff()
        transformix.LogToConsoleOff()

        transformix.SetMovingImage(moving)
        transformix.SetTransformParameterMap(sitk.ReadParameterFile(transform_file))

        transformix.ComputeDeformationFieldOn()
        transformix.Execute()

        registered = transformix.GetResultImage()
        deformation_field = transformix.GetDeformationField()

    if not _same_geometry(registered, reference):
        raise RuntimeError("Transformix result does not match the reference geometry.")

    if not _same_geometry(deformation_field, reference):
        raise RuntimeError(
            "Transformix deformation field does not match the reference geometry."
        )

    if deformation_field.GetNumberOfComponentsPerPixel() != 2:
        raise RuntimeError("Expected a 2-component 2D deformation field.")

    return registered, deformation_field


def deformation_fields_to_numpy(fields: list[sitk.Image]) -> np.ndarray:
    """Convert 2D deformation fields to shape (T, Y, X, 2)."""
    if not fields:
        return np.empty((0, 0, 0, 2), dtype=np.float32)

    arrays = []
    for index, field in enumerate(fields):
        if field is None:
            raise ValueError(f"Missing deformation field at frame {index}.")
        if field.GetDimension() != 2:
            raise ValueError(f"Deformation field at frame {index} must be 2D.")
        if field.GetNumberOfComponentsPerPixel() != 2:
            raise ValueError(
                f"Deformation field at frame {index} must have 2 components."
            )
        arrays.append(_to_float32_array(field))

    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Deformation-field shapes differ between frames.")

    return np.stack(arrays, axis=0)


def _build_groups(
    surrogate: np.ndarray,
    num_groups: int,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Assign frames to respiratory phase groups using quantile bins."""
    effective_num_groups = min(num_groups, len(surrogate))

    bins = np.quantile(surrogate, np.linspace(0.0, 1.0, effective_num_groups + 1))
    assignments = np.digitize(surrogate, bins, right=False) - 1
    assignments = np.clip(assignments, 0, effective_num_groups - 1).astype(int)

    groups = {
        group_id: np.flatnonzero(assignments == group_id).tolist()
        for group_id in range(effective_num_groups)
    }
    return groups, assignments


def _select_representative(frames: list[sitk.Image], indices: list[int]) -> int:
    """Return the frame closest to the mean image of a group."""
    arrays = np.stack([_to_float32_array(frames[index]) for index in indices])
    group_mean = arrays.mean(axis=0)
    mse = np.mean((arrays - group_mean) ** 2, axis=(1, 2))
    return indices[int(np.argmin(mse))]


def _register_groups(
    frames: list[sitk.Image],
    groups: dict[int, list[int]],
    parameter_file: str,
) -> tuple[dict[int, sitk.Image], dict[int, list[sitk.ParameterMap]]]:
    """Register every frame to its group representative.

    Returns
    -------
    group_averages:
        Noise-reduced average image of each non-empty group.
    intra_transforms:
        Transform chain ``frame -> representative`` for every frame.
    """
    group_averages = {}
    intra_transforms = {}

    for group_id, indices in groups.items():
        if not indices:
            continue

        representative_idx = _select_representative(frames, indices)
        representative = frames[representative_idx]
        registered_arrays = []

        for frame_idx in indices:
            if frame_idx == representative_idx:
                intra_transforms[frame_idx] = []
                registered_arrays.append(_to_float32_array(representative))
                continue

            moving = frames[frame_idx]
            transforms = register_2d_pair(
                fixed=representative,
                moving=moving,
                parameter_file=parameter_file,
            )
            intra_transforms[frame_idx] = transforms

            registered, _ = apply_composite_transforms(
                moving=moving,
                registration_chains=[transforms],
                reference=representative,
            )
            registered_arrays.append(_to_float32_array(registered))

        group_average = sitk.GetImageFromArray(np.mean(registered_arrays, axis=0))
        group_average.CopyInformation(representative)
        group_averages[group_id] = group_average

    return group_averages, intra_transforms


def _register_neighboring_groups(
    group_averages: dict[int, sitk.Image],
    target_group: int,
    parameter_file: str,
) -> dict[tuple[int, int], list[sitk.ParameterMap]]:
    """Register neighboring group averages towards the target group.

    Each key ``(source_group, target_group)`` describes one step of the spatial
    path towards the target group (``0 -> 1 -> ... -> target <- ... <- N``),
    so the chains can be appended to a frame's chain in exactly that order.
    """
    groups = sorted(group_averages)
    neighbors = list(zip(groups, groups[1:]))

    steps_up = [(lower, upper) for lower, upper in neighbors if lower < target_group]
    steps_down = [
        (upper, lower) for lower, upper in reversed(neighbors) if upper > target_group
    ]

    return {
        (source, target): register_2d_pair(
            fixed=group_averages[target],
            moving=group_averages[source],
            parameter_file=parameter_file,
        )
        for source, target in steps_up + steps_down
    }


def _build_frame_transform_chain(
    frame_idx: int,
    group_assignments: np.ndarray,
    intra_transforms: dict[int, list[sitk.ParameterMap]],
    inter_transforms: dict[tuple[int, int], list[sitk.ParameterMap]],
    target_group: int,
) -> list[list[sitk.ParameterMap]]:
    """Build the ordered GOREG registration chain for one frame.

    The order is: frame -> representative -> neighboring group -> ... ->
    target group.
    """
    current_group = int(group_assignments[frame_idx])
    chains = []

    intra = intra_transforms.get(frame_idx, [])
    if intra:
        chains.append(intra)

    while current_group != target_group:
        step = 1 if current_group < target_group else -1
        next_group = current_group + step
        key = (current_group, next_group)

        if key not in inter_transforms:
            raise RuntimeError(
                "Incomplete GOREG transform chain: "
                f"missing transformation {current_group} -> {next_group}."
            )

        chains.append(inter_transforms[key])
        current_group = next_group

    return chains


def image_series_goreg_registration(
    moving_series: sitk.Image,
    parameter_file_path: str,
    num_groups: int = 6,
    skip_first: int = 0,
) -> RegistrationResult:
    """Perform Group-Oriented Registration (GOREG) on a 3D time series.

    ``moving_series`` is expected to have its warm-up frames removed already.
    ``skip_first`` only exists for callers that have not done so; leave it at 0
    otherwise, or the frames are removed twice. Returned frame indices refer to
    the series after ``skip_first`` frames have been removed.
    """
    if moving_series.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D time series, got a {moving_series.GetDimension()}D image."
        )

    original_num_frames = moving_series.GetSize()[2]

    if original_num_frames == 0:
        raise ValueError("Moving series contains no frames.")

    if num_groups <= 0:
        raise ValueError("num_groups must be > 0.")

    if skip_first < 0:
        raise ValueError("skip_first must be >= 0.")

    if skip_first >= original_num_frames:
        raise ValueError(
            f"skip_first={skip_first} would remove all {original_num_frames} frames."
        )

    if skip_first:
        moving_series = omit_first_frames(moving_series, skip_first)

    num_frames = moving_series.GetSize()[2]
    frames = [extract_2d_slice(moving_series, index) for index in range(num_frames)]

    reference_idx = find_middle_intensity_slice(moving_series)
    reference_image = frames[reference_idx]

    groups, group_assignments = _build_groups(
        surrogate=get_respiratory_surrogate(moving_series),
        num_groups=num_groups,
    )
    target_group = int(group_assignments[reference_idx])

    group_averages, intra_transforms = _register_groups(
        frames=frames,
        groups=groups,
        parameter_file=parameter_file_path,
    )
    inter_transforms = _register_neighboring_groups(
        group_averages=group_averages,
        target_group=target_group,
        parameter_file=parameter_file_path,
    )

    registered_arrays = []
    deformation_fields = []

    for frame_idx, frame in enumerate(frames):
        registration_chain = _build_frame_transform_chain(
            frame_idx=frame_idx,
            group_assignments=group_assignments,
            intra_transforms=intra_transforms,
            inter_transforms=inter_transforms,
            target_group=target_group,
        )
        registered, deformation_field = apply_composite_transforms(
            moving=frame,
            registration_chains=registration_chain,
            reference=reference_image,
        )
        registered_arrays.append(_to_float32_array(registered))
        deformation_fields.append(deformation_field)

    registered_series = sitk.GetImageFromArray(np.stack(registered_arrays, axis=0))
    registered_series.CopyInformation(moving_series)

    return RegistrationResult(
        registered_series=registered_series,
        reference_image=_to_float32_array(reference_image),
        reference_index=reference_idx,
        fixed_lung_mask=None,
        registered_lung_mask=None,
        disp_fields=deformation_fields_to_numpy(deformation_fields),
        transform_parameter_maps={
            "intra": intra_transforms,
            "inter": inter_transforms,
        },
    )
