"""B-spline registration of a 2D time series (Elastix BSplineStackTransform)."""

import os


import numpy as np
import SimpleITK as sitk

try:
    from .Reading_and_Writing import read_images_from_folder
except ImportError:
    from Reading_and_Writing import read_images_from_folder


def find_middle_intensity_slice(series: sitk.Image) -> int:
    """Select the frame whose total intensity is closest to the median.

    The frame intensity is defined as the sum of all pixel intensities
    in that frame. The reference frame is the frame whose total
    intensity is closest to the median frame intensity.
    """
    image_arrays = sitk.GetArrayFromImage(series)
    if image_arrays.ndim != 3:
        raise ValueError(
            f"Expected a 3D image series (T, Y, X), got shape {image_arrays.shape}."
        )
    frame_intensities = np.sum(image_arrays, axis=(1, 2), dtype=np.float64)
    median_intensity = np.median(frame_intensities)
    reference_index = int(np.argmin(np.abs(frame_intensities - median_intensity)))
    return reference_index



def omit_first_frames(series3d: sitk.Image, k: int) -> sitk.Image:
    """Return series3d with the first k frames removed along axis 2.

    The origin is updated to the new first frame.
    """
    width, height, depth = series3d.GetSize()
    k = max(0, min(int(k), depth))
    if depth - k == 0:
        raise ValueError(
            f"omit_first_frames would produce empty stack (k={k}, depth={depth})."
        )
    return sitk.Extract(series3d, [width, height, depth - k], [0, 0, k])



def extract_2d_slice(volume3d: sitk.Image, slice_index: int) -> sitk.Image:
    """Extract a single 2D frame from a 3D (x,y,z/time) image."""
    width, height, _ = volume3d.GetSize()
    return sitk.Extract(
        volume3d,
        [width, height, 0],
        [0, 0, slice_index],
        sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOIDENTITY,
    )


def _repeat_along_time(array2d: np.ndarray, template3d: sitk.Image) -> sitk.Image:
    """Tile a 2D array along the 3rd axis and copy the geometry of the template."""
    depth = template3d.GetSize()[2]
    stack = sitk.GetImageFromArray(np.repeat(array2d[np.newaxis, ...], depth, axis=0))
    stack.CopyInformation(template3d)
    return stack



def make_fixed_stack(fixed2d: sitk.Image, template3d: sitk.Image) -> sitk.Image:
    """Tile a 2D fixed image across the 3rd axis.

    Spacing, origin and direction (i.e. the orientation) are copied from
    ``template3d`` so the stack matches the moving series exactly.
    """
    stack = _repeat_along_time(sitk.GetArrayFromImage(fixed2d), template3d)
    return sitk.Cast(stack, fixed2d.GetPixelID())


def make_mask_stack(mask2d: sitk.Image, template3d: sitk.Image) -> sitk.Image:
    """Repeat a 2D binary mask across the time axis and copy the template geometry."""
    return _repeat_along_time(
        sitk.GetArrayFromImage(mask2d).astype(np.uint8), template3d
    )


def estimate_stack_transform(
    moving_stack_3d: sitk.Image,
    fixed_stack_3d: sitk.Image,
    parameter_file_path: str,
    fixed_mask_3d: sitk.Image | None = None,
    moving_mask_3d: sitk.Image | None = None,
):
    """Single optimization with BSplineStackTransform -> one 2D B-spline per frame.

    Only dataset-dependent parameters are injected into the parameter file:
    NumberOfSubTransforms, StackSpacing and StackOrigin.

    Returns the Elastix transform parameter map(s).
    """
    elastix = sitk.ElastixImageFilter()
    elastix.LogToFileOff()
    elastix.LogToConsoleOff()

    elastix.SetFixedImage(fixed_stack_3d)
    elastix.SetMovingImage(moving_stack_3d)

    if fixed_mask_3d is not None:
        elastix.SetFixedMask(fixed_mask_3d)
    if moving_mask_3d is not None:
        elastix.SetMovingMask(moving_mask_3d)

    parameter_map = sitk.ReadParameterFile(parameter_file_path)

    num_frames = moving_stack_3d.GetSize()[2]
    stack_spacing = moving_stack_3d.GetSpacing()[2]
    stack_origin = moving_stack_3d.GetOrigin()[2]
    if stack_spacing <= 0:
        stack_spacing = 1.0

    parameter_map["NumberOfSubTransforms"] = [str(num_frames)]
    parameter_map["StackSpacing"] = [str(stack_spacing)]
    parameter_map["StackOrigin"] = [str(stack_origin)]

    elastix.SetParameterMap(parameter_map)
    # elastix.SetOutputDirectory(output_dir)

    elastix.Execute()

    return elastix.GetTransformParameterMap()


def _to_in_plane_field(deformation_field: sitk.Image) -> sitk.Image:
    """Reduce a 3-component stack deformation field to its (x, y) components.

    Every frame is deformed by its own 2D B-spline, so the z component (the
    time axis) carries no displacement. Downstream metrics expect 2-component
    fields, i.e. ``(T, Y, X, 2)`` once converted to numpy.
    """
    x_component = sitk.VectorIndexSelectionCast(deformation_field, 0)
    y_component = sitk.VectorIndexSelectionCast(deformation_field, 1)
    return sitk.Compose(x_component, y_component)


def apply_stack_transform(
    moving_stack_3d: sitk.Image,
    transform_parameter_map,
    reference_stack_3d: sitk.Image,
    # output_dir: str
):
    """Apply the stack transform once to the whole 3D time series and write outputs."""
    output_dir = "temp"
    os.makedirs(output_dir, exist_ok=True)

    tfx = sitk.TransformixImageFilter()
    tfx.LogToFileOff()
    tfx.LogToConsoleOff()
    tfx.SetTransformParameterMap(transform_parameter_map)
    tfx.SetMovingImage(moving_stack_3d)
    tfx.SetOutputDirectory(output_dir)
    # Request deformation field in addition to the transformed image.
    tfx.ComputeDeformationFieldOn()
    tfx.Execute()
    registered_stack = tfx.GetResultImage()
    registered_stack.CopyInformation(reference_stack_3d)
    deformation_field = tfx.GetDeformationField()
    return registered_stack, deformation_field


def image_series_registration(
    moving_series: sitk.Image,
    fixed_image_2d: sitk.Image,
    parameter_file_path: str,
):
    """Groupwise, per-frame transforms via BSplineStackTransform."""
    # joint_dir = os.path.join(output_dir, "stack_elastix")
    # apply_dir = os.path.join(output_dir, "stack_transformix")
    # os.makedirs(joint_dir, exist_ok=True)
    original_direction = moving_series.GetDirection()

    # Change moving series orientation for VarianceOverLastDimensionMetric
    moving_series.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # Build 3D fixed stack matching the 3D moving series
    fixed_stack = make_fixed_stack(fixed2d=fixed_image_2d, template3d=moving_series)

    # One optimization -> per-slice (time) 2D B-splines
    transform_parameter_map = estimate_stack_transform(
        moving_series, fixed_stack, parameter_file_path
    )
    # Apply once to the whole stack
    registered_series, deformation_field = apply_stack_transform(
        moving_series, transform_parameter_map, fixed_stack
    )

    # Restore the original geometry representation of the input stack.
    registered_series.SetDirection(original_direction)

    return (
        fixed_stack,
        moving_series,
        registered_series,
        transform_parameter_map,
        deformation_field,
    )


def main():
    main_directory = "C:\\Lung_Project\\Measurements\\CDH_Study\\20250729_age13_1"
    parameter_file = "registration_parameter_file.txt"
    comparison_directory = os.path.join(
        "Results_Registration", "comparison_with_and_without_mask"
    )
    os.makedirs(comparison_directory, exist_ok=True)

    # Read series (expects a 3D image: x,y, time)
    moving_series = read_images_from_folder(main_directory)
    print("shape of moving series:", moving_series.GetSize())

    skip_first = 8
    if skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)
        print(f"trimmed moving series (skipped {skip_first}):", moving_series.GetSize())

    # Pick fixed frame = median-intensity slice
    fixed_image_index = find_middle_intensity_slice(moving_series)
    print("Fixed image index:", fixed_image_index)
    fixed_image = extract_2d_slice(moving_series, fixed_image_index)
    print("shape of fixed image:", fixed_image.GetSize())

    # Groupwise registration without mask
    _fixed_stack_no_mask, moving_series_no_mask, _applied_stack_no_mask, tsp, disp = (
        image_series_registration(moving_series, fixed_image, parameter_file)
    )
    print("shape of moving series (no mask):", moving_series_no_mask.GetSize())

    print("shape of moving series (no mask):", moving_series_no_mask.GetSize())


if __name__ == "__main__":
    main()
