try:
    from .Reading_and_Writing import read_images_from_folder
except ImportError:
    from Reading_and_Writing import read_images_from_folder
import SimpleITK as sitk
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def find_middle_intensity_slice(series: sitk.Image) -> int:
    """Choose the fixed frame as the slice with median total intensity."""
    image_arrays = sitk.GetArrayFromImage(series)  # (z, y, x)
    slice_intensities = [np.sum(image) for image in image_arrays]
    median_index = np.argsort(slice_intensities)[len(slice_intensities) // 2]
    return int(median_index)

def omit_first_frames(series3d: sitk.Image, k: int) -> sitk.Image:
    """Return series3d with the first k frames removed along axis 2."""
    sz = list(series3d.GetSize())  # [x, y, z]
    k = max(0, min(int(k), sz[2]))
    start = [0, 0, k]
    size  = [sz[0], sz[1], sz[2] - k]
    if size[2] == 0:
        raise ValueError(f"omit_first_frames would produce empty stack (k={k}, depth={sz[2]}).")
    ex = sitk.ExtractImageFilter()
    ex.SetIndex(start)
    ex.SetSize(size)
    return ex.Execute(series3d)  # origin is updated automatically to the new start

def extract_2d_slice(volume3d: sitk.Image, slice_index: int) -> sitk.Image:
    """Extract a single 2D frame from a 3D (x,y,z/time) image."""
    size = list(volume3d.GetSize())      # [x, y, z]
    start = [0, 0, slice_index]
    size[2] = 0
    ex = sitk.ExtractImageFilter()
    ex.SetSize(size)
    ex.SetIndex(start)
    ex.SetDirectionCollapseToStrategy(
        sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOIDENTITY
    )
    return ex.Execute(volume3d)

def make_fixed_stack(fixed2d: sitk.Image, template3d: sitk.Image) -> sitk.Image:
    """
    Tile a 2D fixed image across the 3rd axis, but copy full geometry from template3d
    so orientation (coronal/axial/sagittal), spacing, origin, and direction all match.
    """
    depth = template3d.GetSize()[2]

    # 2D -> 3D numpy (z,y,x) by repeating along the stack axis (z)
    arr2d = sitk.GetArrayFromImage(fixed2d)          # (y, x)
    arr3d = np.repeat(arr2d[np.newaxis, ...], depth, axis=0)  # (z, y, x)

    stack = sitk.GetImageFromArray(arr3d)            # creates 3D image with identity geometry
    stack.CopyInformation(template3d)                # now geometry matches the moving series exactly
    return sitk.Cast(stack, fixed2d.GetPixelID())


def make_mask_stack(mask2d: sitk.Image, template3d: sitk.Image) -> sitk.Image:
    """Repeat a 2D binary mask across the time axis and copy the template geometry."""
    depth = template3d.GetSize()[2]
    arr2d = sitk.GetArrayFromImage(mask2d).astype(np.uint8)
    arr3d = np.repeat(arr2d[np.newaxis, ...], depth, axis=0)

    stack = sitk.GetImageFromArray(arr3d)
    stack.CopyInformation(template3d)
    return sitk.Cast(stack, sitk.sitkUInt8)


def estimate_stack_transform(
    moving_stack_3d: sitk.Image,
    fixed_stack_3d: sitk.Image,
    parameter_file_path: str,
    fixed_mask_3d: sitk.Image | None = None,
    moving_mask_3d: sitk.Image | None = None,
    # output_dir: str
):
    """
    Single optimization with BSplineStackTransform -> one 2D B-spline per time frame.
    Only dataset-dependent params are injected: NumberOfSubTransforms, StackSpacing, StackOrigin.
    """
    # os.makedirs(output_dir, exist_ok=True)

    elastix = sitk.ElastixImageFilter()
    elastix.LogToFileOff()
    elastix.LogToConsoleOff()

    elastix.SetFixedImage(fixed_stack_3d)
    elastix.SetMovingImage(moving_stack_3d)

    if fixed_mask_3d is not None:
        elastix.SetFixedMask(fixed_mask_3d)
    if moving_mask_3d is not None:
        elastix.SetMovingMask(moving_mask_3d)

    pm = sitk.ReadParameterFile(parameter_file_path)

    # ----- Inject ONLY data-dependent entries (keep all other params in the file) -----
    num_frames    = moving_stack_3d.GetSize()[2]
    stack_spacing = moving_stack_3d.GetSpacing()[2] if moving_stack_3d.GetDimension() == 3 else 1.0
    stack_origin = moving_stack_3d.GetOrigin()[2] if moving_stack_3d.GetDimension() == 3 else 0.0
    if stack_spacing <= 0:
        stack_spacing = 1.0

    pm["NumberOfSubTransforms"] = [str(num_frames)]
    pm["StackSpacing"]          = [str(stack_spacing)]
    pm["StackOrigin"]           = [str(stack_origin)]

    elastix.SetParameterMap(pm)
    # elastix.SetOutputDirectory(output_dir)

    elastix.Execute()

    # Persist the transform(s)
    tpm = elastix.GetTransformParameterMap()
    # if isinstance(tpm, sitk.ParameterMap):
        # sitk.WriteParameterFile(tpm, os.path.join(output_dir, "TransformParameters.0.txt"))
    # else:
        # for i, pm_i in enumerate(tpm):
            # sitk.WriteParameterFile(pm_i, os.path.join(output_dir, f"TransformParameters.{i}.txt"))

    return tpm


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
    tfx.Execute()
    
    registered_stack = tfx.GetResultImage()
    registered_stack.CopyInformation(reference_stack_3d)

    return registered_stack


def image_series_registration(
    moving_series: sitk.Image,
    fixed_image_2d: sitk.Image,
    parameter_file_path: str,
):
    """Groupwise, per-frame transforms via BSplineStackTransform."""
    # joint_dir = os.path.join(output_dir, "stack_elastix")
    # apply_dir = os.path.join(output_dir, "stack_transformix")
    # os.makedirs(joint_dir, exist_ok=True)
    
    # Change moving series orientation for VarianceOverLastDimensionMetric
    moving_series.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # Build 3D fixed stack matching the 3D moving series
    fixed_stack = make_fixed_stack(fixed2d=fixed_image_2d, template3d=moving_series)

    # One optimization -> per-slice (time) 2D B-splines
    tpm = estimate_stack_transform(
        moving_series,
        fixed_stack,
        parameter_file_path
    )

    # Apply once to the whole stack
    applied_stack = apply_stack_transform(moving_series, tpm, fixed_stack)

    return fixed_stack, moving_series, applied_stack


def main():
    # Directory containing DICOM files
    main_directory = 'C:\\Lung_Project\\Measurements\\CDH_Study\\20250729_age13_1'

    output_directory = 'Results_Registration'
    comparison_directory = os.path.join(output_directory, 'comparison_with_and_without_mask')
    parameter_file = 'registration_parameter_file.txt'  # see file contents below
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(comparison_directory, exist_ok=True)

    # Read series (expects a 3D image: x,y, time)
    moving_series = read_images_from_folder(main_directory)
    print('shape of moving series:',moving_series.GetSize())

    SKIP_FIRST = 8   # set to any integer, e.g. 5–10
    if SKIP_FIRST > 0:
        moving_series = omit_first_frames(moving_series, SKIP_FIRST)
        print(f'trimmed moving series (skipped {SKIP_FIRST}):', moving_series.GetSize())

    # Pick fixed frame = median-intensity slice
    fixed_image_index = find_middle_intensity_slice(moving_series)
    print("Fixed image index:", fixed_image_index)
    fixed_image = extract_2d_slice(moving_series, int(fixed_image_index))
    print('shape of fixed image:',fixed_image.GetSize())

    # Groupwise registration without mask
    fixed_stack_no_mask, moving_series_no_mask, applied_stack_no_mask = image_series_registration(
        moving_series, fixed_image, parameter_file
    )
    print('shape of moving series (no mask):', moving_series_no_mask.GetSize())

if __name__ == "__main__":
    main()

