import os
try:
    from .Reading_and_Writing import read_images_from_folder
except ImportError:
    from Reading_and_Writing import read_images_from_folder
import itk
import numpy as np


def _image_size(image):
    return tuple(int(value) for value in image.GetLargestPossibleRegion().GetSize())


def _copy_geometry(source, target):
    target.SetSpacing(source.GetSpacing())
    target.SetOrigin(source.GetOrigin())
    target.SetDirection(source.GetDirection())
    return target


def find_middle_intensity_slice(series) -> int:
    """Choose the fixed frame as the slice with median total intensity."""
    image_arrays = itk.array_from_image(series)  # (z, y, x)
    slice_intensities = [np.sum(image) for image in image_arrays]
    median_index = np.argsort(slice_intensities)[len(slice_intensities) // 2]
    return int(median_index)

def omit_first_frames(series3d, k: int):
    """Return series3d with the first k frames removed along axis 2."""
    sz = list(_image_size(series3d))  # [x, y, z]
    k = max(0, min(int(k), sz[2]))
    size = [sz[0], sz[1], sz[2] - k]
    if size[2] == 0:
        raise ValueError(f"omit_first_frames would produce empty stack (k={k}, depth={sz[2]}).")
    result = itk.image_from_array(itk.array_from_image(series3d)[k:, :, :])
    return _copy_geometry(series3d, result)

def extract_2d_slice(volume3d, slice_index: int):
    """Extract a single 2D frame from a 3D (x,y,z/time) image."""
    image = itk.image_from_array(itk.array_from_image(volume3d)[slice_index, :, :])
    image.SetSpacing(tuple(volume3d.GetSpacing()[index] for index in range(2)))
    image.SetOrigin(tuple(volume3d.GetOrigin()[index] for index in range(2)))
    return image

def make_fixed_stack(fixed2d, template3d):
    """
    Tile a 2D fixed image across the 3rd axis, but copy full geometry from template3d
    so orientation (coronal/axial/sagittal), spacing, origin, and direction all match.
    """
    depth = _image_size(template3d)[2]

    # 2D -> 3D numpy (z,y,x) by repeating along the stack axis (z)
    arr2d = itk.array_from_image(fixed2d)          # (y, x)
    arr3d = np.repeat(arr2d[np.newaxis, ...], depth, axis=0)  # (z, y, x)

    stack = itk.image_from_array(arr3d)            # creates 3D image with identity geometry
    return _copy_geometry(template3d, stack)


def make_mask_stack(mask2d, template3d):
    """Repeat a 2D binary mask across the time axis and copy the template geometry."""
    depth = _image_size(template3d)[2]
    arr2d = itk.array_from_image(mask2d).astype(np.uint8)
    arr3d = np.repeat(arr2d[np.newaxis, ...], depth, axis=0)

    stack = itk.image_from_array(arr3d)
    return _copy_geometry(template3d, stack)


def estimate_stack_transform(
    moving_stack_3d,
    fixed_stack_3d,
    parameter_file_path: str,
    fixed_mask_3d=None,
    moving_mask_3d=None,
    # output_dir: str
):
    """
    Single optimization with BSplineStackTransform -> one 2D B-spline per time frame.
    Only dataset-dependent params are injected: NumberOfSubTransforms, StackSpacing, StackOrigin.
    """
    # os.makedirs(output_dir, exist_ok=True)

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(parameter_file_path)
    pm = parameter_object.GetParameterMap(0)

    # ----- Inject ONLY data-dependent entries (keep all other params in the file) -----
    num_frames    = _image_size(moving_stack_3d)[2]
    is_3d = len(_image_size(moving_stack_3d)) == 3
    stack_spacing = moving_stack_3d.GetSpacing()[2] if is_3d else 1.0
    stack_origin = moving_stack_3d.GetOrigin()[2] if is_3d else 0.0
    if stack_spacing <= 0:
        stack_spacing = 1.0

    pm["NumberOfSubTransforms"] = [str(num_frames)]
    pm["StackSpacing"]          = [str(stack_spacing)]
    pm["StackOrigin"]           = [str(stack_origin)]

    parameter_object.SetParameterMap(pm)

    elastix = itk.ElastixRegistrationMethod.New(
        fixed_image=fixed_stack_3d,
        moving_image=moving_stack_3d,
        parameter_object=parameter_object,
    )
    if fixed_mask_3d is not None:
        elastix.SetFixedMask(fixed_mask_3d)
    if moving_mask_3d is not None:
        elastix.SetMovingMask(moving_mask_3d)
    elastix.SetLogToConsole(False)
    elastix.Update()

    return elastix.GetTransformParameterObject()


def apply_stack_transform(
    moving_stack_3d,
    transform_parameter_map,
    reference_stack_3d,
    # output_dir: str
):
    """Apply the stack transform once to the whole 3D time series and write outputs."""
    transformix = itk.TransformixFilter.New(
        moving_image=moving_stack_3d,
        transform_parameter_object=transform_parameter_map,
    )
    transformix.SetLogToConsole(False)
    transformix.Update()

    return _copy_geometry(reference_stack_3d, transformix.GetOutput())


def image_series_registration(
    moving_series,
    fixed_image_2d,
    parameter_file_path: str,
):
    """Groupwise, per-frame transforms via BSplineStackTransform."""
    # joint_dir = os.path.join(output_dir, "stack_elastix")
    # apply_dir = os.path.join(output_dir, "stack_transformix")
    # os.makedirs(joint_dir, exist_ok=True)
    
    # Change moving series orientation for VarianceOverLastDimensionMetric
    moving_series.SetDirection(itk.matrix_from_array(np.eye(3)))

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
    print('shape of moving series:', _image_size(moving_series))

    SKIP_FIRST = 8   # set to any integer, e.g. 5–10
    if SKIP_FIRST > 0:
        moving_series = omit_first_frames(moving_series, SKIP_FIRST)
        print(f'trimmed moving series (skipped {SKIP_FIRST}):', _image_size(moving_series))

    # Pick fixed frame = median-intensity slice
    fixed_image_index = find_middle_intensity_slice(moving_series)
    print("Fixed image index:", fixed_image_index)
    fixed_image = extract_2d_slice(moving_series, int(fixed_image_index))
    print('shape of fixed image:', _image_size(fixed_image))

    # Groupwise registration without mask
    fixed_stack_no_mask, moving_series_no_mask, applied_stack_no_mask = image_series_registration(
        moving_series, fixed_image, parameter_file
    )
    print('shape of moving series (no mask):', _image_size(moving_series_no_mask))

if __name__ == "__main__":
    main()

