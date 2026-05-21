"""Main driver for ventilation and perfusion mapping.

This module was refactored to remove top-level execution, reduce redundancy,
and group operations into helper functions. Calling 'main()' runs the
full pipeline for a given 'series_indicator'.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

try: 
    from .Dynamic_Mode_Decomposition import (
        dynamic_mode_decomp,
        mask_images,
        mean_step_size,
        process_DMD_modes,
        create_rgb_overlay,
    )

    from .Fourier_Decomposition import (
        fourier_decomp,
        frequency_spectrum_plot,
    )

    from .Reading_and_Writing import (
        center_crop_last_dim,
        get_dicom_acquisition_times,
        read_images_from_folder,
        array_to_sitk,
        get_ismrmrd_acquisition_times,
    )

    from .Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        image_series_registration,
        omit_first_frames,
    )

    from .Segmentation import (
        augment_mask,
        connect_lungs_sitk,
        extract_body_mask,
        rough_lung_segmentation,
        manual_segmentation,
        napari_segmentation,
    )

    from .Plotting import (
        plot_overlays,
        plot_individual_modes,
        plot_results,
    )
except ImportError:
    print("Using absolute import paths instead of relative")
    from Dynamic_Mode_Decomposition import (
        dynamic_mode_decomp,
        mask_images,
        mean_step_size,
        process_DMD_modes,
        create_rgb_overlay,
    )

    from Fourier_Decomposition import (
        fourier_decomp,
        frequency_spectrum_plot,
    )

    from Reading_and_Writing import (
        center_crop_last_dim,
        get_dicom_acquisition_times,
        read_images_from_folder,
        array_to_sitk,
        get_ismrmrd_acquisition_times,
    )

    from Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        image_series_registration,
        omit_first_frames,
    )

    from Segmentation import (
        augment_mask,
        connect_lungs_sitk,
        extract_body_mask,
        rough_lung_segmentation,
        manual_segmentation,
        napari_segmentation,
    )

    from Plotting import (
        plot_overlays,
        plot_individual_modes,
        plot_results,
    )

def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def setup_paths(series_indicator, base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'Measurements'))

    input_path_registration = os.path.join(data_dir, series_indicator)
    output_path_registration = os.path.join(base_dir, 'Results', series_indicator)
    parameter_file_registration = os.path.join(base_dir, 'registration_parameter_file.txt')
    output_path = os.path.join(base_dir, 'Results', series_indicator, 'FD_DMD')

    ensure_dir(output_path_registration)
    ensure_dir(output_path)

    return {
        'input_path_registration': input_path_registration,
        'output_path_registration': output_path_registration,
        'parameter_file_registration': parameter_file_registration,
        'output_path': output_path,
    }


def run_registration(parameter_file_registration, moving_series, skip_first=8):
    """
    Perform image series registration.
    
    Args:
        parameter_file_registration: Path to registration parameter file
        moving_series: Pre-loaded SITK image series
        skip_first: Number of initial frames to omit
    
    Returns:
        arr3d: 3D registered array (z, y, x)
        vol2dt: Transposed array for 2D + time processing (y, x, z)
    """
    if skip_first and skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)

    fixed_index = find_middle_intensity_slice(moving_series)
    fixed_image = extract_2d_slice(moving_series, int(fixed_index))

    # Perform registration (no file output, returns in-memory result)
    fixed_stack, moving_series, applied_stack = image_series_registration(moving_series, fixed_image, parameter_file_registration)

    img3d = applied_stack
    # arr3d = center_crop_last_dim(sitk.GetArrayFromImage(img3d))
    arr3d = sitk.GetArrayFromImage(img3d)
    #compute mean for arr3d
    print('arr3d shape:', arr3d.shape)
    meanarr3d = np.mean(arr3d[0, :, :])
    print('mean', meanarr3d)
    vol2dt = arr3d.transpose(1, 2, 0)
    return arr3d, vol2dt


def compute_masks_and_mean(vol2dt, segmentation_method, plotting=False):
    mean2d_np = np.mean(vol2dt, axis=2).astype(np.float32)
    mean2d_sitk = sitk.GetImageFromArray(mean2d_np)

    body_mask = extract_body_mask(mean2d_sitk, lowerThreshold=0.25, radius=10)
    _, lung_init = rough_lung_segmentation(mean2d_sitk, body_mask, lung_lower_factor=0.0, lung_upper_factor=0.43)

    if segmentation_method =='manual':
        augmented_lung = manual_segmentation(mean2d_np, output_path = None, brush_size = 5)
        augmented_lung = sitk.GetImageFromArray(augmented_lung.astype(np.uint8))
        augmented_lung.CopyInformation(mean2d_sitk)
        
    elif segmentation_method == 'automatic':
        augmented_lung = augment_mask(mean2d_sitk, lung_init, body_mask, neighborhood_radius=1, num_iterations=15)

    elif segmentation_method == 'napari':
        napari_mask = napari_segmentation(mean2d_np)
        augmented_lung = sitk.GetImageFromArray(napari_mask.astype(np.uint8))
        augmented_lung.CopyInformation(mean2d_sitk)

    else:
        raise ValueError(f"Unsupported segmentation_method: {segmentation_method}")

    # full_thorax_mask = connect_lungs_sitk(augmented_lung, closing_radius=(90, 90, 30))

    body_np = sitk.GetArrayFromImage(body_mask).astype(bool)
    lung_init_np = sitk.GetArrayFromImage(lung_init).astype(bool)
    augmented_np = sitk.GetArrayFromImage(augmented_lung).astype(bool)
    # full_thorax_np = sitk.GetArrayFromImage(full_thorax_mask).astype(bool)
    full_thorax_np = augmented_np

    if plotting == True:
        plt.figure(figsize=(18, 6))
        masks = [
            (body_np, 'Body Mask', 'g', '--'),
            (lung_init_np, 'Initial Lung Mask', 'b', '-.'),
            (augmented_np, 'Augmented Lung Mask', 'r', '-'),
            (full_thorax_np, 'Lung and Heart', 'b', '-.'),
        ]

        for i, (mask, title, color, ls) in enumerate(masks, 1):
            ax = plt.subplot(1, len(masks), i)
            ax.imshow(mean2d_np, cmap='gray', interpolation='nearest')
            ax.contour(mask, levels=[0.5], colors=color, linestyles=ls, linewidths=2)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show(block=False)

    return mean2d_np, full_thorax_np


def run_fourier(vol2dt, mask2d, time_step, output_path, mean2d_np, phantom=False, plotting = False):
    # Run Fourier decomposition first so spectrum plotting can reuse detected peaks
    Im1, Im2, Im0, V1, V2, vent_hz, perf_hz = fourier_decomp(
        vol2dt, dt=time_step, bw=mask2d, prominence=0.3, phantom=phantom
    )
    if plotting == True:
        # Show spectrum; when phantom=True, skip perfusion detection in the plot
        if phantom:
            frequency_spectrum_plot(
                vol2dt,
                dt=time_step,
                bw=mask2d,
                output_path=output_path,
                prominence=0.3,
                perf_range=None,
                fd_output=(Im1, Im2, Im0, V1, V2, vent_hz, perf_hz)
                )
        else:
            frequency_spectrum_plot(
                vol2dt,
                dt=time_step,
                bw=mask2d,
                output_path=output_path,
                prominence=0.3,
                fd_output=(Im1, Im2, Im0, V1, V2, vent_hz, perf_hz)
            )
    masked_dc, masked_vent, masked_perf = mask_images(mask2d, Im0, Im1, Im2, background_value=0)

    return masked_vent, masked_perf, vent_hz, perf_hz, masked_dc

def run_dmd(arr3d, mask2d, time_step, output_path, phantom=False):
    arr3d = center_crop_last_dim(arr3d)
    mask2d = center_crop_last_dim(mask2d)
    num_frames, height, width = arr3d.shape
    flattened = arr3d.reshape(num_frames, -1).T

    DMD_ventRange = [0.25, 0.5]
    DMD_perfRange = [1.2, 3.5]

    rank = 15
    Phi, omega, lambda_, b, freq, Xdmd, r = dynamic_mode_decomp(flattened, mask=mask2d, dt=time_step, r = rank)

    maskf = freq >= 0.01
    freq_filt = freq[maskf]
    b_filt = b[maskf]

    # plt.scatter(freq_filt, np.abs(b_filt), linewidth=2)
    # plt.xlim(0, freq_filt.max() * 1.1)
    # plt.ylim(0, np.abs(b_filt).max() * 1.1)
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel('Amplitude', fontsize=12)
    # plt.title('Mean-signal spectrum DMD')
    # plt.savefig(os.path.join(output_path, 'dmd_modes.jpg'))
    # plt.show(block=False)

    # When analyzing a phantom, skip perfusion detection by passing perfRange=None
    perfRange_arg = None if phantom else DMD_perfRange
    dc_DMD, ventMap, perfMap = process_DMD_modes(Phi, freq, lambda_, b, r, sx=256, sy=256, ventRange=DMD_ventRange, perfRange=perfRange_arg, mask=mask2d)

    vent_idxs = np.where((freq > DMD_ventRange[0]) & (freq < DMD_ventRange[1]))[0]
    vent_freqs = np.sort(freq[vent_idxs])
    perf_freqs = None
    if not phantom:
        perf_idxs = np.where((freq > DMD_perfRange[0]) & (freq < DMD_perfRange[1]))[0]
        perf_freqs = np.sort(freq[perf_idxs])

    # Return DMD maps, frequency lists, and DMD internals needed for plotting
    return dc_DMD, ventMap, perfMap, vent_freqs, perf_freqs, Phi, freq, b, r, lambda_


def compute_ventilation_perfusion(
    moving_series,
    parameter_file_registration,
    time_array,
    skip_first=8,
    segmentation_method='automatic',
    phantom=False,
    output_path=None,
    plotting=False,
):
    """
    Core pipeline: registration → segmentation → Fourier decomposition.
    
    Args:
        moving_series: Pre-loaded SITK image series
        parameter_file_registration: Path to registration parameter file
        time_array: Acquisition times in seconds
        skip_first: Number of initial frames to omit
        segmentation_method: 'manual', 'automatic', or 'napari'
        phantom: Skip perfusion if True
        output_path: Optional path for saving Fourier results
    
    Returns:
        dict with registration, segmentation, and Fourier results
    """
    # Registration
    arr3d, vol2dt = run_registration(parameter_file_registration, moving_series, skip_first=skip_first)
    
    # Segmentation
    mean2d_np, mask2d = compute_masks_and_mean(vol2dt, segmentation_method=segmentation_method, plotting=plotting)
    
    # Timing
    time_step = mean_step_size(time_array)
    print('Estimated time step (s):', time_step)
    
    # Fourier decomposition
    masked_vent, masked_perf, vent_hz, perf_hz, masked_dc = run_fourier(
        vol2dt, mask2d, time_step, output_path, mean2d_np, phantom=phantom, plotting = plotting
    )
    
    return {
        'arr3d': arr3d,
        'vol2dt': vol2dt,
        'mean2d_np': mean2d_np,
        'mask2d': mask2d,
        'ventMap': masked_vent,
        'perfMap': masked_perf,
        'vent_hz': vent_hz,
        'perf_hz': perf_hz,
        'masked_dc': masked_dc,
        'time_step': time_step,
    }

def main(series_indicator, base_dir=None, skip_first=8, segmentation_method='automatic', phantom=False):
    paths = setup_paths(series_indicator, base_dir=base_dir)

    print('Reading moving series from', paths['input_path_registration'])
    moving_series = read_images_from_folder(paths['input_path_registration'])
    
    print('Getting DICOM acquisition times...')
    time_array = get_dicom_acquisition_times(paths['input_path_registration'])
    
    result = compute_ventilation_perfusion(
        moving_series,
        paths['parameter_file_registration'],
        time_array,
        skip_first=skip_first,
        segmentation_method=segmentation_method,
        phantom=phantom,
        output_path=paths['output_path'],
        plotting = False
    )

    # save results for potential further analysis
    np.savez(os.path.join(r'C:\Lung_Project\PostProcessing\ventilation_and_perfusion_maps\ventilation_and_perfusion_maps\Results', series_indicator + '_results.npz'), **result)
    
    print('Checking input shapes')
    print('arr3d shape:', result['arr3d'].shape)
    print('mask2d shape:', result['mask2d'].shape)

    VQMaps = np.stack((result['ventMap'], result['perfMap']), axis=-1)
    print('VQMaps shape:', VQMaps.shape) 

def VQMapping_online(data, head, base_dir=None, skip_first=8, segmentation_method='automatic', phantom=False):

    """Process data in OpenRecon-style format: inputs are 'data, head'.

    This mirrors the offline 'main()' pipeline but accepts an in-memory
    image array and ISMRMRD header.
    """
    if base_dir is None:
         base_dir = os.path.dirname(os.path.abspath(__file__))

    parameter_file_registration = os.path.join(base_dir, 'registration_parameter_file.txt')
    temp = os.path.join(base_dir, 'temp')
    ensure_dir(temp)

    # Convert incoming raw data to a SITK image
    moving_series = array_to_sitk(data)


    # Get acquisition times from ISMRMRD header (or use hardcoded fallback)
    time_array = get_ismrmrd_acquisition_times(head)
    
    # Run core pipeline
    result = compute_ventilation_perfusion(
        moving_series,
        parameter_file_registration,
        time_array,
        skip_first=skip_first,
        segmentation_method=segmentation_method,
        phantom=phantom,
        output_path=temp,
        plotting = False
    )

    # Stack and return ventilation/perfusion maps
    VQMaps = np.stack((result['ventMap'], result['perfMap']), axis=-1)
    print('checking values') # debugging
    print('min:', VQMaps.min()) # debugging
    print('max:', VQMaps.max()) # debugging
    return VQMaps, result['mean2d_np'] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', '-s', default='20260113_age2', help='Dataset series indicator') # tag for phantom: trufi_lung_VT600ml_Freq20
    parser.add_argument('--skip-first', type=int, default=8, help='Number of initial frames to omit')
    parser.add_argument('--segmentation_method', type=str, default='automatic', choices=['manual', 'automatic', 'napari'], help='Method used for Segmentation')
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in ('yes', 'y', 'true', 't', '1', 'on'):
            return True
        if v in ('no', 'n', 'false', 'f', '0', 'off'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

    parser.add_argument('--phantom', type=str2bool, nargs='?', const=True, default=False,
                        help='Only analyze ventilation (skip perfusion). Accepts true/false (default: false)')
    args = parser.parse_args()
    main(args.series, skip_first=args.skip_first, segmentation_method=args.segmentation_method, phantom=args.phantom)