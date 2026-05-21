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
)

from .Plotting import (
    plot_overlays,
    plot_individual_modes,
    plot_results,
)

def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


# def setup_paths(series_indicator, base_dir=None):
#     if base_dir is None:
#         base_dir = os.path.dirname(os.path.abspath(__file__))

#     data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'Measurements'))

#     input_path_registration = os.path.join(data_dir, series_indicator)
#     output_path_registration = os.path.join(base_dir, 'Results', series_indicator)
#     parameter_file_registration = os.path.join(base_dir, 'registration_parameter_file.txt')
#     input_path = os.path.join(base_dir, 'Results', series_indicator, 'stack_transformix', 'registered_series.dcm')
#     output_path = os.path.join(base_dir, 'Results', series_indicator, 'FD_DMD')

#     ensure_dir(output_path_registration)
#     ensure_dir(output_path)

#     return {
#         'input_path_registration': input_path_registration,
#         'output_path_registration': output_path_registration,
#         'parameter_file_registration': parameter_file_registration,
#         'input_path': input_path,
#         'output_path': output_path,
#     }


def run_registration(input_path_registration, parameter_file_registration, output_path_registration, skip_first=8):
    print('Reading moving series from', input_path_registration)
    moving_series = read_images_from_folder(input_path_registration)

    if skip_first and skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)

    fixed_index = find_middle_intensity_slice(moving_series)
    fixed_image = extract_2d_slice(moving_series, int(fixed_index))

    image_series_registration(moving_series, fixed_image, parameter_file_registration, output_path_registration)


def compute_masks_and_mean(vol2dt, segmentation_method):
    mean2d_np = np.mean(vol2dt, axis=2).astype(np.float32)
    mean2d_sitk = sitk.GetImageFromArray(mean2d_np)

    body_mask = extract_body_mask(mean2d_sitk, lowerThreshold=0.25, radius=10)
    _, lung_init = rough_lung_segmentation(mean2d_sitk, body_mask, lung_lower_factor=0.0, lung_upper_factor=0.43)

    if segmentation_method =='manual':
        augmented_lung = manual_segmentation(mean2d_np, output_path = None, brush_size = 5)
        augmented_lung = sitk.GetImageFromArray(augmented_lung.astype(np.uint8))
        
    elif segmentation_method == 'automatic':
        augmented_lung = augment_mask(mean2d_sitk, lung_init, body_mask, neighborhood_radius=10, num_iterations='max')

    full_thorax_mask = connect_lungs_sitk(augmented_lung, closing_radius=(90, 90, 30))

    body_np = sitk.GetArrayFromImage(body_mask).astype(bool)
    lung_init_np = sitk.GetArrayFromImage(lung_init).astype(bool)
    augmented_np = sitk.GetArrayFromImage(augmented_lung).astype(bool)
    # full_thorax_np = sitk.GetArrayFromImage(full_thorax_mask).astype(bool)
    full_thorax_np = augmented_np
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


def run_fourier(vol2dt, mask2d, time_step, output_path, mean2d_np, phantom=False):
    # Show spectrum; when phantom=True, skip perfusion detection in the plot
    if phantom:
        frequency_spectrum_plot(vol2dt, dt=time_step, bw=mask2d, output_path=output_path, prominence=0.3, perf_range=None)
    else:
        frequency_spectrum_plot(vol2dt, dt=time_step, bw=mask2d, output_path=output_path, prominence=0.3)

    Im1, Im2, Im0, V1, V2, vent_hz, perf_hz = fourier_decomp(vol2dt, dt=time_step, bw=mask2d, prominence=0.3, phantom=phantom)
    masked_dc, masked_vent, masked_perf = mask_images(mask2d, Im0, Im1, Im2, background_value=-1)

    # plot_results(mean2d_np, masked_dc, masked_vent, masked_perf, 'FD', filepath=os.path.join(output_path, 'FD.png'), vent_freqs=vent_hz, perf_freqs=perf_hz)
    # plot_overlays(mean2d_np, masked_vent, masked_perf, output_dir=output_path)
    return masked_dc, masked_vent, masked_perf, vent_hz, perf_hz

def run_dmd(arr3d, mask2d, time_step, output_path, phantom=False):
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


# def main(series_indicator, base_dir=None, skip_first=8, segmentation_method='automatic', phantom=False):
#     paths = setup_paths(series_indicator, base_dir=base_dir)

#     run_registration(paths['input_path_registration'], paths['parameter_file_registration'], paths['output_path_registration'], skip_first=skip_first)

#     img3d = sitk.ReadImage(paths['input_path'])
#     arr3d = center_crop_last_dim(sitk.GetArrayFromImage(img3d))
#     vol2dt = arr3d.transpose(1, 2, 0)

#     mean2d_np, mask2d = compute_masks_and_mean(vol2dt, segmentation_method=segmentation_method)

#     print('Getting DICOM acquisition times...')
#     time_array = get_dicom_acquisition_times(paths['input_path_registration'])
#     time_step = mean_step_size(time_array)

#     # run_fourier(vol2dt, mask2d, time_step, paths['output_path'], mean2d_np, phantom=phantom)

#     print('Checking input shapes')
#     print('arr3d shape:', arr3d.shape)
#     print('mask2d shape:', mask2d.shape)
#     dc_DMD, ventMap, perfMap, vent_freqs, perf_freqs, Phi, freq, b, r, lambda_ = run_dmd(arr3d, mask2d, time_step, paths['output_path'], phantom=phantom)

#     plot_results(mean2d_np, dc_DMD, ventMap, perfMap, 'DMD', filepath=os.path.join(paths['output_path'], 'DMD.png'), vent_freqs=vent_freqs, perf_freqs=perf_freqs)
#     plot_overlays(mean2d_np, ventMap, perfMap, output_dir=paths['output_path'])

#     # Ensure mode_plots directory exists and then plot individual DMD modes
#     ensure_dir(os.path.join(paths['output_path'], 'mode_plots'))
#     plot_individual_modes(Phi=Phi, freq=freq, b=b, r=r, mask=mask2d, lambda_=lambda_, output_dir=os.path.join(paths['output_path'], 'mode_plots'))


def VQMapping_online(data, head, base_dir=None, skip_first=8, segmentation_method='automatic', phantom=False):
    """Process data in OpenRecon-style format: inputs are 'data, head'.

    This mirrors the offline 'main()' pipeline but accepts an in-memory
    image array and ISMRMRD header instead of a filesystem series id.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    parameter_file_registration = os.path.join(base_dir, 'registration_parameter_file.txt')
    output_path = None # os.path.join(base_dir, 'Results', 'OpenRecon')
    # ensure_dir(output_path)

    # Convert incoming raw data to a SITK image
    moving_series = array_to_sitk(data)

    if skip_first and skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)

    fixed_index = find_middle_intensity_slice(moving_series)
    fixed_image = extract_2d_slice(moving_series, int(fixed_index))

    # Perform registration (writes to Results/OpenRecon/...)
    fixed_stack, moving_series, applied_stack = image_series_registration(moving_series, fixed_image, parameter_file_registration)

    img3d = applied_stack
    arr3d = center_crop_last_dim(sitk.GetArrayFromImage(img3d))

    # arr3d = center_crop_last_dim(img3d)
    vol2dt = arr3d.transpose(1, 2, 0)

    mean2d_np, mask2d = compute_masks_and_mean(vol2dt, segmentation_method=segmentation_method)

    # Get acquisition times from ISMRMRD header
    time_array = get_ismrmrd_acquisition_times(head)
    time_step = mean_step_size(time_array)

    masked_dc, ventMap, perfMap, vent_hz, perf_hz = run_fourier(vol2dt, mask2d, time_step, output_path, mean2d_np, phantom=phantom)

    # dc_DMD, ventMap, perfMap, vent_freqs, perf_freqs, Phi, freq, b, r, lambda_ = run_dmd(arr3d, mask2d, time_step, output_path, phantom=phantom)

    # plot_results(mean2d_np, dc_DMD, ventMap, perfMap, 'DMD', filepath=os.path.join(output_path, 'DMD.png'), vent_freqs=vent_freqs, perf_freqs=perf_freqs)
    # plot_overlays(mean2d_np, ventMap, perfMap, output_dir=output_path)

    # ensure_dir(os.path.join(output_path, 'mode_plots'))
    # plot_individual_modes(Phi=Phi, freq=freq, b=b, r=r, mask=mask2d, lambda_=lambda_, output_dir=os.path.join(output_path, 'mode_plots'))

    # VMap = create_rgb_overlay(mean2d_np, ventMap, map_type= 'ventilation', alpha=1)
    # QMap = create_rgb_overlay(mean2d_np, perfMap, map_type = 'perfusion', alpha=1)
    # VMap_overlay = create_rgb_overlay(mean2d_np, ventMap, map_type= 'ventilation', alpha= 0.5)
    # QMap_overlay = create_rgb_overlay(mean2d_np, perfMap, map_type = 'perfusion', alpha=0.5)
    

    ### different plots
    # plot_results_DMD(mean2d_np, dc_DMD, ventMap, perfMap, filepath=output_path+'DMD.png', vent_freqs=vent_freqs, perf_freqs=perf_freqs)
    # plot_DMD_overlays(mean2d_np, ventMap, perfMap, output_dir=output_path)
    # plot_individual_modes(Phi=Phi, freq=freq, b=b, r=r, mask=mask2d, lambda_=lambda_, output_dir=output_path+'mode_plots')

    # set top p percent to max value and normalize to [0,1]
    p = 0.95
    VMap = ventMap / np.percentile(ventMap[mask2d], p*100)
    QMap = perfMap / np.percentile(perfMap[mask2d], p*100)
    VMap[VMap > 1] = 1  
    QMap[QMap > 1] = 1

    print('Checking shapes before stacking:', VMap.shape, QMap.shape)

    VQMaps = np.stack((VMap, QMap), axis = -1)
    VQMaps *= 255
    print('VentilationChecking max and mean values', np.max(VMap), np.mean(VMap))
    print('Perfusion Checking max and mean values', np.max(QMap), np.mean(QMap))
    VQMaps = VQMaps.astype(np.uint16)

    return VQMaps

def VQMapping_func(arg1, arg2=None, output_path=None, base_dir=None, skip_first=8, segmentation_method='automatic', phantom=False, OpenRecon=False):
    """Dispatch wrapper: if 'OpenRecon==True', treat inputs as 'data, head'.

    Otherwise 'arg1' is treated as the offline 'series_indicator' and the
    existing 'main()' entrypoint is invoked.
    """
    # if OpenRecon:
    return VQMapping_online(arg1, arg2, base_dir=None, skip_first=skip_first, segmentation_method=segmentation_method, phantom=phantom)
    # else:
        # return main(arg1, base_dir=base_dir, skip_first=skip_first, segmentation_method=segmentation_method, phantom=phantom)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', '-s', default='20251202_age13years', help='Dataset series indicator') # tag for phantom: trufi_lung_VT600ml_Freq20
    parser.add_argument('--skip-first', type=int, default=8, help='Number of initial frames to omit')
    parser.add_argument('--segmentation_method', type=str, default='automatic', choices=['manual', 'automatic'], help='Method used for Segmentation')
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