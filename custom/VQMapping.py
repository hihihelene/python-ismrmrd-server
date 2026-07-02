"""Main driver for ventilation and perfusion mapping.

This module was refactored to remove top-level execution, reduce redundancy,
and group operations into helper functions. Calling 'main()' runs the
full pipeline for a given 'series_indicator'.
"""

import argparse
import os
import numpy as np
import SimpleITK as sitk
from dataclasses import dataclass


try: 
    from .Dynamic_Mode_Decomposition import (
        dynamic_mode_decomp,
        mask_images,
        mean_step_size,
        process_DMD_modes,
    )

    from .Fourier_Decomposition import (
        fourier_decomp,
    )

    from .Reading_and_Writing import (
        get_dicom_acquisition_times,
        read_images_from_folder,
        array_to_sitk, # this is necessary for OpenRecon 
        get_ismrmrd_acquisition_times, # this is necessary for OpenRecon 
    )

    from .Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        image_series_registration,
        omit_first_frames,
    )

    from .Segmentation import (
        segment_automatic,
        segment_napari,
        segment_load,
    )

    from .nnUnet_Segmentation import segment_nnunet

    from .Plotting import (
        plot_overlays,
        plot_individual_modes,
        plot_results,
        plot_segmentation,
        plot_frequency_spectrum_FD,
        plot_modes_DMD,
    )

except ImportError:
    print("Using absolute import paths instead of relative")
    from Dynamic_Mode_Decomposition import (
        dynamic_mode_decomp,
        mask_images,
        mean_step_size,
        process_DMD_modes,
    )

    from Fourier_Decomposition import (
        fourier_decomp,
    )

    from Reading_and_Writing import (
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
        segment_automatic,
        segment_napari,
        segment_load,
    )

    from nnUnet_Segmentation import segment_nnunet

    from Plotting import (
        plot_overlays,
        plot_individual_modes,
        plot_results,
        plot_segmentation,
        plot_frequency_spectrum_FD,
        plot_modes_DMD,
    )

SEGMENTATION_METHODS = {
"automatic": segment_automatic,
"napari": segment_napari,
"load": segment_load,
"nnunet": segment_nnunet,
}

# introducing data class to configure pipeline parameters
@dataclass
class PipelineConfig:
    spectral_method: str = "DMD"
    segmentation_method: str = "automatic"
    series_indicator: str = "20251110_age17"
    skip_first: int = 8
    phantom: bool = False
    plotting: bool = False
    output_path: str | None = None


# introducing data class to configure pipeline parameters
@dataclass
class PipelineResult:
    moving_series: sitk.Image

    registered_volume: np.ndarray
    image_series_xyt: np.ndarray

    mean_image: np.ndarray
    lung_mask: np.ndarray

    vent_map: np.ndarray
    perf_map: np.ndarray

    vent_hz: np.ndarray | None
    perf_hz: np.ndarray | None

    masked_dc: np.ndarray

    time_step: float

    spectrum_freq: np.ndarray | None = None
    spectrum_amp: np.ndarray | None = None

    phi: np.ndarray | None = None
    freq: np.ndarray | None = None
    b: np.ndarray | None = None
    r: int | None = None
    lambda_: np.ndarray | None = None

def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def setup_paths(series_indicator, base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'Measurements', 'CDH_Study'))

    input_path = os.path.join(data_dir, series_indicator)
    output_path = os.path.join(base_dir, 'Results', series_indicator)
    registration_parameter_file = os.path.join(base_dir, 'registration_parameter_file.txt')

    ensure_dir(output_path)

    return {
        'input_path': input_path,
        'output_path': output_path,
        'registration_parameter_file': registration_parameter_file,
    }


def run_registration(parameter_file, moving_series, skip_first=8):
    """
    Perform image series registration.
    
    Args:
        parameter_file: Path to registration parameter file
        moving_series: Pre-loaded SITK image series
        skip_first: Number of initial frames to omit
    
    Returns:
        registered_volume: 3D registered array (z, y, x)
        image_series_xyt: Transposed array for 2D + time processing (y, x, z)
    """
    if skip_first and skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)

    fixed_index = find_middle_intensity_slice(moving_series)
    fixed_image = extract_2d_slice(moving_series, int(fixed_index))

    # Perform registration (no file output, returns in-memory result)
    fixed_stack, moving_series, applied_stack = image_series_registration(moving_series, fixed_image, parameter_file)

    img3d = applied_stack
    registered_volume = sitk.GetArrayFromImage(img3d)

    #compute mean for registered_volume
    print('registered_volume shape:', registered_volume.shape)
    mean_intensity = np.mean(registered_volume[0, :, :])
    print('mean', mean_intensity)
    image_series_xyt = registered_volume.transpose(1, 2, 0)
    return registered_volume, image_series_xyt


def compute_masks_and_mean(image_series_xyt, config):

    """
    Compute 2D masks and mean image from a 3D image series.

    Args:
        image_series_xyt: 3D array (y, x, z)
        segmentation_method: 'automatic', 'napari', 'load', or 'nnunet'
        series_indicator: Optional identifier for saving/loading masks

    Returns:
        mean_image: 2D mean image (y, x)
        lung_mask: 2D binary mask (y, x)
    """

    mean_image = np.mean(image_series_xyt, axis=2).astype(np.float32)

    segmenter = SEGMENTATION_METHODS[config.segmentation_method]

    augmented_lung = segmenter(mean_image, series_indicator=config.series_indicator)

    lung_mask = sitk.GetArrayFromImage(augmented_lung).astype(bool)

    return mean_image, lung_mask 


def run_fourier(image_series_xyt, lung_mask, time_step, config):
    # Run Fourier decomposition first so spectrum plotting can reuse detected peaks
    Im1, Im2, Im0, V1, V2, vent_freq, perf_freq, spectrum_freq, spectrum_amp = fourier_decomp(
        image_series_xyt, dt=time_step, bw=lung_mask, prominence=0.3, phantom=config.phantom
    )

    masked_dc, vent_map, perf_map = mask_images(lung_mask, Im0, Im1, Im2, background_value=0)
    return vent_map, perf_map, vent_freq, perf_freq, masked_dc, spectrum_freq, spectrum_amp

def run_dmd(registered_volume, lung_mask, time_step, config):
    # lung_mask = center_crop_last_dim(lung_mask)
    # registered_volume = center_crop_last_dim(registered_volume)
    num_frames, height, width = registered_volume.shape
    flattened = registered_volume.reshape(num_frames, -1).T

    DMD_VENT_RANGE = [0.25, 0.5]
    DMD_PERF_RANGE = [1.2, 3.5]

    rank = 15
    phi, omega, lambda_, b, freq, Xdmd, r = dynamic_mode_decomp(flattened, mask=lung_mask, dt=time_step, r = rank)

    # When analyzing a phantom, skip perfusion detection by passing perfRange=None
    perfRange_arg = None if config.phantom else DMD_PERF_RANGE
    sy, sx = registered_volume.shape[1:]
    dc_DMD, vent_map, perf_map = process_DMD_modes(phi, freq, lambda_, b, r, sx=sx, sy=sy, ventRange=DMD_VENT_RANGE, perfRange=perfRange_arg, mask=lung_mask)

    vent_idxs = np.where((freq > DMD_VENT_RANGE[0]) & (freq < DMD_VENT_RANGE[1]))[0]
    vent_freqs = np.sort(freq[vent_idxs])
    perf_freqs = None

    if not config.phantom:
        perf_idxs = np.where((freq > DMD_PERF_RANGE[0]) & (freq < DMD_PERF_RANGE[1]))[0]
        perf_freqs = np.sort(freq[perf_idxs])


    # Return DMD maps, frequency lists, and DMD internals needed for plotting
    return dc_DMD, vent_map, perf_map, vent_freqs, perf_freqs, phi, freq, b, r, lambda_


def compute_ventilation_perfusion(
    moving_series,
    registration_parameter_file,
    time_array,
    config
):
    """
    Core pipeline: registration → segmentation → Fourier decomposition.
    
    Args:
        moving_series: Pre-loaded SITK image series
        registration_parameter_file: Path to registration parameter file
        time_array: Acquisition times in seconds
        skip_first: Number of initial frames to omit
        segmentation_method: 'manual', 'automatic', 'napari', or 'presegmented'
        phantom: Skip perfusion if True
        output_path: Optional path for saving Fourier results
        series_indicator: Identifier for the current series (used for saving results)
    Returns:
        dict with registration, segmentation, and Fourier results
    """
    spectrum_freq = None
    spectrum_amp = None

    phi = None
    freq = None
    b = None
    r = None
    lambda_ = None

    # Registration
    registered_volume, image_series_xyt = run_registration(registration_parameter_file, moving_series, skip_first=config.skip_first)

    # Segmentation
    mean_image, lung_mask = compute_masks_and_mean(image_series_xyt, config)
    
    # Timing
    time_step = mean_step_size(time_array)
    print('Estimated time step (s):', time_step)
    
    # Fourier decomposition
    if config.spectral_method == 'FD':
        vent_map, perf_map, vent_hz, perf_hz, masked_dc, spectrum_freq, spectrum_amp = run_fourier(
            image_series_xyt, lung_mask, time_step, config=config
        )
    # Dynamic Mode Decomposition
    if config.spectral_method == 'DMD':
        masked_dc, vent_map, perf_map, vent_hz, perf_hz, phi, freq, b, r, lambda_ = run_dmd(
            registered_volume, lung_mask, time_step, config=config)


    return PipelineResult(
        moving_series=moving_series,
        registered_volume=registered_volume,
        image_series_xyt=image_series_xyt,
        mean_image=mean_image,
        lung_mask=lung_mask,
        vent_map=vent_map,
        perf_map=perf_map,
        vent_hz=vent_hz,
        perf_hz=perf_hz,
        masked_dc=masked_dc,
        time_step=time_step,
        spectrum_freq=spectrum_freq,
        spectrum_amp=spectrum_amp,
        phi=phi,
        freq=freq,
        b=b,
        r=r,
        lambda_=lambda_,
    )
def main(config: PipelineConfig, base_dir=None):
    paths = setup_paths(config.series_indicator, base_dir=base_dir)

    print('Reading moving series from', paths['input_path'])
    moving_series = read_images_from_folder(paths['input_path'])
    
    print('Getting DICOM acquisition times...')
    time_array = get_dicom_acquisition_times(paths['input_path'])
    
    result = compute_ventilation_perfusion(
        moving_series,
        paths['registration_parameter_file'],
        time_array,
        config
    )

    if config.plotting == True:
        plot_segmentation(result.mean_image, result.lung_mask, config.segmentation_method, paths['output_path'], config.series_indicator)
        plot_results(result.mean_image, result.masked_dc, result.vent_map, result.perf_map, output_path = paths['output_path'], config=config)
        plot_overlays(result.mean_image, result.vent_map, result.perf_map, output_path = paths['output_path'], config=config)
        if config.spectral_method == 'FD':
            plot_frequency_spectrum_FD(result.spectrum_freq, result.spectrum_amp, result.vent_hz, result.perf_hz, output_path = paths['output_path'])
        elif config.spectral_method == 'DMD':
            plot_modes_DMD(result.freq, result.b, result.vent_hz, result.perf_hz, output_path = paths['output_path'])
            image_size_y, image_size_x = result.mean_image.shape
            plot_individual_modes(result.phi, result.freq, result.lambda_, result.b, result.r, mask=result.lung_mask, sx=image_size_x, sy=image_size_y, output_path = paths['output_path'])
    
    # save results for potential further analysis
    # np.savez(os.path.join(r'C:\Lung_Project\PostProcessing\ventilation_and_perfusion_maps\ventilation_and_perfusion_maps\Results', series_indicator + '_results.npz'), **asdict(result))

def VQMapping_online(data, head, base_dir=None, config: PipelineConfig = PipelineConfig()):

    """Process data in OpenRecon-style format: inputs are 'data, head'.

    This mirrors the offline 'main()' pipeline but accepts an in-memory
    image array and ISMRMRD header.
    """

    
    paths = setup_paths(config.series_indicator, base_dir=base_dir)

    print('Converting input array to SimpleITK image series...')
    moving_series = array_to_sitk(data)
    
    print('Getting ISMRMRD acquisition times...')
    time_array = get_ismrmrd_acquisition_times(head)
    
    result = compute_ventilation_perfusion(
        moving_series,
        paths['registration_parameter_file'],
        time_array,
        config
    )

    # transforming data for display on scanner console, scaling to 0-255 and converting to uint16

    # setting scaling factor to 95th percentile to avoid outliers dominating the scaling
    p = 0.95
    VMap = result.vent_map / np.percentile(result.vent_map[result.lung_mask], p*100)
    QMap = result.perf_map / np.percentile(result.perf_map[result.lung_mask], p*100)
    VMap[VMap > 1] = 1  
    QMap[QMap > 1] = 1

    print('Checking shapes before stacking:', VMap.shape, QMap.shape)

    VQMaps = np.stack((VMap, QMap), axis = -1)
    VQMaps *= 255
    print('VentilationChecking max and mean values', np.max(VMap), np.mean(VMap))
    print('Perfusion Checking max and mean values', np.max(QMap), np.mean(QMap))
    VQMaps = VQMaps.astype(np.uint16)

    return VQMaps
    # transforming data for display on scanner console, scaling to 0-255 and converting to uint16

    # setting scaling factor to 95th percentile to avoid outliers dominating the scaling
    p = 0.95
    VMap = result.vent_map / np.percentile(result.vent_map[result.lung_mask], p*100)
    QMap = result.perf_map / np.percentile(result.perf_map[result.lung_mask], p*100)
    VMap[VMap > 1] = 1  
    QMap[QMap > 1] = 1

    print('Checking shapes before stacking:', VMap.shape, QMap.shape)

    VQMaps = np.stack((VMap, QMap), axis = -1)
    VQMaps *= 255
    print('VentilationChecking max and mean values', np.max(VMap), np.mean(VMap))
    print('Perfusion Checking max and mean values', np.max(QMap), np.mean(QMap))
    VQMaps = VQMaps.astype(np.uint16)

    return VQMaps

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="20260120_age18")
    parser.add_argument("--segmentation-method", default="nnunet")
    parser.add_argument("--spectral-method", default="DMD")
    parser.add_argument("--plotting", default=True)
    parser.add_argument("--phantom", default=False)
    args = parser.parse_args()
    config = PipelineConfig(
        series_indicator=args.series,
        segmentation_method=args.segmentation_method,
        spectral_method=args.spectral_method,
        plotting=args.plotting,
        phantom=args.phantom
    )

    main(config)