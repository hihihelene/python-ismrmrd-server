"""Main driver for ventilation and perfusion mapping.

This module was refactored to remove top-level execution, reduce redundancy,
and group operations into helper functions. Calling "main()" runs the
full pipeline for a given "series_indicator".
"""

import argparse
import os
import warnings
from dataclasses import dataclass
import numpy as np
import itk


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
        array_to_itk, # this is necessary for OpenRecon
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
        plot_registered_series,
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
        array_to_itk,
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
        plot_registered_series,
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
    """Configuration for running the processing pipeline.

    Attributes:
        spectral_method: Spectral decomposition method to use (e.g. "DMD").
        segmentation_method: Segmentation approach key.
        series_indicator: Identifier for the dataset series.
        skip_first: Number of initial frames to skip.
        phantom: Whether the data is a phantom.
        plotting: Whether to generate plots.
        output_path: Optional output directory override.
    """

    spectral_method: str = "FD"
    segmentation_method: str = "nnUnet"
    series_indicator: str = "20251110_age17"
    skip_first: int = 8
    phantom: bool = False
    plotting: bool = True
    output_path: str | None = None


# introducing data class to configure pipeline parameters
@dataclass
class PipelineResult:
    """Results from running the processing pipeline.

    Attributes:
        moving_series: Original ITK image series.
        registered_volume: 3D registered array (z, y, x).
        image_series_xyt: Transposed array for 2D + time processing (y, x, z).
        mean_image: 2D mean image (y, x).
        lung_mask: 2D binary mask (y, x).
        vent_map: Ventilation map.
        perf_map: Perfusion map.
        vent_hz: Ventilation frequencies.
        perf_hz: Perfusion frequencies.
        masked_dc: Masked DC component of the image series.
        time_step: Time step between frames.
        spectrum_freq: Frequency spectrum from Fourier decomposition.
        spectrum_power: Power spectrum from Fourier decomposition.
        phi: DMD modes.
        freq: DMD frequencies.
        b: DMD amplitudes.
        r: DMD rank.
        lambda_: DMD eigenvalues.
    """
    moving_series: object

    registered_volume: np.ndarray
    image_series_xyt: np.ndarray

    mean_image: np.ndarray
    lung_mask: np.ndarray

    vent_map: np.ndarray | None
    perf_map: np.ndarray | None

    vent_hz: np.ndarray | None
    perf_hz: np.ndarray | None

    masked_dc: np.ndarray

    time_step: float

    spectrum_freq: np.ndarray | None = None
    spectrum_power: np.ndarray | None = None

    phi: np.ndarray | None = None
    freq: np.ndarray | None = None
    b: np.ndarray | None = None
    r: int | None = None
    lambda_: np.ndarray | None = None

def ensure_dir(path):
    """ Ensure that a directory exists; create it if it does not."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def normalize_map_for_output(map_data, lung_mask, upper_percentile=95):
    """Normalize a map for scanner output, including unavailable maps."""
    mask = np.asarray(lung_mask, dtype=bool)
    if map_data is None:
        warnings.warn(
            "Map is unavailable because its frequency was not detected.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    output = np.asarray(map_data, dtype=float)
    valid_mask = mask & np.isfinite(output)
    if not np.any(valid_mask):
        return np.zeros(output.shape, dtype=float)

    scale = np.nanpercentile(output[valid_mask], upper_percentile)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(output.shape, dtype=float)

    output = np.divide(output, scale, out=np.zeros_like(output), where=np.isfinite(output))
    output[~np.isfinite(output)] = 0
    return np.clip(output, 0, 1)

def setup_paths(series_indicator, base_dir=None):
    """Set up input and output paths based on the series indicator."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # build data directory path (keep lines short for linting)
    file_dir = os.path.dirname(__file__)
    parts = (
        file_dir,
        os.pardir,
        os.pardir,
        os.pardir,
        "Measurements",
        "CDH_Study",
    )

    data_dir = os.path.abspath(os.path.join(*parts))

    input_path = os.path.join(data_dir, series_indicator)
    output_path = os.path.join(base_dir, "Results", series_indicator)
    registration_parameter_file = os.path.join(base_dir, "registration_parameter_file.txt")

    ensure_dir(output_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "registration_parameter_file": registration_parameter_file,
    }


def run_registration(parameter_file, moving_series, skip_first=8):
    """
    Perform image series registration.
    
    Args:
        parameter_file: Path to registration parameter file
        moving_series: Pre-loaded ITK image series
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
    _fixed_stack, moving_series, applied_stack = (
        image_series_registration(
            moving_series,
            fixed_image,
            parameter_file
        )
    )

    img3d = applied_stack
    registered_volume = itk.array_from_image(img3d)

    #compute mean for registered_volume
    print("registered_volume shape:", registered_volume.shape)
    mean_intensity = np.mean(registered_volume[0, :, :])
    print("mean", mean_intensity)
    image_series_xyt = registered_volume.transpose(1, 2, 0)
    return registered_volume, image_series_xyt


def compute_masks_and_mean(image_series_xyt, config):

    """
    Compute 2D masks and mean image from a 3D image series.

    Args:
        image_series_xyt: 3D array (y, x, z)
        segmentation_method: "automatic", "napari", "load", or "nnunet"
        series_indicator: Optional identifier for saving/loading masks

    Returns:
        mean_image: 2D mean image (y, x)
        lung_mask: 2D binary mask (y, x)
    """

    mean_image = np.mean(image_series_xyt, axis=2).astype(np.float32)

    segmenter = SEGMENTATION_METHODS[config.segmentation_method]

    augmented_lung = segmenter(mean_image, series_indicator=config.series_indicator)

    lung_mask = np.asarray(augmented_lung).astype(bool)

    return mean_image, lung_mask


def run_fourier(image_series_xyt, lung_mask, time_step, config):
    """
    Perform Fourier decomposition on the image series.
    
    Inputs:
        image_series_xyt: 3D array (y, x, z)
        lung_mask: 2D binary mask (y, x)
        time_step: Time step between frames
        config: PipelineConfig object with configuration parameters
        
    Returns:
        vent_map: Ventilation map
        perf_map: Perfusion map
        vent_freq: Ventilation frequencies
        perf_freq: Perfusion frequencies
        masked_dc: Masked DC component of the image series
        spectrum_freq: Frequency spectrum from Fourier decomposition
        spectrum_power: Power spectrum from Fourier decomposition
        """
    # Run Fourier decomposition first so spectrum plotting can reuse detected peaks
    (
        vent_image,
        perf_image,
        dc_image,
        _ventilation_series,
        _perfusion_series,
        vent_freq,
        perf_freq,
        spectrum_freq,
        spectrum_power,
    ) = fourier_decomp(
        image_series_xyt,
        time_step=time_step,
        mask=lung_mask,
        prominence=0.3,
        phantom=config.phantom,
    )

    masked_dc, vent_map, perf_map = mask_images(
        lung_mask,
        dc_image,
        vent_image,
        perf_image,
        background_value=0,
    )
    return vent_map, perf_map, vent_freq, perf_freq, masked_dc, spectrum_freq, spectrum_power

def run_dmd(registered_volume, lung_mask, time_step, config):
    """
    Perform Dynamic Mode Decomposition on the registered volume.
    
    Inputs:
        registered_volume: 3D array (z, y, x)
        lung_mask: 2D binary mask (y, x)
        time_step: Time step between frames
        config: PipelineConfig object with configuration parameters

    Returns:
        dc_dmd: Masked DC component of the DMD decomposition
        vent_map: Ventilation map
        perf_map: Perfusion map
        vent_freqs: Ventilation frequencies
        perf_freqs: Perfusion frequencies
        phi: DMD modes
        freq: DMD frequencies
        b: DMD amplitudes
        r: DMD ranks
        lambda_: DMD eigenvalues
    """
    num_frames, _height, _width = registered_volume.shape
    flattened = registered_volume.reshape(num_frames, -1).T

    dmd_vent_range = [0.25, 0.5]
    dmd_perf_range = [1.2, 3.5]

    rank = 15
    phi, _omega, lambda_, b, freq, _x_dmd, r = dynamic_mode_decomp(
        flattened,
        mask=lung_mask,
        dt=time_step,
        r=rank,
    )

    # When analyzing a phantom, skip perfusion detection by passing perfRange=None
    perf_range_arg = None if config.phantom else dmd_perf_range
    sy, sx = registered_volume.shape[1:]
    dc_dmd, vent_map, perf_map = process_DMD_modes(
        phi,
        freq,
        lambda_,
        b,
        r,
        sx=sx,
        sy=sy,
        ventRange=dmd_vent_range,
        perfRange=perf_range_arg,
        mask=lung_mask,
    )

    vent_idxs = np.where((freq > dmd_vent_range[0]) & (freq < dmd_vent_range[1]))[0]
    vent_freqs = np.sort(freq[vent_idxs])
    perf_freqs = None

    if not config.phantom:
        perf_idxs = np.where((freq > dmd_perf_range[0]) & (freq < dmd_perf_range[1]))[0]
        perf_freqs = np.sort(freq[perf_idxs])


    # Return DMD maps, frequency lists, and DMD internals needed for plotting
    return dc_dmd, vent_map, perf_map, vent_freqs, perf_freqs, phi, freq, b, r, lambda_


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
        segmentation_method: "manual", "automatic", "napari", or "presegmented"
        phantom: Skip perfusion if True
        output_path: Optional path for saving Fourier results
        series_indicator: Identifier for the current series (used for saving results)
    Returns:
        dict with registration, segmentation, and Fourier results
    """
    spectrum_freq = None
    spectrum_power = None

    vent_map = None
    perf_map = None
    vent_hz = None
    perf_hz = None
    masked_dc = None

    phi = None
    freq = None
    b = None
    r = None
    lambda_ = None

    # Registration
    registered_volume, image_series_xyt = run_registration(
        registration_parameter_file,
        moving_series,
        skip_first=config.skip_first,
    )

    # Segmentation
    mean_image, lung_mask = compute_masks_and_mean(image_series_xyt, config)
    # Timing
    time_step = mean_step_size(time_array)
    print("Estimated time step (s):", time_step)

    # Fourier decomposition
    if config.spectral_method == "FD":
        vent_map, perf_map, vent_hz, perf_hz, masked_dc, spectrum_freq, spectrum_power = run_fourier(
            image_series_xyt, lung_mask, time_step, config=config
        )

    # Dynamic Mode Decomposition
    elif config.spectral_method == "DMD":
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
        spectrum_power=spectrum_power,
        phi=phi,
        freq=freq,
        b=b,
        r=r,
        lambda_=lambda_,
    )
def main(config: PipelineConfig, base_dir=None):
    """Run the ventilation and perfusion processing pipeline."""
    paths = setup_paths(config.series_indicator, base_dir=base_dir)

    print("Reading moving series from", paths["input_path"])
    moving_series = read_images_from_folder(paths["input_path"])

    print("Getting DICOM acquisition times...")
    time_array = get_dicom_acquisition_times(paths["input_path"])

    result = compute_ventilation_perfusion(
        moving_series,
        paths["registration_parameter_file"],
        time_array,
        config
    )

    if config.plotting is True:
        plot_segmentation(
            result.mean_image,
            result.lung_mask,
            config.segmentation_method,
            paths["output_path"],
            config.series_indicator,
        )
        plot_results(
            result.mean_image,
            result.masked_dc,
            result.vent_map,
            result.perf_map,
            output_path=paths["output_path"],
            config=config,
        )
        plot_overlays(
            result.mean_image,
            result.vent_map,
            result.perf_map,
            output_path=paths["output_path"],
            config=config,
        )
        if config.spectral_method == "FD":
            plot_frequency_spectrum_FD(
                result.spectrum_freq,
                result.spectrum_power,
                result.vent_hz,
                result.perf_hz,
                output_path=paths["output_path"],
            )
        elif config.spectral_method == "DMD":
            plot_modes_DMD(
                result.freq,
                result.b,
                result.vent_hz,
                result.perf_hz,
                output_path=paths["output_path"],
            )
            image_size_y, image_size_x = result.mean_image.shape
            plot_individual_modes(
                result.phi,
                result.freq,
                result.lambda_,
                result.b,
                result.r,
                mask=result.lung_mask,
                sx=image_size_x,
                sy=image_size_y,
                output_path=paths["output_path"],
            )

        # plot the registered series for visual inspection
        # plot_registered_series(result.registered_volume, output_path=paths["output_path"])

def vq_mapping_online(data, head, base_dir=None, config: PipelineConfig = PipelineConfig()):

    """Process data in OpenRecon-style format: inputs are "data, head".

    This mirrors the offline "main()" pipeline but accepts an in-memory
    image array and ISMRMRD header.
    """

    paths = setup_paths(config.series_indicator, base_dir=base_dir)

    print("Checking configuration:")
    print("  Series indicator:", config.series_indicator)
    print("  Segmentation method:", config.segmentation_method)
    print("  Spectral method:", config.spectral_method)
    print("  Plotting:", config.plotting)
    print("  Phantom:", config.phantom)

    print("Converting input array to ITK image series...")
    moving_series = array_to_itk(data)

    print("Getting ISMRMRD acquisition times...")
    time_array = get_ismrmrd_acquisition_times(head)

    result = compute_ventilation_perfusion(
        moving_series,
        paths["registration_parameter_file"],
        time_array,
        config
    )

    # transforming data for display on scanner console, scaling to 0-255 and converting to uint16

    v_map = normalize_map_for_output(result.vent_map, result.lung_mask)
    q_map = normalize_map_for_output(result.perf_map, result.lung_mask)

    # The scanner response requires two numeric channels. Missing maps remain
    # None in the pipeline and are represented only at this final boundary.
    if v_map is None:
        warnings.warn(
            "Ventilation output is unavailable; using an empty scanner channel.",
            RuntimeWarning,
            stacklevel=2,
        )
        v_map = np.zeros(result.lung_mask.shape, dtype=float)
    if q_map is None:
        warnings.warn(
            "Perfusion output is unavailable; using an empty scanner channel.",
            RuntimeWarning,
            stacklevel=2,
        )
        q_map = np.zeros(result.lung_mask.shape, dtype=float)

    print("Checking shapes before stacking:", v_map.shape, q_map.shape)

    vq_maps = np.stack((v_map, q_map), axis = -1)
    vq_maps *= 255
    print("VentilationChecking max and mean values", np.max(v_map), np.mean(v_map))
    print("Perfusion Checking max and mean values", np.max(q_map), np.mean(q_map))
    vq_maps = vq_maps.astype(np.uint16)

    return vq_maps

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="20250729_age13_1")
    # parser.add_argument("--series", default="trufi_lung_VT400ml_Freq20")
    parser.add_argument("--segmentation-method", default="nnunet")
    parser.add_argument("--spectral-method", default="FD")
    parser.add_argument("--plotting", default=True)
    parser.add_argument("--phantom", default=False)
    args = parser.parse_args()
    pipeline_config = PipelineConfig(
        series_indicator=args.series,
        segmentation_method=args.segmentation_method,
        spectral_method=args.spectral_method,
        plotting=args.plotting,
        phantom=args.phantom
    )

    main(pipeline_config)
