"""Main driver for ventilation and perfusion mapping.

This module was refactored to remove top-level execution, reduce redundancy,
and group operations into helper functions. Calling "main()" runs the
full pipeline for a given "series_indicator".
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass


import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO)

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
    from .GOREG_Registration import (
        image_series_goreg_registration,
    )
    from .nnUnet_Segmentation import segment_nnunet
    from .Plotting import (
        plot_frequency_spectrum_FD,
        plot_individual_modes,
        plot_modes_DMD,
        plot_overlays,
        plot_results,
        plot_segmentation,
    )
    from .Reading_and_Writing import (
        array_to_sitk,  # this is necessary for OpenRecon
        get_dicom_acquisition_times,
        get_ismrmrd_acquisition_times,  # this is necessary for OpenRecon
        read_images_from_folder,
    )
    from .Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        image_series_registration,
        omit_first_frames,
    )
    from .Registration_Ablation import INTERPOLATORS, PreprocessingAblation
    from .Segmentation import (
        segment_automatic,
        segment_load,
        segment_napari,
    )
    from .VoxelMorph_Registration import VoxelMorphRegistration
    from .HyperMorph_Registration import HyperMorphRegistration
    from .registration_context import RegistrationResult

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
    from GOREG_Registration import (
        image_series_goreg_registration,
    )
    from nnUnet_Segmentation import segment_nnunet
    from Plotting import (
        plot_frequency_spectrum_FD,
        plot_individual_modes,
        plot_modes_DMD,
        plot_overlays,
        plot_results,
        plot_segmentation,
    )
    from Reading_and_Writing import (
        array_to_sitk,
        get_dicom_acquisition_times,
        get_ismrmrd_acquisition_times,
        read_images_from_folder,
    )
    from Registration import (
        extract_2d_slice,
        find_middle_intensity_slice,
        image_series_registration,
        omit_first_frames,
    )
    from Registration_Ablation import INTERPOLATORS, PreprocessingAblation
    from Segmentation import (
        segment_automatic,
        segment_load,
        segment_napari,
    )
    from VoxelMorph_Registration import VoxelMorphRegistration
    from HyperMorph_Registration import HyperMorphRegistration
    from registration_context import RegistrationResult

SEGMENTATION_METHODS = {
    "automatic": segment_automatic,
    "napari": segment_napari,
    "load": segment_load,
    "nnunet": segment_nnunet,
}
# Classical methods with their own call signature. The learned methods
# ("voxelmorph", "ablation_*", "hypermorph") share one
# call contract and are created by build_registration_model().
REGISTRATION_METHODS = {
    "goreg",
    "bspline",
    "voxelmorph",
    "hypermorph",
    "ablation",
}
# Default Series Folder
DATA_DIR = r"C:\Lung_Project\Measurements\CDH_Study" # 


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
    segmentation_method: str = "automatic"
    registration_method: str = "bspline"
    series_indicator: str = "20251110_age17"

    device: str = "cpu"  # or "cuda" for GPU acceleration
    voxelmorph_checkpoint: str = r"models\Voxel_Hyper_Morph_Modelle\Voxelmorph\best.pt"
    # Training config (YAML/JSON) belonging to voxelmorph_checkpoint. Read
    # directly by VoxelMorphRegistration for the network architecture
    voxelmorph_config: str = r"models\Voxel_Hyper_Morph_Modelle\Voxelmorph\config.yaml"
    hypermorph_checkpoint: str = r"models\Voxel_Hyper_Morph_Modelle\Hypermorph\best.pt"
    hypermorph_config: str = r"models\Voxel_Hyper_Morph_Modelle\Hypermorph\config.yaml"
    # Fixed HyperMorph hyperparameters, passed directly to register().
    # HyperMorph does not search for an optimal pair internally, so
    # these values must be chosen manually.
    hypermorph_lambda: float = 0.431945
    hypermorph_gamma: float = 0.291229
    skip_first: int = 8
    verbose: bool = True
    phantom: bool = False
    plotting: bool = True
    output_path: str | None = None


# introducing data class to configure pipeline parameters
@dataclass
class PipelineResult:
    """Results from running the complete processing pipeline.

    The result is divided conceptually into registration, segmentation,
    and spectral-analysis results.

    Attributes:
        moving_series:
            Original input SITK image series before registration.
        registered_series:
            Registered SITK image series.
        registered_volume:
            Registered series as NumPy array with shape (T, Y, X).
        image_series_xyt:
            Registered series transposed to (Y, X, T).
        reference_image:
            Fixed/reference image used by the registration.
        reference_index:
            Index of the fixed/reference frame.
        fixed_lung_mask:
            Lung mask in reference-image geometry.
        registered_lung_mask:
            Registered moving lung mask in reference-image geometry.
        disp_fields:
            Per-frame displacement fields, shape (T, Y, X, 2), when
            available.
        mean_image:
            Temporal mean image.
        lung_mask:
            Mask used for the spectral analysis.
        vent_map:
            Ventilation map.
        perf_map:
            Perfusion map.
        vent_hz:
            Ventilation frequencies.
        perf_hz:
            Perfusion frequencies.
        masked_dc:
            Masked DC component.
        time_step:
            Mean temporal spacing between frames.
        spectrum_freq:
            Frequency spectrum from Fourier decomposition.
        spectrum_power:
            Power spectrum from Fourier decomposition.
        phi:
            DMD modes.
        freq:
            DMD frequencies.
        b:
            DMD amplitudes.
        r:
            DMD rank.
        lambda_:
            DMD eigenvalues.
    """

    # ------------------------------------------------------------------
    # Input / registration
    # ------------------------------------------------------------------

    moving_series: sitk.Image
    registered_series: sitk.Image

    registered_volume: np.ndarray
    image_series_xyt: np.ndarray

    reference_image: np.ndarray
    reference_index: int

    fixed_lung_mask: np.ndarray | None
    registered_lung_mask: np.ndarray | None
    disp_fields: np.ndarray | None

    # ------------------------------------------------------------------
    # Segmentation / spectral analysis
    # ------------------------------------------------------------------

    mean_image: np.ndarray
    lung_mask: np.ndarray

    vent_map: np.ndarray
    perf_map: np.ndarray

    vent_hz: np.ndarray | None
    perf_hz: np.ndarray | None

    masked_dc: np.ndarray

    time_step: float

    spectrum_freq: np.ndarray | None = None
    spectrum_power: np.ndarray | None = None

    # ------------------------------------------------------------------
    # DMD-specific results
    # ------------------------------------------------------------------

    phi: np.ndarray | None = None
    freq: np.ndarray | None = None
    b: np.ndarray | None = None
    r: int | None = None
    lambda_: np.ndarray | None = None



def ensure_dir(path):
    """Ensure that a directory exists; create it if it does not."""
    """Ensure that a directory exists; create it if it does not."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)



def setup_paths(series_indicator, base_dir=None):
    """Set up input and output paths based on the series indicator."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(DATA_DIR, series_indicator)
    output_path = os.path.join(base_dir, "Results", series_indicator)
    registration_parameter_file = "registration_parameter_file.txt"

    ensure_dir(output_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "registration_parameter_file": registration_parameter_file,
    }


def build_registration_model(method: str, config: PipelineConfig):
    """Create the learned registration model for ``method`` (else None).

    All returned models share one call contract::

        model.register(moving_series, fixed_index, **options) -> sitk.Image

    * ``moving_series``: 3D image (x, y, t) with the warm-up frames already
      removed (``omit_first_frames``).
    * ``fixed_index``: reference frame as index within ``moving_series``.
      Both VoxelMorph and HyperMorph use the *same* reference: the
      median-intensity frame found by ``find_middle_intensity_slice``,
      computed once in ``run_registration`` and passed in. Neither model
      chooses its own reference anymore (no PREFUL/respiratory search).
    * returns a 3D image in the original geometry, one frame per input frame.
    """
    if method.startswith("ablation"):
        return PreprocessingAblation(config_path=config.voxelmorph_config)

    if method == "voxelmorph":
        checkpoint, config_path = config.voxelmorph_checkpoint, config.voxelmorph_config
    elif method == "hypermorph":
        return HyperMorphRegistration(
            checkpoint_path=config.hypermorph_checkpoint,
            config_path=config.hypermorph_config,
            device=config.device,
        )
    else:
        return None

    return VoxelMorphRegistration(
        checkpoint_path=checkpoint,
        config_path=config_path,
        device=config.device,
    )


def run_registration(
    parameter_file,
    moving_series,
    skip_first=8,
    method: str = "bspline",
    registration_model=None,
    hypermorph_lambda: float | None = None,
    hypermorph_gamma: float | None = None,
) -> RegistrationResult:
    """Perform image-series registration.

    The function returns a standardized RegistrationResult so that
    downstream evaluation does not depend on the individual registration
    implementation.
    """

    # --------------------------------------------------------------
    # Frame selection
    # --------------------------------------------------------------

    if skip_first and skip_first > 0:
        moving_series = omit_first_frames(moving_series, skip_first)

    moving_np = sitk.GetArrayFromImage(moving_series)

    if moving_np.ndim != 3:
        raise ValueError(f"Expected a 3D image series, got shape {moving_np.shape}.")

    # --------------------------------------------------------------
    # Raw/original control
    # --------------------------------------------------------------

    if method == "raw_original":
        reference_index = find_middle_intensity_slice(moving_series)
        reference_image = extract_2d_slice(
            moving_series,
            int(reference_index),
        )

        registered_series = moving_series

        return RegistrationResult(
            registered_series=registered_series,
            reference_image=sitk.GetArrayFromImage(reference_image).astype(np.float32),
            reference_index=int(reference_index),
            fixed_lung_mask=None,
            registered_lung_mask=None,
            disp_fields=None,
        )

    # --------------------------------------------------------------
    # Learned registration methods
    # --------------------------------------------------------------

    if (
        method == "hypermorph"
        or method == "voxelmorph"
        or method.startswith("ablation")
    ):
        if registration_model is None:
            raise ValueError(
                f"registration_model must be provided when method='{method}'."
            )

        fixed_index = int(find_middle_intensity_slice(moving_series))

        register_kwargs = {}

        if method.startswith("ablation"):
            ablation_mode = method.split("_", 1)[1]
            register_kwargs = {
                "ablation_mode": "resample",
                "resample_interpolator": INTERPOLATORS.get(ablation_mode),
            }

        if method == "hypermorph":
            if hypermorph_lambda is None or hypermorph_gamma is None:
                raise ValueError(
                    "hypermorph_lambda and hypermorph_gamma must be provided "
                    "when method='hypermorph'."
                )
            register_kwargs["lambda_reg"] = float(hypermorph_lambda)
            register_kwargs["gamma_seg"] = float(hypermorph_gamma)

        applied_stack = registration_model.register(
            moving_series=moving_series,
            fixed_index=fixed_index,
            **register_kwargs,
        )

        # Reference image in the *input* geometry (pre-registration).
        n_out = int(applied_stack.GetSize()[2])
        reference_index = fixed_index
        if not 0 <= reference_index < n_out:
            reference_index = int(np.clip(reference_index, 0, max(n_out - 1, 0)))

        reference_image_sitk = extract_2d_slice(moving_series, reference_index)
        reference_image = sitk.GetArrayFromImage(reference_image_sitk).astype(
            np.float32
        )

        disp_fields = getattr(registration_model, "last_disp_fields", None)

        return RegistrationResult(
            registered_series=applied_stack,
            reference_image=reference_image,
            reference_index=reference_index,
            fixed_lung_mask=None,
            registered_lung_mask=None,
            disp_fields=disp_fields,
        )

    elif method == "bspline":
        fixed_index = find_middle_intensity_slice(moving_series)
        fixed_image = extract_2d_slice(moving_series, int(fixed_index))
        _fixed_stack, moving_series, applied_stack, tpm, disp = (
            image_series_registration(moving_series, fixed_image, parameter_file)
        )
        return RegistrationResult(
            registered_series=applied_stack,
            reference_image=sitk.GetArrayFromImage(fixed_image).astype(np.float32),
            reference_index=int(fixed_index),
            fixed_lung_mask=None,
            registered_lung_mask=None,
            disp_fields=disp,
        )
    elif method == "goreg":
        reference_index = find_middle_intensity_slice(moving_series)
        reference_image_sitk = extract_2d_slice(
            moving_series,
            int(reference_index),
        )
        registration = image_series_goreg_registration(
            moving_series,
            "goreg_registration_parameter_file.txt",
        )

        return RegistrationResult(
            registered_series=registration.registered_series,
            reference_image=sitk.GetArrayFromImage(reference_image_sitk).astype(
                np.float32
            ),
            reference_index=int(reference_index),
            fixed_lung_mask=None,
            registered_lung_mask=None,
            disp_fields=registration.disp_fields,
        )

    raise ValueError(f"Unknown registration method: {method}")


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
    # lung_mask = sitk.GetArrayFromImage(augmented_lung).astype(bool)
    lung_mask = augmented_lung.astype(bool)
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
    return (
        vent_map,
        perf_map,
        vent_freq,
        perf_freq,
        masked_dc,
        spectrum_freq,
        spectrum_power,
    )


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
    moving_series, registration_parameter_file, time_array, config
):
    """
    Core pipeline: registration → segmentation → Fourier decomposition.

    Args:
        moving_series: Pre-loaded SITK image series
        registration_parameter_file: Path to registration parameter file
        time_array: Acquisition times in seconds
        config: PipelineConfig object with configuration parameters

    Returns:
        PipelineResult with registration, segmentation, and spectral results
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

    # None for classical methods (bspline, goreg, raw_original)
    registration_model = build_registration_model(config.registration_method, config)
    if registration_model is not None:
        print(registration_model)
    t = time.perf_counter()
    # Registration
    # TODO Expand return objects to
    # registered_volume, image_series_xyt, reference_image, \
    # registered_lung_mask, disp_field = run_registration(...)
    registration_result = run_registration(
        registration_parameter_file,
        moving_series,
        skip_first=config.skip_first,
        method=config.registration_method,
        registration_model=registration_model,
        hypermorph_lambda=config.hypermorph_lambda,
        hypermorph_gamma=config.hypermorph_gamma,
    )
    print(f"Registration time: {time.perf_counter() - t:.3f}s")
    registered_series = registration_result.registered_series
    registered_volume = sitk.GetArrayFromImage(registered_series).astype(np.float32)
    image_series_xyt = registered_volume.transpose(1, 2, 0)
    print(
        "Registration output:",
        f"shape={registered_volume.shape},",
        f"dtype={registered_volume.dtype}",
    )
    # Segmentation
    mean_image, lung_mask = compute_masks_and_mean(image_series_xyt, config)
    # Timing
    time_step = mean_step_size(time_array)
    print("Estimated time step (s):", time_step)

    # Fourier decomposition
    if config.spectral_method == "FD":
        (
            vent_map,
            perf_map,
            vent_hz,
            perf_hz,
            masked_dc,
            spectrum_freq,
            spectrum_power,
        ) = run_fourier(image_series_xyt, lung_mask, time_step, config=config)

    # Dynamic Mode Decomposition
    elif config.spectral_method == "DMD":
        masked_dc, vent_map, perf_map, vent_hz, perf_hz, phi, freq, b, r, lambda_ = (
            run_dmd(registered_volume, lung_mask, time_step, config=config)
        )

    else:
        raise ValueError(f"Unsupported spectral method: {config.spectral_method}")

    return PipelineResult(
        # ----------------------------------------------------------
        # Input
        # ----------------------------------------------------------
        moving_series=moving_series,
        # ----------------------------------------------------------
        # Registration
        # ----------------------------------------------------------
        registered_series=registered_series,
        registered_volume=registered_volume,
        image_series_xyt=image_series_xyt,
        reference_image=registration_result.reference_image,
        reference_index=registration_result.reference_index,
        fixed_lung_mask=registration_result.fixed_lung_mask,
        registered_lung_mask=registration_result.registered_lung_mask,
        disp_fields=registration_result.disp_fields,
        # ----------------------------------------------------------
        # Segmentation
        # ----------------------------------------------------------
        mean_image=mean_image,
        lung_mask=lung_mask,
        # ----------------------------------------------------------
        # Spectral analysis
        # ----------------------------------------------------------
        vent_map=vent_map,
        perf_map=perf_map,
        vent_hz=vent_hz,
        perf_hz=perf_hz,
        masked_dc=masked_dc,
        time_step=time_step,
        spectrum_freq=spectrum_freq,
        spectrum_power=spectrum_power,
        # ----------------------------------------------------------
        # DMD
        # ----------------------------------------------------------
        phi=phi,
        freq=freq,
        b=b,
        r=r,
        lambda_=lambda_,
    )




def main(config: PipelineConfig, base_dir=None):
    """Run the ventilation and perfusion processing pipeline."""
    paths = setup_paths(config.series_indicator, base_dir)

    print("Reading moving series from", paths["input_path"])
    moving_series = read_images_from_folder(paths["input_path"])

    print("Getting DICOM acquisition times...")
    time_array = get_dicom_acquisition_times(paths["input_path"])

    result = compute_ventilation_perfusion(
        moving_series,
        paths["registration_parameter_file"],
        time_array,
        config,
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

    # save results for potential further analysis
    # Save results for potential further analysis (disabled by default).
    # Example path construction (commented out to avoid accidental writes):
    # results_dir = (
    #     r"C:\\Lung_Project\\PostProcessing\\ventilation_and_perfusion_maps"
    #     r"\\ventilation_and_perfusion_maps\\Results"
    # )
    # results_path = os.path.join(results_dir, config.series_indicator + "_results.npz")
    # np.savez(results_path, **asdict(result))
    return result


def vq_mapping_online(
    data, head, base_dir=None, config: PipelineConfig = PipelineConfig()
):
    """Process data in OpenRecon-style format: inputs are "data, head".

    This mirrors the offline "main()" pipeline but accepts an in-memory
    image array and ISMRMRD header.
    """

    paths = setup_paths(config.series_indicator, base_dir=base_dir)

    print("Converting input array to SimpleITK image series...")
    moving_series = array_to_sitk(data)

    print("Getting ISMRMRD acquisition times...")
    time_array = get_ismrmrd_acquisition_times(head)

    result = compute_ventilation_perfusion(
        moving_series, paths["registration_parameter_file"], time_array, config
    )

    # transforming data for display on scanner console, scaling to 0-255 and converting to uint16

    # setting scaling factor to 95th percentile to avoid outliers dominating the scaling
    p = 0.95
    v_map = result.vent_map / np.percentile(result.vent_map[result.lung_mask], p * 100)
    q_map = result.perf_map / np.percentile(result.perf_map[result.lung_mask], p * 100)
    v_map[v_map > 1] = 1
    q_map[q_map > 1] = 1

    print("Checking shapes before stacking:", v_map.shape, q_map.shape)

    vq_maps = np.stack((v_map, q_map), axis=-1)
    vq_maps *= 255
    print("VentilationChecking max and mean values", np.max(v_map), np.mean(v_map))
    print("Perfusion Checking max and mean values", np.max(q_map), np.mean(q_map))
    vq_maps = vq_maps.astype(np.uint16)

    return vq_maps



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="Volunteer3")
    parser.add_argument("--segmentation-method", default="nnunet")
    parser.add_argument("--registration-method", default="voxelmorph")
    parser.add_argument(
        "--voxelmorph_checkpoint", default=PipelineConfig.voxelmorph_checkpoint
    )
    parser.add_argument("--voxelmorph-config", default=PipelineConfig.voxelmorph_config)
    parser.add_argument(
        "--hypermorph-checkpoint", default=PipelineConfig.hypermorph_checkpoint
    )
    parser.add_argument("--hypermorph-config", default=PipelineConfig.hypermorph_config)
    parser.add_argument(
        "--hypermorph-lambda",
        type=float,
        default=PipelineConfig.hypermorph_lambda,
        help="Fixed HyperMorph lambda (regularization) hyperparameter, passed to register().",
    )
    parser.add_argument(
        "--hypermorph-gamma",
        type=float,
        default=PipelineConfig.hypermorph_gamma,
        help="Fixed HyperMorph gamma (segmentation) hyperparameter, passed to register().",
    )
    parser.add_argument("--spectral-method", default="FD")
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--plotting", default=True)
    parser.add_argument("--phantom", default=False)
    args = parser.parse_args()
    pipeline_config = PipelineConfig(
        series_indicator=args.series,
        segmentation_method=args.segmentation_method,
        registration_method=args.registration_method,
        voxelmorph_checkpoint=args.voxelmorph_checkpoint,
        voxelmorph_config=args.voxelmorph_config,
        hypermorph_checkpoint=args.hypermorph_checkpoint,
        hypermorph_config=args.hypermorph_config,
        hypermorph_lambda=args.hypermorph_lambda,
        hypermorph_gamma=args.hypermorph_gamma,
        spectral_method=args.spectral_method,
        plotting=args.plotting,
        phantom=args.phantom,
        verbose=args.verbose,
    )

    main(pipeline_config)
