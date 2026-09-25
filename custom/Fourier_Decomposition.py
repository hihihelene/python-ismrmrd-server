import warnings

import numpy as np
from scipy.signal import find_peaks


def compute_mean_power_spectrum(V, dt, bw=None):
    """
    Compute the mean-signal power spectrum used for peak detection.

    Returns
    -------
    freqs : ndarray
        Frequency axis [Hz]
    power : ndarray
        Single-sided power spectrum
    """
    voxel_signals = V.reshape(-1, V.shape[-1])

    if bw is not None:
        mean_signal = np.mean(
            voxel_signals[bw.reshape(-1) > 0],
            axis=0
        )
    else:
        mean_signal = np.mean(voxel_signals, axis=0)

    n_timepoints = len(mean_signal)
    duration = n_timepoints * dt

    # Hann window
    window = np.hanning(n_timepoints)
    mean_signal = mean_signal * window

    # FFT
    fft_data = np.fft.fft(mean_signal)

    # Power spectrum
    power = np.abs(fft_data) / (n_timepoints / 2)
    power = power[: n_timepoints // 2] ** 2

    freqs = np.arange(power.size) / duration

    power[0] = 0
    if power.size > 1:
        power[1] = 0

    return freqs, power


def find_local_max(
    time_series_volume=None,
    time_step=None,
    mask=None,
    prominence=None,
    vent_range=(0.1, 0.7),
    perf_range=(0.8, 2.0),
    spectrum=None,
    freqs=None,
):
    """
    Find two maxima in the mean-signal power spectrum — one in `vent_range`
    and one in `perf_range`.

    Args:
        time_series_volume, time_step, mask, prominence: as before.
        vent_range: tuple(low, high) in Hz where ventilation (respiratory)peak is expected.
        perf_range: tuple(low, high) in Hz where perfusion (cardiac) peak is expected.
    """
    # ensure z is defined 
    z = None

    if spectrum is None or freqs is None:
        if time_series_volume is None:
            raise ValueError(
                "Either (time_series_volume, time_step) or (spectrum, freqs) must be provided."
            )

        freqs, spectrum = compute_mean_power_spectrum(
            time_series_volume,
            time_step,
            mask,
        )
        _nx, _ny, z = time_series_volume.shape

    power = spectrum

    # Restrict detection to the specified frequency intervals

    # Convert frequency ranges to bin indices (clamp to valid range)
    if freqs is not None:
        v_low = int(np.searchsorted(freqs, vent_range[0], side='left'))
        v_high = int(np.searchsorted(freqs, vent_range[1], side='right') - 1)
        v_low = max(v_low, 1)
        v_high = min(v_high, power.size - 1)
        if perf_range is not None:
            pf_low = int(np.searchsorted(freqs, perf_range[0], side='left'))
            pf_high = int(np.searchsorted(freqs, perf_range[1], side='right') - 1)
            pf_low = max(pf_low, 1)
            pf_high = min(pf_high, power.size - 1)
        else:
            pf_low = None
            pf_high = None
    else:
        v_low = max(int(np.floor(vent_range[0] * z * time_step)), 1)
        v_high = min(int(np.ceil(vent_range[1] * z * time_step)), power.size - 1)
        if perf_range is not None:
            pf_low = max(int(np.floor(perf_range[0] * z * time_step)), 1)
            pf_high = min(int(np.ceil(perf_range[1] * z  * time_step)), power.size - 1)
        else:
            pf_low = None
            pf_high = None

    # Extract spectrum segments for each physiological band
    v_segment = power[v_low:v_high + 1]

    pf_segment = None if pf_low is None else power[pf_low:pf_high + 1]

    if v_segment.size == 0:
        warnings.warn(
            "No ventilation frequency was found in the requested frequency range.",
            RuntimeWarning,
            stacklevel=2,
        )
        if pf_segment is not None and pf_segment.size == 0:
            warnings.warn(
                "No perfusion frequency was found in the requested frequency range.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [None, None, None, None], None, None

    # Find peaks only within each segment
    v_peaks, _v_props = find_peaks(v_segment, prominence=prominence)
    if v_peaks.size > 0:
        v_candidates = v_low + v_peaks
        vent_bin = int(v_candidates[np.argmax(power [v_candidates])])
    else:
        # no peak detected by find_peaks: choose the maximum bin in the interval
        if v_segment.size == 0:
            raise RuntimeError("Ventilation range is empty or out of bounds")
        vent_bin = int(v_low + np.argmax(v_segment))

    perf_bin = None
    if pf_segment is not None:
        pf_peaks, _pf_props = find_peaks(pf_segment, prominence=prominence)
        if pf_peaks.size > 0:
            pf_candidates = pf_low + pf_peaks
            perf_bin = int(pf_candidates[np.argmax(power[pf_candidates])])
        else:
            if pf_segment.size > 0:
                perf_bin = int(pf_low + np.argmax(pf_segment))

    # If both resolved to the same bin (possible if ranges overlap), pick
    # the next-best in the perfusion interval if available, otherwise try
    # the next-largest global peak excluding vent_bin.
    if perf_bin is not None and vent_bin == perf_bin:
        # try next-best within perfusion segment
        if pf_segment.size > 1:
            # rank bins in perf segment by power
            order = np.argsort(power[pf_low:pf_high + 1])[::-1]
            for idx in order:
                candidate = pf_low + int(idx)
                if candidate != vent_bin:
                    perf_bin = candidate
                    break
        else:
            # fallback to global best excluding vent_bin
            all_bins = np.arange(1, power.size)
            other_bins = all_bins[all_bins != vent_bin]
            if other_bins.size > 0:
                perf_bin = int(other_bins[np.argmax(power[other_bins])])

    vent_freq = freqs[vent_bin]
    perf_freq = None if perf_bin is None else freqs[perf_bin]
    if perf_range is not None and perf_freq is None:
        warnings.warn(
            "No perfusion frequency was found in the requested frequency range.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 6) Build pos: take each bin and its neighbor
    pos = [vent_bin,     min(vent_bin + 1, power.size - 1),
        (None if perf_bin is None else perf_bin),
        (None if perf_bin is None else min(perf_bin + 1, power.size - 1))]
    if pos[0] is not None:
        pos[0] = max(pos[0], 1)
    return pos, vent_freq, perf_freq


def fourier_decomp(time_series_volume, time_step, mask=None, prominence= None,
                   vent_range=(0.1, 0.7), perf_range=(0.8, 2.0), phantom=False):
    """
    Performs Fourier Decomposition method on three-dimensional volume V (2D+time).
    
    Args:
        time_series_volume: numpy array, 3D volume data (2D spatial + time).
        time_step: Time step between frames.
        mask: optional, segmentation containing the region of interest (usually not needed).
        prominence: Prominence threshold for peak detection.
        vent_range: Range for ventilation frequency detection.
        perf_range: Range for perfusion frequency detection.
        phantom: Flag indicating if the data is from a phantom.

    Returns:
        vent_image: Ventilation image.
        perf_image: Perfusion image.
        dc_image: Zero-frequency image (mean image) used for quantification.
        ventilation_series: Ventilation time series.
        perfusion_series: Perfusion time series.
        vent_freq: Detected ventilation frequency.
        perf_freq: Detected perfusion frequency.
        spectrum_freq: Frequency axis of the power spectrum.
        spectrum_power: Power spectrum of the mean signal.
    """

    # Find the peaks. Note, this does not always work 100% automatically!
    peak_perf_range = None if phantom else perf_range

    spectrum_freq, spectrum_power = compute_mean_power_spectrum(
        time_series_volume,
        time_step,
        mask,
    )

    pos, vent_freq, perf_freq = find_local_max(
        spectrum=spectrum_power,
        freqs=spectrum_freq,
        prominence=prominence,
        vent_range=vent_range,
        perf_range=peak_perf_range,
    )

    if phantom: 
        # throw out perfusion frequency for phantom data, since it is not meaningful
        perf_freq = None 

    if vent_freq is None:
        print("ventilation frequency:  None")
    else:
        print(f"ventilation frequency: {vent_freq:.2f} Hz")

    if perf_freq is None:
        print("perfusion frequency:  None")
    else:
        print(f"perfusion frequency:  {perf_freq:.2f} Hz")

    voxel_signals =  time_series_volume.reshape(-1, time_series_volume.shape[-1])  # Reshapes time-volume into time-signal, i.e. vol(x,y,t) into Sig(p,t)
    nx, ny, z = time_series_volume.shape

    fft_data = np.fft.fft(voxel_signals, axis=1)  # Fourier transform along time dimension

    def build_maps(current_pos, report_fraction=True):
        vent_fft = np.zeros_like(voxel_signals, dtype=complex)
        dc_fft = np.zeros_like(voxel_signals, dtype=complex)

        # DC
        dc_fft[:, 0] = fft_data[:, 0]

        dc_amplitude = np.sum(np.abs(dc_fft), axis=1)
        ventilation_amplitude = np.zeros_like(dc_amplitude)
        ventilation_series_local = np.zeros_like(voxel_signals, dtype=float)
        if current_pos[0] is not None:
            vent_fft[:, current_pos[0]:current_pos[1]] = \
                fft_data[:, current_pos[0]:current_pos[1]]
            ventilation_amplitude = 2 * np.sum(np.abs(vent_fft), axis=1)
            ventilation_series_local = np.abs(np.fft.ifft(vent_fft, axis=1))

        # Kjorstad normalization
        a = 2 * ventilation_amplitude
        b = dc_amplitude + ventilation_amplitude

        ventilation_fraction = np.divide(
            a,
            b,
            out=np.zeros_like(a, dtype=float),
            where=b != 0
        )

        if report_fraction:
            print(f"Ventilation fraction: {np.nanmean(ventilation_fraction):.2f}")

        # Perfusion (optional)
        if current_pos[2] is None:
            perfusion_amplitude = None
            perfusion_series_local = None
        else:
            perf_fft = np.zeros_like(voxel_signals, dtype=complex)
            perf_fft[:, current_pos[2]:current_pos[3]] = \
                fft_data[:, current_pos[2]:current_pos[3]]

            perfusion_amplitude = 2 * np.sum(np.abs(perf_fft), axis=1)
            perfusion_series_local = np.abs(np.fft.ifft(perf_fft, axis=1))

        return ventilation_fraction, perfusion_amplitude, dc_amplitude, ventilation_series_local, perfusion_series_local

    # Reshape into images
    ventilation_fraction, perfusion_amplitude, dc_amplitude, ventilation_series, perfusion_series = build_maps(pos)

    vent_image = ventilation_fraction.reshape(nx, ny)
    dc_image = dc_amplitude.reshape(nx, ny)

    perf_image = (
        None
        if perfusion_amplitude is None
        else perfusion_amplitude.reshape(nx, ny)
    )

    ventilation_series = ventilation_series.reshape(nx, ny, z)

    if perfusion_series is not None:
        perfusion_series = perfusion_series.reshape(nx, ny, z)

    # Recompute ventilation peak using a mask that excludes the brightest
    # perfusion pixels, so perfusion does not bias the vent-frequency search.

    if not phantom:
        if mask is not None:
            vent_mask = np.asarray(mask, dtype=bool)
        else:
            vent_mask = np.isfinite(vent_image) & (vent_image > 0)

        vent_mask = vent_mask & np.isfinite(vent_image)
        ventilation_amplitude_image = vent_image * dc_image
        vent_values = ventilation_amplitude_image[vent_mask]

        if vent_values.size > 0:
            if perf_image is not None:
                perf_values = perf_image[vent_mask]
                perf_values = perf_values[np.isfinite(perf_values)]
            else:
                perf_values = np.array([])

            if perf_values.size > 0:
                ventilation_cutoff = np.nanpercentile(vent_values, 90)
                exclude_mask = vent_mask & (perf_image >= ventilation_cutoff)
                vent_mask = vent_mask & (perf_image < ventilation_cutoff)
            else:
                exclude_mask = np.zeros_like(vent_mask, dtype=bool)

            if vent_freq is not None:
                vent_pos, vent_freq, _ = find_local_max(
                    time_series_volume,
                    time_step,
                    mask=vent_mask,
                    prominence=prominence,
                    vent_range=vent_range,
                    perf_range=None,
                )

                ventilation_fraction, _, _, ventilation_series, _ = build_maps(
                    vent_pos,
                    report_fraction=False,
                )
                vent_image = ventilation_fraction.reshape([nx, ny])
                ventilation_series = ventilation_series.reshape(nx, ny, z)

                vent_image = np.array(vent_image, copy=True)
                ventilation_series = np.array(ventilation_series, copy=True)
                vent_image[exclude_mask] = 0
                ventilation_series[exclude_mask, :] = 0

    return vent_image, perf_image, dc_image, ventilation_series, perfusion_series, vent_freq, perf_freq, spectrum_freq, spectrum_power
