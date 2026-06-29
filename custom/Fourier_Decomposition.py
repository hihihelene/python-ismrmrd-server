import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
# from matplotlib.colors import LinearSegmentedColormap


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
    V=None,
    dt=None,
    bw=None,
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
        V, dt, bw, prominence: as before.
        vent_range: tuple(low, high) in Hz where ventilation peak is expected.
        perf_range: tuple(low, high) in Hz where perfusion (cardiac) peak is expected.
    """
    if spectrum is None or freqs is None:
        if V is None:
            raise ValueError(
                "Either (V, dt) or (spectrum, freqs) must be provided."
            )

        freqs, spectrum = compute_mean_power_spectrum(
            V,
            dt,
            bw,
        )
        nx, ny, z = V.shape

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
        v_low = max(int(np.floor(vent_range[0] * z * dt)), 1)
        v_high = min(int(np.ceil(vent_range[1] * z * dt)), power.size - 1)
        if perf_range is not None:
            pf_low = max(int(np.floor(perf_range[0] * z * dt)), 1)
            pf_high = min(int(np.ceil(perf_range[1] * z  * dt)), power.size - 1)
        else:
            pf_low = None
            pf_high = None

    # Extract spectrum segments for each physiological band
    v_segment = power[v_low:v_high + 1]

    pf_segment = None if pf_low is None else power[pf_low:pf_high + 1]

    # Find peaks only within each segment
    v_peaks, v_props = find_peaks(v_segment, prominence=prominence)
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
        pf_peaks, pf_props = find_peaks(pf_segment, prominence=prominence)
        if pf_peaks.size > 0:
            pf_candidates = pf_low + pf_peaks
            perf_bin = int(pf_candidates[np.argmax(power[pf_candidates])])
        else:
            if pf_segment.size == 0:
                raise RuntimeError("Perfusion range is empty or out of bounds")
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

    vent_hz = freqs[vent_bin]
    perf_hz = None if perf_bin is None else freqs[perf_bin]

    # 6) Build pos: take each bin and its neighbor
    pos = [vent_bin,     min(vent_bin + 1, power.size - 1),
        (None if perf_bin is None else perf_bin),
        (None if perf_bin is None else min(perf_bin + 1, power.size - 1))]
    pos[0] = max(pos[0], 1)
    return pos, vent_hz, perf_hz


def fourier_decomp(V, dt, bw=None, prominence= None,
                   vent_range=(0.1, 0.7), perf_range=(0.8, 2.0), phantom=False):
    """
    Performs Fourier Decomposition method on three-dimensional volume V (2D+time).
    
    Args:
        V: numpy array, 3D volume data (2D spatial + time).
        bw: optional, segmentation containing the lung (usually not needed).
    
    Returns:
        Im_Vent: Ventilation image.
        Im_Perf: Perfusion image.
        Im0: Zero-frequency image (mean image) used for quantification.
        V_Vent: Ventilation time series.
        V_Perf: Perfusion time series.
    """
    
    # Find the peaks. Note, this does not always work 100% automatically!
    peak_perf_range = None if phantom else perf_range

    spectrum_freq, spectrum_power = compute_mean_power_spectrum(
        V,
        dt,
        bw,
    )

    pos, vent_hz, perf_hz = find_local_max(
        spectrum=spectrum_power,
        freqs=spectrum_freq,
        prominence=prominence,
        vent_range=vent_range,
        perf_range=peak_perf_range,
    )

    if phantom: perf_hz = None # throw out perfusion frequency for phantom data, since it is not meaningful

    print(f"ventilation frequency: {vent_hz:.2f} Hz")

    if perf_hz is None:
        print("perfusion frequency:  None")
    else:
        print(f"perfusion frequency:  {perf_hz:.2f} Hz")
    voxel_signals =  V.reshape(-1, V.shape[-1])  # Reshapes time-volume into time-signal, i.e. vol(x,y,t) into Sig(p,t)
    nx, ny, z = V.shape

    fft_data = np.fft.fft(voxel_signals, axis=1)  # Fourier transform along time dimension

    def build_maps(current_pos):
        vent_fft = np.zeros_like(voxel_signals, dtype=complex)
        dc_fft = np.zeros_like(voxel_signals, dtype=complex)

        # Ventilation
        vent_fft[:, current_pos[0]:current_pos[1]] = \
            fft_data[:, current_pos[0]:current_pos[1]]

        # DC
        dc_fft[:, 0] = fft_data[:, 0]

        Sig1 = 2 * np.sum(np.abs(vent_fft), axis=1)
        Sig0 = np.sum(np.abs(dc_fft), axis=1)

        V_vent_local = np.abs(np.fft.ifft(vent_fft, axis=1))

        # Perfusion (optional)
        if current_pos[2] is None:
            Sig2 = None
            V_perf_local = None
        else:
            perf_fft = np.zeros_like(voxel_signals, dtype=complex)
            perf_fft[:, current_pos[2]:current_pos[3]] = \
                fft_data[:, current_pos[2]:current_pos[3]]

            Sig2 = 2 * np.sum(np.abs(perf_fft), axis=1)
            V_perf_local = np.abs(np.fft.ifft(perf_fft, axis=1))

        return Sig1, Sig2, Sig0, V_vent_local, V_perf_local

    Sig1, Sig2, Sig0, V_vent, V_Perf = build_maps(pos)


    # Reshape into images
    Sig1, Sig2, Sig0, V_vent, V_Perf = build_maps(pos)

    Im1 = Sig1.reshape(nx, ny)
    Im0 = Sig0.reshape(nx, ny)

    if Sig2 is None:
        Im2 = None
    else:
        Im2 = Sig2.reshape(nx, ny)

    V_vent = V_vent.reshape(nx, ny, z)

    if V_Perf is not None:
        V_Perf = V_Perf.reshape(nx, ny, z)
    # Recompute ventilation peak using a mask that excludes the brightest
    # perfusion pixels, so perfusion does not bias the vent-frequency search.

    if not phantom:
        if bw is not None:
            vent_mask = np.asarray(bw, dtype=bool)
        else:
            vent_mask = np.isfinite(Im2) & (Im2 > 0)

        vent_mask = vent_mask & np.isfinite(Im2)
        vent_values = Im2[vent_mask]

        if vent_values.size > 0:
            perf_cutoff = np.nanpercentile(vent_values, 90)
            exclude_mask = vent_mask & (Im2 >= perf_cutoff)
            vent_mask = vent_mask & (Im2 < perf_cutoff)

            vent_pos, vent_hz, _ = find_local_max(
                V,
                dt,
                bw=vent_mask,
                prominence=prominence,
                vent_range=vent_range,
                perf_range=None,
            )

            Sig1, _, _, V_vent, _ = build_maps(vent_pos)
            Im1 = Sig1.reshape([nx, ny])
            V_vent = V_vent.reshape(nx, ny, z)

            # Explicitly exclude the top perfusion pixels from the vent map by
            # setting them to NaN so downstream percentile/normalization ignores them.
            Im1 = np.array(Im1, copy=True)
            V_vent = np.array(V_vent, copy=True)
            Im1[exclude_mask] = 0
            V_vent[exclude_mask, :] = 0

    return Im1,Im2,Im0,V_vent,V_Perf,vent_hz,perf_hz, spectrum_freq, spectrum_power
