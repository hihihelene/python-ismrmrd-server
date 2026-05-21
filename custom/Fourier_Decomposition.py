import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap


def fourier_decomp(V, dt, bw=None, prominence= None,
                   vent_range=(0.1, 0.7), perf_range=(0.8, 2.0), phantom=False):
    """
    Performs Fourier Decomposition method on three-dimensional volume V (2D+time).
    
    Args:
        V: numpy array, 3D volume data (2D spatial + time).
        bw: optional, segmentation containing the lung (usually not needed).
    
    Returns:
        Im1: Ventilation image.
        Im2: Perfusion image.
        Im0: Zero-frequency image (mean image) used for quantification.
        V1: Ventilation time series.
        V2: Perfusion time series.
    """
    # Find the peaks. Note, this does not always work 100% automatically!
    if phantom:
        pos, vent_hz, perf_hz = find_local_max(V, dt, bw, prominence=prominence,
                                               vent_range=vent_range, perf_range=None)
    else:
        pos, vent_hz, perf_hz  = find_local_max(V, dt, bw, prominence=prominence,
                                               vent_range=vent_range, perf_range=perf_range)
    

    Sig = signal(V)  # Reshapes time-volume into time-signal, i.e. vol(x,y,t) into Sig(p,t)
    L1 = np.zeros_like(Sig, dtype=complex)
    L2 = np.zeros_like(Sig, dtype=complex)
    L0 = np.zeros_like(Sig, dtype=complex)
    z = V.shape[2]

    q = np.fft.fft(Sig, axis=1)  # Fourier transform along time dimension

    # Extract values at calculated frequency. Multiply by two later due to symmetry of Fourier space.
    L1[:, pos[0]:pos[1]] = q[:, pos[0]:pos[1]]
    
    # Only fill L2 when perfusion bin indices were provided
    if pos[2] is not None and pos[3] is not None:
        L2[:, pos[2]:pos[3]] = q[:, pos[2]:pos[3]]
    L0[:, 0] = q[:, 0]

    # Multiply by 2 due to symmetry of Fourier space.
    Sig1 = 2 * np.sum(np.abs(L1), axis=1)  # Pulmonary
    Sig2 = 2 * np.sum(np.abs(L2), axis=1)  # Cardiac
    Sig0 = np.sum(np.abs(L0), axis=1)  # Zero-frequency

    V1 = np.abs(np.fft.ifft(L1, axis=1))  # Pulmonary
    # If phantom mode, L2 remains zero and V2 will be zero volume
    V2 = np.abs(np.fft.ifft(L2, axis=1))  # Cardiac

    Siz = V[:, :, 0].shape

    # Reshape into images
    Im1 = Sig1.reshape(Siz)
    Im2 = Sig2.reshape(Siz)
    Im0 = Sig0.reshape(Siz)

    # Reshape into volumes
    V1 = V1.reshape(Siz[0], Siz[1], z)
    V2 = V2.reshape(Siz[0], Siz[1], z)

    return Im1, Im2, Im0, V1, V2, vent_hz, perf_hz 


def find_local_max(V, dt, bw=None, prominence=None,
                   vent_range=(0.1, 0.7), perf_range=(0.8, 2.0),
                   spectrum=None, freqs=None):
    """
    Find two maxima in the mean-signal power spectrum — one in `vent_range`
    and one in `perf_range`.

    Args:
        V, dt, bw, prominence: as before.
        vent_range: tuple(low, high) in Hz where ventilation peak is expected.
        perf_range: tuple(low, high) in Hz where perfusion (cardiac) peak is expected.
    """
    # 1) Use precomputed spectrum if provided; otherwise compute from V
    if spectrum is not None:
        p = spectrum
        if freqs is None:
            if V is None:
                raise ValueError("Either freqs or V must be provided with spectrum")
            N_full = V.shape[2]
            freqs = np.arange(p.size) / (N_full * dt)
    else:
        Sig = signal(V)
        if bw is not None:
            mask = bw.reshape(-1)
            pix  = np.where(mask)[0]
            m    = np.mean(Sig[pix], axis=0)
        else:
            m = np.mean(Sig, axis=0)

        # 2) Compute power spectrum
        p = breathing_cycle(m, dt)        # length = N_full//2
        N_full = V.shape[2]
        freqs  = np.arange(p.size) / (N_full * dt)

    # 3) Restrict detection to the specified frequency intervals

    # Convert frequency ranges to bin indices (clamp to valid range)
    if freqs is not None:
        v_low = int(np.searchsorted(freqs, vent_range[0], side='left'))
        v_high = int(np.searchsorted(freqs, vent_range[1], side='right') - 1)
        v_low = max(v_low, 1)
        v_high = min(v_high, p.size - 1)
        if perf_range is not None:
            pf_low = int(np.searchsorted(freqs, perf_range[0], side='left'))
            pf_high = int(np.searchsorted(freqs, perf_range[1], side='right') - 1)
            pf_low = max(pf_low, 1)
            pf_high = min(pf_high, p.size - 1)
        else:
            pf_low = None
            pf_high = None
    else:
        v_low = max(int(np.floor(vent_range[0] * N_full * dt)), 1)
        v_high = min(int(np.ceil(vent_range[1] * N_full * dt)), p.size - 1)
        if perf_range is not None:
            pf_low = max(int(np.floor(perf_range[0] * N_full * dt)), 1)
            pf_high = min(int(np.ceil(perf_range[1] * N_full * dt)), p.size - 1)
        else:
            pf_low = None
            pf_high = None

    # Extract spectrum segments for each physiological band
    v_segment = p[v_low:v_high + 1]

    pf_segment = None if pf_low is None else p[pf_low:pf_high + 1]

    # Find peaks only within each segment
    v_peaks, v_props = find_peaks(v_segment, prominence=prominence)
    if v_peaks.size > 0:
        v_candidates = v_low + v_peaks
        vent_bin = int(v_candidates[np.argmax(p[v_candidates])])
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
            perf_bin = int(pf_candidates[np.argmax(p[pf_candidates])])
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
            order = np.argsort(p[pf_low:pf_high + 1])[::-1]
            for idx in order:
                candidate = pf_low + int(idx)
                if candidate != vent_bin:
                    perf_bin = candidate
                    break
        else:
            # fallback to global best excluding vent_bin
            all_bins = np.arange(1, p.size)
            other_bins = all_bins[all_bins != vent_bin]
            if other_bins.size > 0:
                perf_bin = int(other_bins[np.argmax(p[other_bins])])

    vent_hz = freqs[vent_bin]
    perf_hz = None if perf_bin is None else freqs[perf_bin]
    print(f"ventilation frequency: {vent_hz:.2f} Hz")
    if perf_hz is None:
        print("perfusion frequency:  None")
    else:
        print(f"perfusion frequency:  {perf_hz:.2f} Hz")

    # 6) Build pos: take each bin and its neighbor
    pos = [vent_bin,     min(vent_bin + 1, p.size - 1),
        (None if perf_bin is None else perf_bin),
        (None if perf_bin is None else min(perf_bin + 1, p.size - 1))]
    pos[0] = max(pos[0], 1)
    return pos, vent_hz, perf_hz

def signal(V):
    """
    Reshapes time-volume into time-signal.
    
    Args:
        V: numpy array, volume data.
    
    Returns:
        Sig: reshaped signal data.
    """
    return V.reshape(-1, V.shape[-1])

def breathing_cycle(f, dt):
    """
    Transforms f into frequency space to display frequency and amplitude.

    Args:
        f: numpy array, mean signal.

    Returns:
        f: transformed frequency power spectrum.
    """
    N = len(f)
    T = N * dt                  # total duration

    # Apply Hann window
    h = np.hanning(N)
    print('Hann window applied!')
    f = f * h

    # Compute FFT and take the absolute value
    p = np.abs(np.fft.fft(f)) / (N / 2)
    p = p[:N // 2] ** 2  # Take the power of the positive frequency half

    # Find the corresponding frequency in Hz
    freq = np.arange(N // 2) / T
    p[0] = 0
    p[1] = 0

        
    return p

def frequency_spectrum_plot(V, dt, bw=None, output_path=None,
                            fd_output=None,
                            vent_hz=None, perf_hz=None,
                            prominence=None,
                            correct_hann_gain=True,
                            vent_range=None, perf_range=None):
    """
    Plot the single-sided amplitude spectrum of the mean time-series in V.
    If vent_hz/perf_hz are not provided, auto-detect the two dominant peaks.
    If fd_output is provided (output from fourier_decomp), reuse its
    vent/perf frequencies when available.

    Returns:
        vent_hz, perf_hz  (floats in Hz)

    """
    # 1) Mean time-series m(t)
    Sig = V.reshape(-1, V.shape[-1])
    if bw is not None:
        mask = bw.reshape(-1)
        m = np.mean(Sig[mask > 0], axis=0)
    else:
        m = np.mean(Sig, axis=0)

    N = m.size
    print(f'Mean time-series length: {N} samples, duration: {N*dt:.2f} seconds') # debugging
  
    # 2) Hann window (+ optional coherent-gain correction)
    h = np.hanning(N)
    m_win = m * h
    cg = (h.sum() / N) if correct_hann_gain else 1.0  # coherent gain

    # 3) rFFT + single-sided amplitude scaling
    P = np.fft.rfft(m_win)
    # base amplitude (no factor 2 yet), with Hann gain correction
    amp = np.abs(P) / (N * cg)
    # double non-DC and non-Nyquist bins to get single-sided amplitude
    if amp.size > 2:
        amp[1:-1] *= 2.0

    # 4) Frequencies for rfft
    freq = np.fft.rfftfreq(N, d=dt)

    # Zero out DC (and optionally the first bin)
    amp[0] = 0.0
    if amp.size > 1:
        amp[1] = 0.0

    # 5) Use provided fourier_decomp output if available
    if fd_output is not None and (vent_hz is None or perf_hz is None):
        try:
            vent_hz = vent_hz if vent_hz is not None else fd_output[5]
            perf_hz = perf_hz if perf_hz is not None else fd_output[6]
        except (TypeError, IndexError):
            pass

    # 6) Peak detection only if not provided — use the same logic as find_local_max
    if vent_hz is None or perf_hz is None:
        try:
            # Call find_local_max with provided ranges when available so both
            # functions select identical peaks.
            if vent_range is None and perf_range is None:
                _, vent_hz, perf_hz = find_local_max(V, dt, bw=bw, prominence=prominence,
                                                     spectrum=amp, freqs=freq)
            elif vent_range is None:
                _, vent_hz, perf_hz = find_local_max(V, dt, bw=bw, prominence=prominence,
                                                     perf_range=perf_range,
                                                     spectrum=amp, freqs=freq)
            elif perf_range is None:
                _, vent_hz, perf_hz = find_local_max(V, dt, bw=bw, prominence=prominence,
                                                     vent_range=vent_range,
                                                     spectrum=amp, freqs=freq)
            else:
                _, vent_hz, perf_hz = find_local_max(V, dt, bw=bw, prominence=prominence,
                                                     vent_range=vent_range, perf_range=perf_range,
                                                     spectrum=amp, freqs=freq)
        except RuntimeError:
            # Fallback: pick top-2 by amplitude from the already-computed 'amp'
            peaks = np.argsort(amp)[-2:]
            vbin, pbin = sorted(peaks, key=lambda b: freq[b])
            vent_hz, perf_hz = freq[vbin], freq[pbin]
    # mark peaks
    # find the nearest frequency bins for markers
    def nearest_bin(f):
        return int(np.argmin(np.abs(freq - f)))
    

    vbin = nearest_bin(vent_hz)

    # 7) Plot
    plt.figure(figsize=(6, 4))
    plt.plot(freq, amp, linewidth=2)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Mean-signal spectrum')
    plt.scatter([freq[vbin]], [amp[vbin]])
    plt.axvline(freq[vbin], linestyle='--', alpha=0.7)
    plt.annotate(f'Vent {vent_hz:.2f} Hz', (freq[vbin], amp[vbin]),
                textcoords='offset points', xytext=(8, 8))

    # Plot perfusion marker only if available
    if perf_hz is not None:
        pbin = nearest_bin(perf_hz)
        plt.scatter([freq[pbin]], [amp[pbin]])
        plt.axvline(freq[pbin], linestyle='--', alpha=0.7)
        plt.annotate(f'Perf {perf_hz:.2f} Hz', (freq[pbin], amp[pbin]),
                    textcoords='offset points', xytext=(8, 8))

    plt.xlim(0, freq.max())
    plt.ylim(0, amp.max() * 1.1 if amp.size else 1.0)
    plt.tight_layout()

    plt.show() # (block=True)

    # 7) Save (accept either a directory or a full filename)
    if output_path:
        if output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            out_file = output_path
            dirpath = os.path.dirname(out_file)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)
            out_file = os.path.join(output_path, 'frequency_spectrum.jpg')
        plt.savefig(out_file, bbox_inches='tight', dpi=200)
    return vent_hz, perf_hz