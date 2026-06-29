import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_image(ax, data, cmap, title, vmin=None, vmax=None):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def plot_results(image, dc_image, ventMap, perfMap,
                 ventMap_range=None, perfMap_range=None, output_path=None,
                 vent_freqs=None, perf_freqs=None, show_freq_text=True, max_list_items=6, config=None):  
    """
    Displays images for any spectral decomposition technique.

    Parameters:
    image, dc_image : numpy.ndarray
        2D arrays representing the original image and DC component.
    ventMap, perfMap : numpy.ndarray
        2D arrays representing ventilation and perfusion maps.
    technique: str ('DMD' or 'FD'), (other options may be added later)
    """
    ocean_cmap = LinearSegmentedColormap.from_list('ocean', [
        '#000000','#000080', '#0000cd', '#1e90ff', '#00bfff', '#87ceeb',
        '#e0ffff','#ffffff'
    ])
    blackbody_cmap = LinearSegmentedColormap.from_list('blackbody', [
        '#000000','#550000', '#dd0000', '#ff8000', '#ffff80', '#ffffff'
    ])

    def normalize_image(image, upper_percentile=None):
        """Normalize image to 0-1 range with optional upper percentile clipping."""
        img_copy = np.array(image, dtype=float)
        valid_mask = ~np.isnan(img_copy)

        if not np.any(valid_mask):
            return img_copy

        if upper_percentile is not None:
            p_upper = np.nanpercentile(img_copy[valid_mask], upper_percentile)
            img_copy[valid_mask] = np.minimum(img_copy[valid_mask], p_upper)

        vmin = np.nanmin(img_copy[valid_mask])
        vmax = np.nanmax(img_copy[valid_mask])

        if vmax > vmin:
            img_copy[valid_mask] = (img_copy[valid_mask] - vmin) / (vmax - vmin)
        else:
            img_copy[valid_mask] = 0.5

        return img_copy

    image_norm = normalize_image(image)
    dc_image_norm = normalize_image(dc_image)
    ventMap_norm = normalize_image(ventMap, upper_percentile=99)
    perfMap_norm = normalize_image(perfMap, upper_percentile=99) if perfMap is not None else None

    # # Compute P95 only in segmented tissue and ignore background.
    # segmented_mask = np.isfinite(perfMap_norm) & (np.asarray(perfMap) > 0) if perfMap is not None else None
    # perf_p95 = np.nanpercentile(perfMap_norm[segmented_mask], 95) if np.any(segmented_mask) else None

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))


    def _fmt_freqs(arr, max_items=6):
        if arr is None or len(arr) == 0:
            return "none"
        arr = np.asarray(arr)
        shown = ", ".join(f"{v:.3f}" for v in arr[:max_items])
        if len(arr) > max_items:
            shown += f", +{len(arr)-max_items} more"
        return shown

    # ranges
    vmin_vent, vmax_vent = (0, 1) if ventMap_range is None else ventMap_range
    vmin_perf, vmax_perf = (0, 1) if perfMap_range is None else perfMap_range

    # plots
    vent_title = 'Fractional Ventilation [ml/ml]' + '\n Technique:' + config.spectral_method 
    perf_title = 'Perfusion [normalized]' + '\n Technique:' + config.spectral_method

    plot_image(axs[0, 0], image_norm, 'gray', 'Phantom [a.u.]',
               0, 1)
    plot_image(axs[0, 1], dc_image_norm, 'gray', 'DC Component [a.u.]',
               0, 1)
    plot_image(axs[1, 0], ventMap_norm, ocean_cmap, vent_title,
               vmin_vent, vmax_vent)
    
    plot_image(axs[1, 1], perfMap_norm, blackbody_cmap, perf_title,
               vmin_perf, vmax_perf) if config.phantom is False else axs[1, 1].set(visible=False)

    # annotate frequencies directly on the vent/perf maps ---
    if show_freq_text and config.spectral_method == 'DMD':
        vent_text = f"Vent freqs (Hz): {_fmt_freqs(vent_freqs, max_list_items)}"
        perf_text = f"Perf freqs (Hz): {_fmt_freqs(perf_freqs, max_list_items)}" if perf_freqs is not None else "Perf freqs (Hz): none"

        # top-left corner inside each axes
        axs[1, 0].text(
            0.02, 0.02, vent_text,
            transform=axs[1, 0].transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=3)
        )

        if config.phantom is False:
            axs[1, 1].text(
                0.02, 0.02, perf_text,
                transform=axs[1, 1].transAxes, ha='left', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=3)
            )
        else:
            axs[1, 1].set(visible=False)
    # ----------------------------------------------------------------

    # # Dedicated perfusion plot with highlighted/outlined 95th percentile region.
    # fig_p95, ax_p95 = plt.subplots(1, 1, figsize=(6, 6))
    # perf_segmented = np.ma.masked_where(~segmented_mask, perfMap_norm)
    # im_p95 = ax_p95.imshow(perf_segmented, cmap=blackbody_cmap, vmin=vmin_perf, vmax=vmax_perf)
    # if perf_p95 is not None:
    #     top95_mask = segmented_mask & (perfMap_norm >= perf_p95)
    #     highlight = np.ma.masked_where(~top95_mask, top95_mask.astype(float))
    #     ax_p95.imshow(highlight, cmap='Greys', alpha=0.25, vmin=0, vmax=1)
    #     ax_p95.contour(top95_mask.astype(float), levels=[0.5], colors='cyan', linewidths=1.8)
    #     title_suffix = f"\nP95 threshold: {perf_p95:.3f}"
    # else:
    #     title_suffix = "\nP95 threshold: N/A"
    #     ax_p95.text(
    #         0.5, 0.5, 'No segmented perfusion pixels found',
    #         transform=ax_p95.transAxes, ha='center', va='center', fontsize=10,
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4)
    #     )
    # ax_p95.set_title('Perfusion map with 95th percentile outline' + title_suffix)
    # ax_p95.axis('off')
    # cbar_p95 = plt.colorbar(im_p95, ax=ax_p95, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar_p95.set_label('Perfusion [normalized]')

    # fig.tight_layout()
    # fig_p95.tight_layout()
    
    if output_path:
        if config.spectral_method == 'FD':
            output_path  = os.path.join(output_path, 'FD.png')
        elif config.spectral_method == 'DMD':
            output_path = os.path.join(output_path, 'DMD.png')
        dirpath = os.path.dirname(output_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')

        # p95_filepath = os.path.join(dirpath if dirpath else '.', f"{config.spectral_method}_perfusion_p95.png")
        # fig_p95.savefig(p95_filepath, bbox_inches='tight')
    


def reconstruct_freq_image(b, res, indices):
    """
    Reconstruct image from specific frequency indices.

    Parameters:
    b : numpy.ndarray
        Amplitudes vector.
    res : numpy.ndarray
        DMD modes or other components.
    indices : list
        Indices for selected frequencies.

    Returns:
    numpy.ndarray
        Reconstructed image based on selected frequencies.
    """

    return np.abs(np.sum(res[:, :, indices] * b[indices], axis=2))

def plot_overlays(
    phantom_image,
    ventMap,
    perfMap,
    vent_range=None,
    perf_range=None,
    phantom_range=None,
    vent_alpha: float = 0.5,
    perf_alpha: float = 0.5,
    mask: np.ndarray | None = None,
    output_path: str | None = None,
    vent_filename: str = "overlay_ventilation.png",
    perf_filename: str = "overlay_perfusion.png",
    dpi: int = 300,
    show: bool = False,
    config: object = None
):
    """
    Create two figures:
      1) Phantom background with ventilation map overlay
      2) Phantom background with perfusion map overlay

    Args:
        phantom_image: 2D background image.
        ventMap, perfMap: 2D maps to overlay.
        vent_range, perf_range: (vmin, vmax) for overlays. Default -> (0, max(map)).
        phantom_range: (vmin, vmax) for background. Default -> data min/max.
        vent_alpha, perf_alpha: overlay transparency [0..1].
        mask: optional boolean mask; False/0 pixels won't be drawn in the overlay.
        output_path: if given, save PNGs here.
        vent_filename, perf_filename: output filenames.
        dpi: save resolution.
        show: if True, plt.show() the figures.

    Returns:
        (vent_path or None, perf_path or None)
    """
    # Colormaps (same style as your function)
    ocean_cmap = LinearSegmentedColormap.from_list('ocean', [
        '#000000','#000080', '#0000cd', '#1e90ff', '#00bfff', '#87ceeb',
        '#e0ffff','#ffffff'
    ])
    blackbody_cmap = LinearSegmentedColormap.from_list('blackbody', [
        '#000000','#550000', '#dd0000', '#ff8000', '#ffff80', '#ffffff'
    ])

    def _safe_range(data, given):
        if given is not None:
            return given
        vmax = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
        vmin = 0.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        return vmin, vmax

    def _phantom_range(img, given):
        if given is not None:
            return given
        vmin = float(np.nanmin(img)) if np.isfinite(np.nanmin(img)) else 0.0
        vmax = float(np.nanmax(img)) if np.isfinite(np.nanmax(img)) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        return vmin, vmax

    # Prepare masked overlays (NaNs won't render, revealing background)
    vent_overlay = np.where(mask, ventMap, np.nan) if mask is not None else ventMap
    if perfMap is not None:
        perf_overlay = np.where(mask, perfMap, np.nan) if mask is not None else perfMap if perfMap is not None else None
    else:
        perf_overlay = None

    pvmin, pvmax = _phantom_range(phantom_image, phantom_range)
    vvmin, vvmax = _safe_range(vent_overlay, vent_range)
    qvmin, qvmax = _safe_range(perf_overlay, perf_range) if not config.phantom else (None, None)

    saved_vent_path, saved_perf_path = None, None

    # --- Ventilation overlay ---
    fig_v, ax_v = plt.subplots(1, 1, figsize=(6, 6))
    ax_v.imshow(phantom_image, cmap='gray', vmin=pvmin, vmax=pvmax)
    im_v = ax_v.imshow(vent_overlay, cmap=ocean_cmap, vmin=vvmin, vmax=vvmax, alpha=vent_alpha)
    ax_v.set_title('Ventilation overlay')
    ax_v.axis('off')
    cb_v = plt.colorbar(im_v, ax=ax_v, orientation='vertical', fraction=0.046, pad=0.04)
    cb_v.set_label('Fractional Ventilation [ml/ml]')
    plt.tight_layout()

    # --- Perfusion overlay ---
    if not config.phantom:
        fig_p, ax_p = plt.subplots(1, 1, figsize=(6, 6))
        ax_p.imshow(phantom_image, cmap='gray', vmin=pvmin, vmax=pvmax)
        im_p = ax_p.imshow(perf_overlay, cmap=blackbody_cmap, vmin=qvmin, vmax=qvmax, alpha=perf_alpha)
        ax_p.set_title('Perfusion overlay')
        ax_p.axis('off')
        cb_p = plt.colorbar(im_p, ax=ax_p, orientation='vertical', fraction=0.046, pad=0.04)
        cb_p.set_label('Perfusion [normalized]')
        plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        saved_vent_path = os.path.join(output_path, f"{config.spectral_method}_{vent_filename}")
        fig_v.savefig(saved_vent_path, dpi=dpi, bbox_inches='tight')
        saved_perf_path = os.path.join(output_path,f"{config.spectral_method}_{perf_filename}") if perf_overlay is not None else None
        if not config.phantom:
            fig_p.savefig(saved_perf_path, dpi=dpi, bbox_inches='tight')

def plot_individual_modes(Phi, freq, lambda_, b, r,
                          sx=256, sy=256,
                          mode_range=None, freq_range=None,
                          mask=None,
                          output_path="mode_plots",
                          freq_tol=5e-4):
    """
    Plot the DC component followed by each DMD mode as individual images,
    only using one side of each positive/negative frequency pair,
    sorted by increasing frequency.

    Parameters:
    -----------
    Phi        : np.ndarray, shape = (n_pixels, r)
        DMD modes (flattened over the full image or masked vector).
    freq       : np.ndarray, shape = (r,)
        Mode frequencies in Hz (can be negative).
    lambda_    : np.ndarray, shape = (r,)
        Discrete-time eigenvalues.
    b          : np.ndarray, shape = (r,)
        Mode amplitudes (unused in plot normalization).
    r          : int
        Number of modes (rank) produced by DMD.
    sx, sy     : int
        Width and height of the original image.
    mode_range : tuple (i_min, i_max) or None
        Plot modes with indices i_min through i_max inclusive.
    freq_range : tuple (f_min, f_max) or None
        Plot modes whose absolute frequencies lie within [f_min, f_max].
    mask       : np.ndarray(bool), shape = (sy, sx) or None
        If provided, Phi is assumed only for True pixels; will embed into full image.
    output_path : str
        Directory to save individual mode plots.
    freq_tol   : float
        Frequency tolerance for zero (DC) component.

    Returns:
    --------
    None; saves PNG files for DC component and each selected mode under output_dir.
    """
    os.makedirs(output_path, exist_ok=True)

    # Build the 3D stack of modes (sy × sx × r)
    # Build the 3D stack of modes (sy × sx × r)
    if mask is None:
        res_DMD = Phi[:sx*sy, :].reshape((sy, sx, r))
    else:
        flat_mask = mask.ravel()
        # Initialisiere das vollständige Bild-Array mit Nullen für alle Pixel (H*W, r)
        res_flat = np.zeros((len(flat_mask), r), dtype=Phi.dtype)
        # Fülle nur die True-Pixel der Maske mit den berechneten DMD-Modi
        res_flat[flat_mask, :] = Phi
        # Reshape zurück auf die echten Bildabmessungen (sy, sx, r)
        res_DMD = res_flat.reshape((sy, sx, r))

    # Compute DC image and plot it first
    dc_idx = np.where(np.abs(freq) < freq_tol)[0]
    dc_DMD = reconstruct_freq_image(b/2, res_DMD, dc_idx)
    plt.figure(figsize=(6, 6))
    plt.imshow(dc_DMD, cmap='gray')
    plt.title("DC Component")
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')
    plt.savefig(os.path.join(output_path, "001_dc_component.png"), bbox_inches='tight')
    plt.close()

    # Determine which mode indices to plot (exclude DC)
    if mode_range is not None:
        i_min, i_max = mode_range
        indices = np.arange(i_min, min(i_max + 1, r))
    elif freq_range is not None:
        f_min, f_max = freq_range
        indices = np.where((np.abs(freq) >= f_min) & (np.abs(freq) <= f_max))[0]
    else:
        indices = np.arange(r)
    # Keep only positive frequencies
    pos_indices = [idx for idx in indices if freq[idx] > freq_tol]

    # Sort positive-frequency indices by increasing frequency
    sorted_indices = sorted(pos_indices, key=lambda i: freq[i])

    # Plot each mode in sorted order
    for count, idx in enumerate(sorted_indices, start=2):
        img = np.abs(res_DMD[:, :, idx] * b[idx])
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"freq = {freq[idx]:.3f} Hz | lambda = {np.abs(lambda_[idx]):.3f} | amplitude = {np.abs(b[idx]):.3f}")
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label('Intensity')
        filename = os.path.join(output_path, f"DMD_individual_mode_{count:03d}_component.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    print(f"Saved {1 + len(sorted_indices)} images (DC + modes) sorted by frequency to '{output_path}'")


def plot_segmentation(mean2d_np, augmented_np, segmentation_method, output_path, series_indicator):
    """ 
    Plot the mean 2D image with the segmentation overlay and save the figure.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mean2d_np, cmap="gray")
    plt.contour(augmented_np, levels=[0.5], colors="r")
    plt.axis("off")
    plt.title(f"Segmentation: {segmentation_method}")
    plt.savefig(f"{output_path}/segmentation_{segmentation_method}_{series_indicator}.png", bbox_inches='tight')


def plot_frequency_spectrum_FD(spectrum_freq, spectrum_amp, vent_hz, perf_hz, output_path=None):
    # mark peaks
    # find the nearest frequency bins for markers
    def nearest_bin(f):
        return int(np.argmin(np.abs(spectrum_freq - f)))
    

    vbin = nearest_bin(vent_hz)

    plt.figure(figsize=(6, 4))
    plt.plot(spectrum_freq, spectrum_amp, linewidth=2)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Mean-signal spectrum')
    plt.scatter([spectrum_freq[vbin]], [spectrum_amp[vbin]])
    plt.axvline(spectrum_freq[vbin], linestyle='--', alpha=0.7)
    plt.annotate(f'Vent {vent_hz:.2f} Hz', (spectrum_freq[vbin], spectrum_amp[vbin]),
                textcoords='offset points', xytext=(8, 8))

    # Plot perfusion marker only if available
    if perf_hz is not None:
        pbin = nearest_bin(perf_hz)
        plt.scatter([spectrum_freq[pbin]], [spectrum_amp[pbin]])
        plt.axvline(spectrum_freq[pbin], linestyle='--', alpha=0.7)
        plt.annotate(f'Perf {perf_hz:.2f} Hz', (spectrum_freq[pbin], spectrum_amp[pbin]),
                    textcoords='offset points', xytext=(8, 8))

    plt.xlim(0, spectrum_freq.max())
    # Protect against zero-height y-limits which can produce a blank/empty-looking figure
    if spectrum_amp.size and spectrum_amp.max() > 0:
        ymax = spectrum_amp.max()
    else:
        ymax = 1.0
    plt.ylim(0, ymax * 1.1)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, 'frequency_spectrum.jpg')
    plt.savefig(out_file, bbox_inches='tight', dpi=200)

def plot_modes_DMD(freq, b, vent_hz, perf_hz, output_path=None):

    maskf = freq >= 0.01
    freq_filt = freq[maskf]
    b_filt = b[maskf]
    plt.figure()
    plt.scatter(freq_filt, np.abs(b_filt), linewidth=2)
    plt.xlim(0, freq_filt.max() * 1.1)
    plt.ylim(0, np.abs(b_filt).max() * 1.1)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Mean-signal spectrum DMD')
    plt.savefig(os.path.join(output_path, 'dmd_modes.jpg'))