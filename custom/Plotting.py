import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_image(ax, data, cmap, title, vmin=None, vmax=None):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def plot_results(image, dc_image, ventMap, perfMap, technique = 'DMD', 
                 ventMap_range=None, perfMap_range=None, filepath=None,
                 vent_freqs=None, perf_freqs=None, show_freq_text=True, max_list_items=6):  
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
    vmin_vent, vmax_vent = (0, np.max(ventMap)) if ventMap_range is None else ventMap_range
    vmin_perf, vmax_perf = (0, np.max(perfMap)) if perfMap_range is None else perfMap_range

    # plots
    vent_title = 'Fractional Ventilation [ml/ml]' + '\n Technique:' + technique 
    perf_title = 'Perfusion [normalized]' + '\n Technique:' + technique

    plot_image(axs[0, 0], image, 'gray', 'Phantom [a.u.]',
               np.min(image), np.max(image))
    plot_image(axs[0, 1], dc_image, 'gray', 'DC Component [a.u.]',
               np.min(dc_image), np.max(dc_image))
    plot_image(axs[1, 0], ventMap, ocean_cmap, vent_title,
               vmin_vent, vmax_vent)
    plot_image(axs[1, 1], perfMap, blackbody_cmap, perf_title,
               vmin_perf, vmax_perf)

    # annotate frequencies directly on the vent/perf maps ---
    if show_freq_text and technique == 'DMD':
        vent_text = f"Vent freqs (Hz): {_fmt_freqs(vent_freqs, max_list_items)}"
        perf_text = f"Perf freqs (Hz): {_fmt_freqs(perf_freqs, max_list_items)}"

        # top-left corner inside each axes
        axs[1, 0].text(
            0.02, 0.02, vent_text,
            transform=axs[1, 0].transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=3)
        )
        axs[1, 1].text(
            0.02, 0.02, perf_text,
            transform=axs[1, 1].transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=3)
        )
    # ----------------------------------------------------------------

    plt.tight_layout()
    plt.show()
    
    if filepath:
        if technique == 'FD':
            filepath  = os.path.join(filepath, 'FD.png')
        elif technique == 'DMD':
            filepath = os.path.join(filepath, 'DMD.png')
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight')

    


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
    output_dir: str | None = None,
    vent_filename: str = "overlay_ventilation.png",
    perf_filename: str = "overlay_perfusion.png",
    dpi: int = 300,
    show: bool = False,
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
        output_dir: if given, save PNGs here.
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
    perf_overlay = np.where(mask, perfMap, np.nan) if mask is not None else perfMap

    pvmin, pvmax = _phantom_range(phantom_image, phantom_range)
    vvmin, vvmax = _safe_range(vent_overlay, vent_range)
    qvmin, qvmax = _safe_range(perf_overlay, perf_range)

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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        saved_vent_path = os.path.join(output_dir, vent_filename)
        fig_v.savefig(saved_vent_path, dpi=dpi, bbox_inches='tight')

    # --- Perfusion overlay ---
    fig_p, ax_p = plt.subplots(1, 1, figsize=(6, 6))
    ax_p.imshow(phantom_image, cmap='gray', vmin=pvmin, vmax=pvmax)
    im_p = ax_p.imshow(perf_overlay, cmap=blackbody_cmap, vmin=qvmin, vmax=qvmax, alpha=perf_alpha)
    ax_p.set_title('Perfusion overlay')
    ax_p.axis('off')
    cb_p = plt.colorbar(im_p, ax=ax_p, orientation='vertical', fraction=0.046, pad=0.04)
    cb_p.set_label('Perfusion [normalized]')
    plt.tight_layout()

    if output_dir:
        saved_perf_path = os.path.join(output_dir, perf_filename)
        fig_p.savefig(saved_perf_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig_v)
        plt.close(fig_p)

def plot_individual_modes(Phi, freq, lambda_, b, r,
                          sx=256, sy=256,
                          mode_range=None, freq_range=None,
                          mask=None,
                          output_dir="mode_plots",
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
    output_dir : str
        Directory to save individual mode plots.
    freq_tol   : float
        Frequency tolerance for zero (DC) component.

    Returns:
    --------
    None; saves PNG files for DC component and each selected mode under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build the 3D stack of modes (sy × sx × r)
    if mask is None:
        res_DMD = Phi[:sx*sy, :].reshape((sy, sx, r))
    else:
        flat_mask = mask.ravel()
        res_flat = np.zeros((sx*sy, r), dtype=Phi.dtype)
        res_flat[flat_mask, :] = Phi
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
    plt.savefig(os.path.join(output_dir, "001_dc_component.png"), bbox_inches='tight')
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
        filename = os.path.join(output_dir, f"{count:03d}_component.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    print(f"Saved {1 + len(sorted_indices)} images (DC + modes) sorted by frequency to '{output_dir}'")
