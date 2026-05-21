import numpy as np
from scipy.linalg import svd, eig
import scipy.linalg as linalg
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def dynamic_mode_decomp(X, dt=1, r=1e32, nstacks=1, mask= None):
    """
    Computes the Dynamic Mode Decomposition of data X.

    Parameters:
    X : numpy.ndarray
        Data matrix where columns are state snapshots and rows are measurements.
    dt : float, optional
        Time step between snapshots (default is 1).
    r : int, optional
        Truncate to rank-r (default is large number, effectively no truncation).
    nstacks : int, optional
        Number of stacks of the raw data (default is 1).

    Returns:
    Phi : numpy.ndarray
        The DMD modes.
    omega : numpy.ndarray
        The continuous-time DMD eigenvalues.
    lambda_ : numpy.ndarray
        The discrete-time DMD eigenvalues.
    b : numpy.ndarray
        A vector of magnitudes of modes Phi.
    freq : numpy.ndarray
        The estimated frequencies in Hertz.
    Xdmd : numpy.ndarray
        The data matrix reconstructed by Phi, omega, b.
    r : int
        The rank used for truncation.
    
    Adaptation of the codes: https://github.com/hanyoseob/python-DMD/blob/master/demo_DMD.py
                       and   https://github.com/EfeIlicak/DMD_Lung
    """
    if mask is not None:
        flat_mask = mask.ravel()            # length = H*W
        X = X[flat_mask, :]                 # now shape = (n_lung, T)
        
    hermitian = lambda x: np.conj(np.transpose(x))
    
    if nstacks > 1:
        Xaug = np.hstack([X[:, st:X.shape[1] - nstacks + st] for st in range(nstacks)])
        X1 = Xaug[:, :-1]
        X2 = Xaug[:, 1:]
    else:
        X1 = X[:, :-1]
        X2 = X[:, 1:]

    ## STEP 1: singular value decomposition (SVD)
    m1, n1 = X1.shape
    U, Sdiag, Vh = svd(X1, full_matrices=False)
    # compact diagonal matrix of singular values (length k)
    k = Sdiag.size
    S = np.diag(Sdiag)
    V = hermitian(Vh)

    # DMD

    print(f"Shape of U: {U.shape}")
    print(f"Shape of S: {S.shape}")
    print(f"Shape of V: {V.shape}")

    # Ensure r is an integer and does not exceed available rank or snapshot count
    r = int(min(int(r), k, n1))

    # Truncate to r modes and build reduced operator
    Ur = U[:, :r]
    Sr = S[:r, :r]
    Vr = V[:, :r]
    Atilde = np.dot(hermitian(Ur), np.dot(X2, np.dot(Vr, np.linalg.inv(Sr))))
    print(f"Shape of Atilde: {Atilde.shape}")
    Ddiag, W = eig(Atilde)
    mA, nA = Atilde.shape
    D = np.zeros((mA, nA), dtype=Ddiag.dtype)
    D[:nA, :nA] = np.diag(Ddiag)
    print(f"Shape of W: {W.shape}")
    print(f"Shape of D: {D.shape}")
    print(f'Shape of X2: {X2.shape}')
    Phi = np.dot(X2, np.dot(Vr, np.dot(np.linalg.inv(Sr), W)))
    print(f"Shape of Phi: {Phi.shape}")

    lambda_ = np.diag(D)
    omega = np.log(lambda_) / dt

    # Compute the frequencies
    freq = np.angle(lambda_) / (2 * np.pi * dt)

    # Compute DMD mode amplitudes
    x1 = X[:, 0]    # time = 0
    b = np.dot(linalg.pinv(Phi), x1)
    
    t = np.arange(n1) * dt
    time_dynamics = np.zeros((r, len(t)), dtype=complex)

    for i in range(len(t)):
        time_dynamics[:, i] = b * np.exp(omega*t[i])

    Xdmd = np.dot(Phi, time_dynamics)

    return Phi, omega, lambda_, b, freq, Xdmd, r


def mean_step_size(array):
    """
    Calculate the mean step size of a given array.

    Parameters:
    array (list or numpy.ndarray): The input array.

    Returns:
    float: The mean step size of the array.
    """
    if len(array) < 2:
        raise ValueError("Array must contain at least two elements to calculate step size.")
    
    # Calculate the differences between consecutive elements
    differences = np.diff(array)
    
    # Compute the mean of these differences
    mean_step = np.mean(differences)
    
    return mean_step

def process_DMD_modes(Phi, freq, lambda_, b, r,
                      sx=256, sy=256,
                      ventRange=[0.05, 0.35],
                      perfRange=[0.75, 1.25],
                      mask=None):
    """
    Process DMD modes to extract ventilation and perfusion maps.
    
    Parameters:
    -----------
    Phi      : np.ndarray, shape = (n_pixels, r)
        DMD modes (flattened over the full image).
    freq     : np.ndarray, shape = (r,)
        Mode frequencies in Hz.
    lambda_  : np.ndarray, shape = (r,)
        Discrete eigenvalues.
    b        : np.ndarray, shape = (r,)
        Mode amplitudes.
    r        : int
        Number of modes.
    sx, sy   : int
        Width and height of the original image.
    ventRange: list of two floats
        [min, max] frequency bounds for ventilation.
    perfRange: list of two floats
        [min, max] frequency bounds for perfusion.
    mask     : None or np.ndarray(bool) shape = (sy, sx)
        If None, uses full image. Otherwise, mask==True are lung pixels.
    
    Returns:
    --------
    dc_DMD   : np.ndarray, shape = (sy, sx)
    ventMap  : np.ndarray, shape = (sy, sx)
    perfMap  : np.ndarray, shape = (sy, sx)
    """
    # 1) Build the 3D stack of modes (sy × sx × r)
    if mask is None:
        # original full‐image behavior
        res_DMD = Phi[:(sx*sy), :].reshape((sy, sx, r))
    else:
        # masked: fill zeros outside ROI
        flat_mask = mask.ravel()                # length = sx*sy
        # allocate full plane
        res_flat = np.zeros((sx*sy, r), dtype=Phi.dtype)
        # fill lung pixels
        res_flat[flat_mask, :] = Phi
        # reshape back to image
        res_DMD = res_flat.reshape((sy, sx, r))
    
    # 2) find mode indices
    vent_idx = np.where((freq > ventRange[0]) & (freq < ventRange[1]))[0]
    
    # Handle the case where perfRange is None (phantom mode):
    if perfRange is None:
        perf_idx = np.array([], dtype=int)
    else:
        perf_idx = np.where((freq > perfRange[0]) & 
                            (freq < perfRange[1])
                            )[0]

    print('ventilation frequencies:', freq[vent_idx]) 

    if perf_idx.size == 0:
        print('perfusion frequencies: None')
    else:
        print('perfusion frequencies:', freq[perf_idx])
    dc_idx   = np.where(np.abs(freq) < 5e-4)[0]
    

    # 3) reconstruct images
    dc_DMD   = reconstruct_freq_image(b/2, res_DMD, dc_idx)
    vent_DMD = reconstruct_freq_image(b,   res_DMD, vent_idx)
    # Perfusion map: if no perfusion indices (phantom mode), set zeros
    if perf_idx.size == 0:
        perf_DMD = np.zeros_like(dc_DMD)
    else:
        perf_DMD = reconstruct_freq_image(b,   res_DMD, perf_idx)
    
    ## Commented out several changes to the maps
    # 4) compute ventilation map
    #BGr     = dc_DMD[:30, :30]
    #BG      = np.std(BGr)
    #ventMap = np.abs(vent_DMD / ((vent_DMD/2) + dc_DMD - BG))
    #ventMap = np.abs(vent_DMD / ((vent_DMD/2) + dc_DMD))
    ventMap = vent_DMD
    
    # 5) compute perfusion map
    #perfp        = np.percentile(perf_DMD, 99)
    #perf_DMD[perf_DMD > perfp] = perfp
    #perfMap      = perf_DMD / perfp
    perfMap = perf_DMD
    
    return dc_DMD, ventMap, perfMap
    
def mask_images(mask_bool, dc_image, vent_image, perf_image, background_value=0):

    # Create masked images with NaN for the mask
    masked_dc = np.where(mask_bool, dc_image, background_value)
    masked_vent = np.where(mask_bool, vent_image, background_value)
    masked_perf = np.where(mask_bool, perf_image, background_value)

    return masked_dc, masked_vent, masked_perf

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

def create_rgb_overlay(
    anatomical_image,
    Map,
    map_type: str = 'ventilation',
    map_range=None,
    phantom_range=None,
    alpha: float = 0.5,
    mask: np.ndarray | None = None,
    output_path: str | None = None,
    show: bool = False,
):
    """
    Create an RGB overlay matrix by blending a chosen colormap-mapped Map
    onto a grayscale anatomical_image and return the RGB matrix.

    Parameters
    ----------
    anatomical_image : 2D array
        Grayscale background image.
    Map : 2D array
        Map to overlay (ventilation or perfusion values).
    map_type : {'ventilation','perfusion'}
        Which style of colormap to use for the Map.
    map_range : (vmin, vmax) or None
        Value range for Map normalization. If None, uses (0, max(Map)).
    phantom_range : (vmin, vmax) or None
        Value range for anatomical image normalization. If None, uses data min/max.
    alpha : float
        Blend factor between background and overlay (0..1). 0 -> anatomy only.
    mask : 2D boolean array or None
        If provided, where mask is False the background is shown (overlay hidden).

    Returns
    -------
    rgb : np.ndarray, shape (H, W, 3)
        RGB image in float [0,1].
    """
    # local helper: compute ranges safely
    def _safe_range(data, given, default_vmin=0.0):
        if given is not None:
            return given
        # prefer 0..max for maps, but guard for non-finite
        vmax = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
        vmin = default_vmin
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

    # choose colormap
    style = (map_type or 'ventilation').lower()
    ocean_cmap = LinearSegmentedColormap.from_list('ocean', [
        '#000000','#000080', '#0000cd', '#1e90ff', '#00bfff', '#87ceeb',
        '#e0ffff','#ffffff'
    ])
    blackbody_cmap = LinearSegmentedColormap.from_list('blackbody', [
        '#000000','#550000', '#dd0000', '#ff8000', '#ffff80', '#ffffff'
    ])

    if style in ('perfusion', 'perf', 'red'):
        cmap = blackbody_cmap
    else:
        cmap = ocean_cmap

    # normalize anatomical image to [0,1]
    pvmin, pvmax = _phantom_range(anatomical_image, phantom_range)
    anat_norm = (anatomical_image - pvmin) / (pvmax - pvmin)
    anat_norm = np.clip(anat_norm, 0.0, 1.0)

    # normalize map to [0,1]
    vmin, vmax = _safe_range(Map, map_range, default_vmin=0.0)
    map_norm = (Map - vmin) / (vmax - vmin)
    map_norm = np.nan_to_num(map_norm, nan=0.0)
    map_norm = np.clip(map_norm, 0.0, 1.0)

    # map values to RGB
    cmap_rgba = cmap(map_norm)
    cmap_rgb = cmap_rgba[..., :3]

    # background RGB
    bg_rgb = np.stack([anat_norm, anat_norm, anat_norm], axis=-1)

    # Use colormap-mapped RGB directly (like plt.imshow does) ---
    # do NOT premultiply by map_norm again (that caused darker colors)
    cmap_rgba_full = cmap(map_norm)
    cmap_rgb_full = cmap_rgba_full[..., :3]

    # blend: alpha mixes between background and overlay (uniform alpha)
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    blended = (1.0 - alpha) * bg_rgb + alpha * cmap_rgb_full

    # if mask provided, apply it so masked-out pixels show background only
    if mask is not None:
        mask_bool = mask.astype(bool)
        if mask_bool.shape != anat_norm.shape:
            try:
                mask_bool = mask_bool.reshape(anat_norm.shape)
            except Exception:
                mask_bool = np.broadcast_to(mask_bool, anat_norm.shape)
        rgb = np.where(mask_bool[..., np.newaxis], blended, bg_rgb)
    else:
        rgb = blended

    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb