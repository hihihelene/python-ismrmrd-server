import os
import cv2
import numpy as np
from scipy import ndimage

try:
    from skimage.filters import threshold_otsu
    import napari
except ImportError:
    threshold_otsu = None
    napari = None


def _as_array(image):
    if isinstance(image, np.ndarray):
        return image
    import itk
    return itk.array_from_image(image)


def _spacing_scaled_radius(image, radius):
    if isinstance(radius, (list, tuple)):
        return list(radius)
    return [max(1, int(radius))] * _as_array(image).ndim


def correct_signal_nonuniformity(image):
    return _as_array(image).astype(np.float32)


def locally_adaptive_thresholding(image, lowerThreshold=1):
    values = _as_array(image)
    return ((values >= values.mean() * lowerThreshold) & (values <= values.max())).astype(np.uint8)


def largest_connected_component(segmented_image, min_component_size=0):
    labels, count = ndimage.label(_as_array(segmented_image) > 0)
    if count == 0:
        return np.zeros_like(labels, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    largest = int(np.argmax(sizes[1:]) + 1)
    if sizes[largest] < min_component_size:
        return np.zeros_like(labels, dtype=np.uint8)
    return (labels == largest).astype(np.uint8)


def fill_gaps(mask, radius=5, out_pixel_id=np.uint8):
    binary = _as_array(mask) > 0
    structure = ndimage.generate_binary_structure(binary.ndim, binary.ndim)
    closed = ndimage.binary_closing(binary, structure=structure, iterations=max(_spacing_scaled_radius(mask, radius)))
    filled = ndimage.binary_fill_holes(binary)
    return (binary | (closed & ~binary & filled)).astype(np.uint8)


def extract_body_mask(image, lowerThreshold=1, radius=5, out_pixel_id=np.uint8):
    corrected = correct_signal_nonuniformity(image)
    initial = largest_connected_component(locally_adaptive_thresholding(corrected, lowerThreshold))
    return fill_gaps(initial, radius, out_pixel_id)


def rough_lung_segmentation(image, body_mask, lung_lower_factor=0.2,
                            lung_upper_factor=0.5, opening_radius=5,
                            closing_radius=5, out_pixel_id=np.uint8):
    image_array = _as_array(image).astype(np.float32)
    body = _as_array(body_mask) > 0
    masked = np.where(body, image_array, -1024.0)
    mean_value = float(masked[body].mean()) if np.any(body) else 0.0
    thresholded = (masked >= lung_lower_factor * mean_value) & (masked <= lung_upper_factor * mean_value)
    structure = ndimage.generate_binary_structure(thresholded.ndim, thresholded.ndim)
    opened = ndimage.binary_opening(thresholded, structure=structure, iterations=opening_radius)
    closed = ndimage.binary_closing(opened, structure=structure, iterations=closing_radius)

    if closed.ndim == 2:
        midpoint = closed.shape[1] // 2
        left = largest_connected_component(closed[:, :midpoint])
        right = largest_connected_component(closed[:, midpoint:])
        filtered = np.zeros_like(closed, dtype=np.uint8)
        filtered[:, :midpoint] = left
        filtered[:, midpoint:] = right
    else:
        labels, count = ndimage.label(closed)
        sizes = np.bincount(labels.ravel())
        keep = np.argsort(sizes[1:])[::-1][:2] + 1 if count else []
        filtered = np.isin(labels, keep).astype(np.uint8)

    return masked, filtered


def average_signal_inside_mask(image, mask):
    image_array = _as_array(image)
    mask_array = _as_array(mask) > 0
    return float(image_array[mask_array].mean()) if np.any(mask_array) else 0.0


def calculate_threshold(image, lung_mask, body_mask):
    body = _as_array(body_mask) > 0
    lung = _as_array(lung_mask) > 0
    surrounding = body & ~lung
    return (average_signal_inside_mask(image, lung) + average_signal_inside_mask(image, surrounding)) / 2


def segment_automatic(mean_image, **kwargs):
    body_mask = extract_body_mask(mean_image, lowerThreshold=0.25, radius=10)
    _, lung_initial = rough_lung_segmentation(
        mean_image, body_mask, lung_lower_factor=0.0, lung_upper_factor=0.43
    )
    threshold = calculate_threshold(mean_image, lung_initial, body_mask)
    candidate = (_as_array(mean_image) < threshold) & (_as_array(body_mask) > 0)
    structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 2), 1)
    augmented = _as_array(lung_initial).astype(bool)

    while True:
        expanded = ndimage.binary_dilation(augmented, structure=structure) & candidate
        if np.array_equal(expanded, augmented):
            break
        augmented = expanded

    return augmented.astype(np.uint8)


def connect_lungs_sitk(augmented_lung, closing_radius=(20, 20, 5)):
    binary = _as_array(augmented_lung) > 0
    structure = ndimage.generate_binary_structure(binary.ndim, binary.ndim)
    return ndimage.binary_closing(binary, structure=structure, iterations=max(closing_radius)).astype(np.uint8)


def upscale_for_ui(img, target_max=1200):
    height, width = img.shape[:2]
    scale = target_max / max(height, width)
    if scale <= 1.0:
        return img, 1.0
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR), scale


def segment_napari(mean_image, **kwargs):
    if napari is None or threshold_otsu is None:
        raise ImportError("napari and scikit-image are required for interactive segmentation")

    image = np.asarray(mean_image, dtype=np.float32)
    finite = np.isfinite(image)
    low = float(np.nanmin(image)) if np.any(finite) else 0.0
    high = float(np.nanmax(image)) if np.any(finite) else 1.0
    display = np.nan_to_num((image - low) / max(high - low, 1e-8))
    viewer = napari.Viewer()
    viewer.add_image(display, name="windowed_image")
    labels_layer = viewer.add_labels(np.zeros_like(image, dtype=np.int32))
    napari.run()

    scribbles = labels_layer.data.astype(np.int32)
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
    mask = normalized < threshold_otsu(normalized)
    mask = ndimage.binary_closing(ndimage.binary_opening(mask))
    if np.any(scribbles == 2):
        return (scribbles == 2).astype(bool)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        mask = labels == int(np.argmax(sizes[1:]) + 1)
    return mask.astype(bool)


def segment_load(mean_image, series_indicator=None):
    if not series_indicator:
        raise ValueError("series_indicator is required for segmentation_method='presegmented'.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "Results", series_indicator, f"{series_indicator}_results.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Presegmented mask file not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        key = "mask2d" if "mask2d" in saved else "napari_mask"
        if key not in saved:
            raise KeyError(f"No mask found in {path}. Expected key 'mask2d' (or 'napari_mask').")
        mask = saved[key]
    if mask.shape != mean_image.shape:
        raise ValueError(f"Loaded mask shape {mask.shape} does not match current image shape {mean_image.shape}.")
    return mask.astype(np.uint8)
