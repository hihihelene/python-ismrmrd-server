import SimpleITK as sitk
import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
# import napari
# from skimage.filters import threshold_otsu
import os
# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def _spacing_scaled_radius(image, radius):
    """
    Convert a scalar radius to a voxel-aware radius based on spacing.
    Keeps backward compatibility if a tuple is already supplied.
    """
    if isinstance(radius, (list, tuple)):
        return list(radius)

    spacing = image.GetSpacing()
    dim = image.GetDimension()

    scaled = []
    for i in range(dim):
        scaled.append(int(max(1, round(radius / spacing[i]))))

    return scaled


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def correct_signal_nonuniformity(image):
    corrected_image = sitk.N4BiasFieldCorrection(image)
    return corrected_image


# ---------------------------------------------------------
# Thresholding
# ---------------------------------------------------------

def locally_adaptive_thresholding(image, lowerThreshold=1):

    arr = sitk.GetArrayFromImage(image)

    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(float(arr.mean()) * lowerThreshold)
    threshold_filter.SetUpperThreshold(float(arr.max()))
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)

    threshold_image = threshold_filter.Execute(image)

    return threshold_image


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------

def largest_connected_component(segmented_image, min_component_size=0):

    binary = sitk.Cast(segmented_image > 0, sitk.sitkUInt8)

    cc = sitk.ConnectedComponent(binary)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    largest_label = None
    max_size = 0

    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size > max_size:
            max_size = size
            largest_label = label

    if largest_label is None or max_size < min_component_size:
        return sitk.Image(binary.GetSize(), binary.GetPixelID())

    return sitk.Cast(cc == largest_label, binary.GetPixelID())


# ---------------------------------------------------------
# Hole filling
# ---------------------------------------------------------

def fill_gaps(mask, radius=5, out_pixel_id=sitk.sitkUInt8):

    bin_mask = sitk.Cast(mask > 0, out_pixel_id)

    radius_vector = _spacing_scaled_radius(mask, radius)

    closed = sitk.BinaryMorphologicalClosing(bin_mask, radius_vector, sitk.sitkBall)

    filled_holes = sitk.BinaryFillhole(bin_mask)

    interior_holes = sitk.And(filled_holes, sitk.Not(bin_mask))

    closing_hole_candidates = sitk.And(closed, sitk.Not(bin_mask))

    holes_to_apply = sitk.And(closing_hole_candidates, interior_holes)

    result = sitk.Or(bin_mask, holes_to_apply)

    return sitk.Cast(result > 0, out_pixel_id)


# ---------------------------------------------------------
# Body extraction
# ---------------------------------------------------------

def extract_body_mask(image,
                      lowerThreshold=1,
                      radius=5,
                      out_pixel_id=sitk.sitkUInt8):

    corrected = sitk.N4BiasFieldCorrection(image)

    seg = locally_adaptive_thresholding(corrected, lowerThreshold)

    init_cc = largest_connected_component(seg)

    body_bin = sitk.Cast(init_cc > 0, out_pixel_id)

    body_filled = fill_gaps(body_bin, radius, out_pixel_id)

    return body_filled


# ---------------------------------------------------------
# Rough lung segmentation
# ---------------------------------------------------------

def rough_lung_segmentation(image, body_mask,
                            lung_lower_factor=0.2,
                            lung_upper_factor=0.5,
                            opening_radius=5,
                            closing_radius=5,
                            out_pixel_id=sitk.sitkUInt8):

    def _largest_component_or_empty(binary_img):
        cc_local = sitk.ConnectedComponent(sitk.Cast(binary_img > 0, sitk.sitkUInt8))
        stats_local = sitk.LabelShapeStatisticsImageFilter()
        stats_local.Execute(cc_local)

        labels_local = list(stats_local.GetLabels())
        if not labels_local:
            return sitk.Image(binary_img.GetSize(), sitk.sitkUInt8)

        keep_local = max(labels_local, key=lambda lbl: stats_local.GetPhysicalSize(lbl))
        return sitk.Cast(sitk.Equal(cc_local, keep_local), sitk.sitkUInt8)

    img_f = sitk.Cast(image, sitk.sitkFloat32)

    mask = sitk.Cast(body_mask > 0, sitk.sitkUInt8)

    segmented_image_f = sitk.Mask(img_f, mask, outsideValue=-1024)

    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(segmented_image_f, mask)

    mean_val = label_stats.GetMean(1)

    lower = lung_lower_factor * mean_val
    upper = lung_upper_factor * mean_val

    lungs_thresh = sitk.BinaryThreshold(
        segmented_image_f,
        lowerThreshold=lower,
        upperThreshold=upper,
        insideValue=1,
        outsideValue=0
    )

    open_rad = _spacing_scaled_radius(image, opening_radius)
    close_rad = _spacing_scaled_radius(image, closing_radius)

    lungs_opened = sitk.BinaryMorphologicalOpening(
        lungs_thresh, open_rad, sitk.sitkBall
    )

    lungs_closed = sitk.BinaryMorphologicalClosing(
        lungs_opened, close_rad, sitk.sitkBall
    )

    full_size = list(lungs_closed.GetSize())
    dim = lungs_closed.GetDimension()
    split_axis = 0
    mid = full_size[split_axis] // 2

    # Split at image midline and keep the dominant lung component in each half.
    if mid > 0 and (full_size[split_axis] - mid) > 0:
        left_size = full_size.copy()
        right_size = full_size.copy()
        left_size[split_axis] = mid
        right_size[split_axis] = full_size[split_axis] - mid

        left_index = [0] * dim
        right_index = [0] * dim
        right_index[split_axis] = mid

        left_half = sitk.RegionOfInterest(lungs_closed, left_size, left_index)
        right_half = sitk.RegionOfInterest(lungs_closed, right_size, right_index)

        left_filtered = _largest_component_or_empty(left_half)
        right_filtered = _largest_component_or_empty(right_half)

        filtered = sitk.Image(lungs_closed.GetSize(), sitk.sitkUInt8)
        filtered.CopyInformation(lungs_closed)

        filtered = sitk.Paste(
            filtered,
            left_filtered,
            left_filtered.GetSize(),
            [0] * dim,
            left_index
        )
        filtered = sitk.Paste(
            filtered,
            right_filtered,
            right_filtered.GetSize(),
            [0] * dim,
            right_index
        )
    else:
        cc = sitk.ConnectedComponent(lungs_closed)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        labels = list(stats.GetLabels())

        labels_sorted = sorted(
            labels,
            key=lambda l: stats.GetPhysicalSize(l),
            reverse=True
        )

        keep = labels_sorted[:2]
        filtered = None

        for lab in keep:
            this_comp = sitk.Equal(cc, lab)
            filtered = this_comp if filtered is None else sitk.Or(filtered, this_comp)

        if filtered is None:
            filtered = sitk.Image(lungs_closed.GetSize(), sitk.sitkUInt8)
            filtered.CopyInformation(lungs_closed)

    segmented_image = sitk.Cast(segmented_image_f, image.GetPixelID())

    rough_lung_mask = sitk.Cast(filtered, out_pixel_id)

    return segmented_image, rough_lung_mask


# ---------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------

def average_signal_inside_mask(image, mask):

    masked_image = sitk.Mask(image, mask)

    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(masked_image, mask)

    mean_value = stats_filter.GetMean(1)

    return mean_value



def calculate_threshold(image, lung_mask, body_mask):

    surrounding_body = sitk.Subtract(body_mask, lung_mask)

    mean_lung = average_signal_inside_mask(image, lung_mask)

    mean_body = average_signal_inside_mask(image, surrounding_body)

    T = (mean_lung + mean_body) / 2

    return T


# ---------------------------------------------------------
# automatic lung segmentation 
# ---------------------------------------------------------

def segment_automatic(mean_image, **kwargs):

    # defining parameters
    neighborhood_radius=1
    num_iterations='max'
    erosion_iters=0
    
    mean_image_sitk = sitk.GetImageFromArray(mean_image)
    body_mask = extract_body_mask(mean_image_sitk, lowerThreshold=0.25, radius=10)
    _, lung_init = rough_lung_segmentation(mean_image_sitk, body_mask, lung_lower_factor=0.0, lung_upper_factor=0.43)

    threshold = calculate_threshold(mean_image_sitk, lung_init, body_mask)

    img_vals = sitk.GetArrayFromImage(mean_image_sitk)

    lung_arr = sitk.GetArrayFromImage(lung_init).astype(bool)

    body_arr = sitk.GetArrayFromImage(body_mask).astype(bool)

    candidate = (img_vals < threshold) & body_arr
    print('Checking neighborhood radius:', neighborhood_radius)
    structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 2), neighborhood_radius)

    augmented = lung_arr.copy()

    iteration = 1

    if num_iterations == 'max':

        changed = True

        while changed:

            dilated = ndimage.binary_dilation(augmented, structure=structure)

            new_mask = dilated & candidate

            changed = not np.array_equal(new_mask, augmented)

            augmented = new_mask

            print(f"Iteration {iteration}: {'change' if changed else 'no change'}")

            iteration += 1

    else:

        for _ in range(num_iterations):

            dilated = ndimage.binary_dilation(augmented, structure=structure)

            new_mask = dilated & candidate

            changed = not np.array_equal(new_mask, augmented)

            augmented = new_mask

            print(f"Iteration {iteration}: {'change' if changed else 'no change'}")

            iteration += 1

    if erosion_iters and erosion_iters > 0:
        augmented = ndimage.binary_erosion(
            augmented,
            structure=ndimage.generate_binary_structure(2, 2),
            iterations=erosion_iters,
        )

    out = sitk.GetImageFromArray(augmented.astype(np.uint8))

    # out.CopyInformation(lung_mask)

    return out


# ---------------------------------------------------------
# Thorax connection
# ---------------------------------------------------------

def connect_lungs_sitk(augmented_lung: sitk.Image,
                       closing_radius: Tuple[int, int, int] = (20, 20, 5)
                       ) -> sitk.Image:

    bin_img = sitk.BinaryThreshold(
        augmented_lung,
        lowerThreshold=1,
        upperThreshold=65535,
        insideValue=1,
        outsideValue=0
    )

    closed = sitk.BinaryMorphologicalClosing(
        bin_img,
        closing_radius,
        sitk.sitkBall
    )

    return closed



# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------

def upscale_for_ui(img, target_max=1200):

    h, w = img.shape[:2]

    scale = target_max / max(h, w)

    if scale <= 1.0:
        return img, 1.0

    new_size = (int(w * scale), int(h * scale))

    img_up = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return img_up, scale


# ---------------------------------------------------------
# Napari segmentation
# ---------------------------------------------------------


def segment_napari(mean_image,  **kwargs):
    """
    Interactive napari-assisted threshold segmentation.

    Workflow:
    - User can optionally paint a rough ROI hint in Napari (label == roi_label).
    - Otsu threshold is computed on normalized image.
    - Morphological cleanup is applied.
    - If ROI hint exists, result is constrained to that ROI neighborhood.
    - Otherwise, largest interior connected component is selected.
    - Optionally, arteries can be painted as a separate label and merged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lung_mask, artery_mask) — both boolean arrays.
        artery_mask is all-False when segment_arteries=False.
    """

    # defining parameters

    roi_label=2,
    brush_size=8,
    border_width=10,
    roi_dilation=10,
    erosion_iters=5,
    segment_arteries=False,
    artery_label=3,
    artery_dilation=0,
    window_center=100,
    window_width=100

    image_f = np.asarray(mean_image, dtype=np.float32)
    finite_mask = np.isfinite(image_f)

    if np.any(finite_mask):
        finite_vals = image_f[finite_mask]
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
    else:
        min_val = 0.0
        max_val = 1.0

    if window_center is None or window_width is None:
        p2 = float(np.percentile(image_f[finite_mask], 2.0)) if np.any(finite_mask) else min_val
        p98 = float(np.percentile(image_f[finite_mask], 98.0)) if np.any(finite_mask) else max_val
        auto_width = max(p98 - p2, 1e-8)
        window_center = (p2 + p98) / 2.0 if window_center is None else float(window_center)
        window_width = auto_width if window_width is None else float(window_width)

    if not np.isfinite(window_width) or window_width <= 0:
        window_width = max(max_val - min_val, 1e-8)
    if not np.isfinite(window_center):
        window_center = (min_val + max_val) / 2.0

    window_min = window_center - (window_width / 2.0)
    windowed_display = (image_f - window_min) / window_width
    windowed_display = np.nan_to_num(windowed_display, nan=0.0, neginf=0.0, posinf=1.0)
    windowed_display = np.clip(windowed_display, 0.0, 1.0)

    viewer = napari.Viewer()
    viewer.add_image(windowed_display, name='windowed_image')

    labels_layer = viewer.add_labels(np.zeros_like(mean_image, dtype=np.int32))
    labels_layer.selected_label = roi_label
    labels_layer.brush_size = brush_size

    print("Napari instructions (optional):")
    print(f"- Paint a rough region hint using label {roi_label} to guide segmentation")
    print(f"- Display window: center={window_center:.3f}, width={window_width:.3f}")
    
    if segment_arteries:
        print(f"- Paint arteries using label {artery_label} (optional)")
    print("- Leave blank for automatic threshold-based segmentation")
    print("- Close the Napari window when done")

    napari.run()

    scribbles = labels_layer.data.astype(np.int32)

    img_norm = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min() + 1e-8)
    thresh_val = threshold_otsu(img_norm)

    seg_mask_binary = img_norm < thresh_val
    seg_mask_clean = ndimage.binary_opening(
        seg_mask_binary,
        structure=ndimage.generate_binary_structure(2, 2),
        iterations=1,
    )
    seg_mask_clean = ndimage.binary_closing(
        seg_mask_clean,
        structure=ndimage.generate_binary_structure(2, 2),
        iterations=1,
    )

    interior_mask = np.ones_like(mean_image, dtype=bool)
    interior_mask[:border_width, :] = False
    interior_mask[-border_width:, :] = False
    interior_mask[:, :border_width] = False
    interior_mask[:, -border_width:] = False
        
    if np.any(scribbles == roi_label):
        seg_mask = scribbles == roi_label
        # print(f"Using painted ROI as guidance. Threshold={thresh_val:.3f}")
        # roi_mask = ndimage.binary_dilation(scribbles == roi_label, iterations=roi_dilation)
        # seg_mask = seg_mask_clean & roi_mask
        # if erosion_iters and erosion_iters > 0:
        #     seg_mask = ndimage.binary_erosion(
        #         seg_mask,
        #         structure=ndimage.generate_binary_structure(2, 2),
        #         iterations=erosion_iters,
        #     )
    else:
        print(f"No ROI hint. Using automatic threshold={thresh_val:.3f}")
        interior_candidate = seg_mask_clean & interior_mask
        labeled, num_features = ndimage.label(interior_candidate)
        if num_features > 0:
            sizes = ndimage.sum(interior_candidate, labeled, range(1, num_features + 1))
            largest_label = int(np.argmax(sizes)) + 1
            seg_mask = labeled == largest_label
        else:
            seg_mask = seg_mask_clean

    artery_mask = np.zeros_like(seg_mask, dtype=bool)
    if segment_arteries:
        artery_mask = scribbles == artery_label
        if artery_dilation and artery_dilation > 0:
            artery_mask = ndimage.binary_dilation(
                artery_mask,
                structure=ndimage.generate_binary_structure(2, 2),
                iterations=artery_dilation,
            )

        return seg_mask.astype(bool), artery_mask.astype(bool)
    else:
        return seg_mask.astype(bool)


# ---------------------------------------------------------
# Loading pre-segmented masks
# ---------------------------------------------------------

def segment_load(mean_image, series_indicator=None):
        
        if not series_indicator:
            raise ValueError("series_indicator is required for segmentation_method='presegmented'.")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        preseg_path = os.path.join(base_dir, 'Results', series_indicator, f'{series_indicator}_results.npz')
        if not os.path.exists(preseg_path):
            raise FileNotFoundError(f"Presegmented mask file not found: {preseg_path}")

        with np.load(preseg_path, allow_pickle=False) as saved:
            if 'mask2d' in saved:
                loaded_mask = saved['mask2d']
            elif 'napari_mask' in saved:
                loaded_mask = saved['napari_mask']
            else:
                raise KeyError(
                    f"No mask found in {preseg_path}. Expected key 'mask2d' (or 'napari_mask')."
                )

        if loaded_mask.shape != mean_image.shape:
            raise ValueError(
                f"Loaded mask shape {loaded_mask.shape} does not match current image shape {mean_image .shape}."
            )

        return sitk.GetImageFromArray(loaded_mask.astype(np.uint8))