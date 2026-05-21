import SimpleITK as sitk
import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt




def correct_signal_nonuniformity(image):
    # Perform signal nonuniformity correction using N4ITK
    corrected_image = sitk.N4BiasFieldCorrection(image)
    return corrected_image

def locally_adaptive_thresholding(image,lowerThreshold=1):

    # Perform global thresholding
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(float(sitk.GetArrayFromImage(image).mean())*lowerThreshold)
    threshold_filter.SetUpperThreshold(float(sitk.GetArrayFromImage(image).max()))
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_image = threshold_filter.Execute(image)

    return threshold_image

def maximum_connected_component_analysis(segmented_image, min_component_size=600):
    # Threshold the segmented image to create a binary mask
    binary_image = segmented_image > 0
    
    # Label the connected components in the binary mask
    connected_components = sitk.ConnectedComponent(binary_image)
    
    # Compute the size of each connected component
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected_components)
    
    # Get the number of labels (connected components)
    num_labels = statistics.GetNumberOfLabels()
    
    # Initialize the mask to include all components above the minimum size
    mask = sitk.Image(binary_image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(binary_image)
    
    # Iterate through each label (connected component)
    for label in range(1, num_labels+1):
        size = statistics.GetPhysicalSize(label)
        if size >= min_component_size:
            # Add the connected component to the mask
            mask |= connected_components == label
    
    return mask

def largest_connected_component(segmented_image, min_component_size=0):
    """
    Keep only the largest connected component in `segmented_image`.
    If its size is below `min_component_size`, returns an empty mask.
    """
    # 1) Binarize
    binary = sitk.Cast(segmented_image > 0, sitk.sitkUInt8)

    # 2) Label each connected component
    cc = sitk.ConnectedComponent(binary)

    # 3) Compute size of each label
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    # 4) Find the label with the maximum size
    largest_label = None
    max_size = 0
    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size > max_size:
            max_size = size
            largest_label = label

    # 5) If it's too small, return an empty mask
    if largest_label is None or max_size < min_component_size:
        return sitk.Image(binary.GetSize(), binary.GetPixelID())

    # 6) Return mask of only that component
    return sitk.Cast(cc == largest_label, binary.GetPixelID())


def fill_gaps(mask, radius=5, out_pixel_id=sitk.sitkUInt8):
    """
    Binary‐close any small holes *that are fully enclosed* in `mask`,
    and return an 8‐bit binary image. Holes that touch the image border
    are left untouched.
    """
    # 1) Binarize: anything >0 → 1
    bin_mask = sitk.Cast(mask > 0, out_pixel_id)

    # 2) Global morphological closing (dilate→erode)
    radius_vector = [radius] * mask.GetDimension()
    closed = sitk.BinaryMorphologicalClosing(bin_mask, radius_vector, sitk.sitkBall)

    # 3) Find *all* truly enclosed holes:
    #    BinaryFillhole only fills holes that do *not* touch the image border.
    filled_holes = sitk.BinaryFillhole(bin_mask)
    #    Isolate those hole‐pixels:
    interior_holes = sitk.And(filled_holes, sitk.Not(bin_mask))

    # 4) Of the pixels your closing would fill (closed minus original),
    #    keep only the interior ones:
    closing_hole_candidates = sitk.And(closed, sitk.Not(bin_mask))
    holes_to_apply = sitk.And(closing_hole_candidates, interior_holes)

    # 5) Merge back into original mask
    result = sitk.Or(bin_mask, holes_to_apply)

    # 6) Ensure binary 0/1
    return sitk.Cast(result > 0, out_pixel_id)

def extract_body_mask(image,
                      lowerThreshold=1,
                      radius=5,
                      out_pixel_id=sitk.sitkUInt8):
    """
    - Bias‐correct
    - Locally threshold
    - Keep the largest connected component
    - Fill tiny gaps
    - Return an 8‐bit binary mask
    """
    # 1) N4 correction
    corrected = sitk.N4BiasFieldCorrection(image)
    # 2) Local thresholding → still probably a UInt32 label image after CC
    seg = locally_adaptive_thresholding(corrected, lowerThreshold)
    init_cc = largest_connected_component(seg)
    
    # 3) Binarize + cast before filling gaps
    body_bin = sitk.Cast(init_cc > 0, out_pixel_id)
    
    # 4) Close small holes
    body_filled = fill_gaps(body_bin, radius, out_pixel_id)
    
    return body_filled


def rough_lung_segmentation(image, body_mask,
                            lung_lower_factor=0.2,
                            lung_upper_factor=0.5,
                            opening_radius=5, closing_radius=5,
                            out_pixel_id=sitk.sitkUInt8):
    # 1) Cast & mask outside body to -1024 HU
    img_f = sitk.Cast(image, sitk.sitkFloat32)
    mask  = sitk.Cast(body_mask > 0, sitk.sitkUInt8)
    segmented_image_f = sitk.Mask(img_f, mask, outsideValue=-1024)

    # 2) Correct filter: use LabelStatisticsImageFilter here
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(segmented_image_f, mask)
    mean_val = label_stats.GetMean(1)

    # 3) Compute thresholds as fractions of that mean
    lower = lung_lower_factor * mean_val
    upper = lung_upper_factor * mean_val

    # 4) Threshold to get lung candidates
    lungs_thresh = sitk.BinaryThreshold(
        segmented_image_f,
        lowerThreshold=lower,
        upperThreshold=upper,
        insideValue=1,
        outsideValue=0
    )

    # 5) Morphological cleanup
    dim = img_f.GetDimension()
    lungs_opened = sitk.BinaryMorphologicalOpening(
        lungs_thresh, [opening_radius]*dim, sitk.sitkBall
    )
    lungs_closed = sitk.BinaryMorphologicalClosing(
        lungs_opened, [closing_radius]*dim, sitk.sitkBall
    )

    # 6) Connected component labeling
    cc = sitk.ConnectedComponent(lungs_closed)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    # get all labels sorted by physical size descending
    labels = stats.GetLabels()
    labels_sorted = sorted(labels,
                           key=lambda l: stats.GetPhysicalSize(l),
                           reverse=True)

    # pick the top‐num_largest labels
    keep = labels_sorted[:2]

    # build a mask that is 1 where cc==any of keep
    filtered = None
    for lab in keep:
        this_comp = sitk.Equal(cc, lab)           # binary image of that label
        filtered = this_comp if filtered is None else sitk.Or(filtered, this_comp)

    # 7) Cast outputs
    segmented_image = sitk.Cast(segmented_image_f, image.GetPixelID())
    rough_lung_mask = sitk.Cast(filtered, out_pixel_id)

    return segmented_image, rough_lung_mask
  

def extract_lung_masks(mask):
    
    inverted_body_mask= sitk.BinaryNot(mask)
    
    # Extract connected components
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    connected_components = connected_components_filter.Execute(inverted_body_mask)
    
    # Get the number of connected components
    num_components = connected_components_filter.GetObjectCount()
    
    # Check if there are at least two components
    if num_components < 3:
        print('Error: Not enough connected components. Try to adjust the radius and scaling factors.')
        return connected_components  # Handle edge case where there are not enough components
    
    else:

        # Use LabelShapeStatisticsImageFilter to get the size of each connected component
        label_shape_statistics_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_statistics_filter.Execute(connected_components)

        # Get the size of each connected component
        connected_components_sizes = [(label, label_shape_statistics_filter.GetPhysicalSize(label)) for label in range(1, num_components + 1)]

        # Sort the connected components by size
        sorted_components = sorted(connected_components_sizes, key=lambda x: x[1], reverse=True)

        # Get the labels of the second and third largest components
        second_largest_label = sorted_components[1][0]
        third_largest_label = sorted_components[2][0]

        # Create a binary image with only the second and third largest connected components
        binary_image = connected_components == second_largest_label
        binary_image |= connected_components == third_largest_label

        return binary_image
    
def average_signal_inside_mask(image,mask):
    # Apply the mask to the image
    masked_image = sitk.Mask(image, mask)
    
    # Compute statistics for nonzero pixels within the masked region
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(masked_image, mask)
    
    # Retrieve mean value of nonzero pixels
    mean_value = stats_filter.GetMean(1) 
    
    return mean_value

def calculate_threshold(image, lung_mask, body_mask):
    
    # Get full body mask
    #body_mask = extract_body_mask(image,lowerThreshold=1, radius=0)
    
    # Get mask of body surrounding the lung
    surrounding_body = sitk.Subtract(body_mask, lung_mask)
    
    # Get the mean values od the images inside the masks
    mean_lung = average_signal_inside_mask(image,lung_mask)
    mean_body = average_signal_inside_mask(image,surrounding_body)
    
    # Calculate Threshold
    T = (mean_lung+mean_body)/2
    
    return T


def augment_mask(image,
                 lung_mask,
                 body_mask,
                 neighborhood_radius=1,
                 num_iterations='max'):
    """
    Grow the lung_mask outwards (within body_mask) wherever all
    neighborhood intensities are below a threshold computed from image.

    Parameters
    ----------
    image : sitk.Image
        The original image (e.g. CT slice).
    lung_mask : sitk.Image
        Binary lung mask to start from.
    body_mask : sitk.Image
        Binary body mask; augmentation cannot cross its boundary.
    neighborhood_radius : int
        Radius (in pixels) of the circular neighborhood.
    num_iterations : int or 'max'
        If integer, run exactly that many dilation steps.
        If 'max', iterate until no further change occurs.

    Returns
    -------
    sitk.Image
        The augmented lung mask (same spacing/origin as lung_mask).
    """
    # 1) Compute your threshold
    threshold = calculate_threshold(image, lung_mask, body_mask)
    print(f'Threshold used for augmentation: {threshold}')

    # 2) Get arrays
    img_vals   = sitk.GetArrayFromImage(image)
    mask_arr   = sitk.GetArrayFromImage(lung_mask).astype(bool)
    body_arr   = sitk.GetArrayFromImage(body_mask).astype(bool)
    aug_arr    = mask_arr.copy()

    def one_pass(prev_arr):
        new_arr = prev_arr.copy()
        # for every pixel that is already in the mask...
        for (i, j), val in np.ndenumerate(prev_arr):
            if not val:
                continue
            # build list of neighbor coords within radius
            neigh = []
            for di in range(-neighborhood_radius, neighborhood_radius + 1):
                for dj in range(-neighborhood_radius, neighborhood_radius + 1):
                    if di*di + dj*dj <= neighborhood_radius*neighborhood_radius:
                        ni, nj = i + di, j + dj
                        # clamp to image bounds
                        ni = min(max(ni, 0), prev_arr.shape[0]-1)
                        nj = min(max(nj, 0), prev_arr.shape[1]-1)
                        neigh.append((ni, nj))

            # only grow into body
            # check that *all* neighbors are below threshold AND *inside* body
            if all(img_vals[ni, nj] < threshold and body_arr[ni, nj]
                   for ni, nj in neigh):
                for ni, nj in neigh:
                    new_arr[ni, nj] = True
        return new_arr

    count = 1
    if num_iterations == 'max':
        changed = True
        while changed:
            prev = aug_arr.copy()
            aug_arr = one_pass(prev)
            changed = not np.array_equal(prev, aug_arr)
            print(f'Iteration {count}: {"change" if changed else "no change"}')
            count += 1
    else:
        for _ in range(num_iterations):
            prev = aug_arr.copy()
            aug_arr = one_pass(prev)
            changed = not np.array_equal(prev, aug_arr)
            print(f'Iteration {count}: {"change" if changed else "no change"}')
            count += 1

    # 3) Convert back to SITK and preserve geometry
    out = sitk.GetImageFromArray(aug_arr.astype(np.uint8))
    out.CopyInformation(lung_mask)
    return out

def connect_lungs_sitk(augmented_lung: sitk.Image,
                       closing_radius: Tuple[int, int, int] = (20, 20, 5)
                       ) -> sitk.Image:
    """
    Given a binary lung mask image (two separate regions), perform a 3D
    binary closing to bridge the gap between them, returning a new mask image.

    Parameters
    ----------
    augmented_lung : sitk.Image
        A 3D binary mask (any integer pixel type) where lung voxels are non-zero.
    closing_radius : tuple of int
        The radius (in voxels) of the ball structuring element along each axis
        for the closing operation. Tune this to your expected inter-lung gap.

    Returns
    -------
    sitk.Image
        A binary mask image with the space between lungs filled.
    """
    # 1. Ensure binary (0/1)
    bin_img = sitk.BinaryThreshold(augmented_lung,
                                   lowerThreshold=1,
                                   upperThreshold=65535,
                                   insideValue=1,
                                   outsideValue=0)

    # 2. 3D morphological closing
    closed = sitk.BinaryMorphologicalClosing(
        bin_img,
        closing_radius,       # tuple of radii per dimension
        sitk.sitkBall         # ball-shaped structuring element
    )

    return closed

# Option to draw a mask manually on the image using openCV

drawing = False

def draw_polygon(event, x, y, flags, param):
    img, points, color = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(img, (x, y), 3, color, -1)

        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], color, 1)

def upscale_for_ui(img, target_max=1200):
    h, w = img.shape[:2]
    scale = target_max / max(h, w)

    if scale <= 1.0:
        return img, 1.0

    new_size = (int(w * scale), int(h * scale))
    img_up = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return img_up, scale


def manual_segmentation(image, output_path=None, brush_size=None):
    if image is None:
        raise IOError("Could not load image")

    def normalize_to_uint8(img):
        img = img.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-8)
        img *= 255.0
        return img.astype(np.uint8)

    # Normalize image for UI
    image_ui = normalize_to_uint8(image)

    # Upscale image for UI
    image_ui_big, scale = upscale_for_ui(image_ui, target_max=800)

    # OpenCV display image
    image_bgr = cv2.cvtColor(image_ui_big, cv2.COLOR_GRAY2BGR)
    image_display = image_bgr.copy()

    # Mask must match UI resolution
    mask = np.zeros(image_ui_big.shape, dtype=np.uint8)

    # State
    current_points = []
    drawing_hole = False

    # Colors
    FG_COLOR = (0, 255, 0)    # green
    HOLE_COLOR = (0, 0, 255)  # red

    cv2.namedWindow("Draw Polygon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw Polygon", image_display.shape[1], image_display.shape[0])

    cv2.setMouseCallback(
        "Draw Polygon",
        draw_polygon,
        param=[image_display, current_points, FG_COLOR]
    )

    print("Instructions:")
    print("- LEFT click: add vertex")
    print("- c: close polygon")
    print("- h: toggle hole mode")
    print("- z: undo last point")
    print("- r: reset")
    print("- s: save & exit")
    print("- q: quit")

    while True:
        cv2.imshow("Draw Polygon", image_display)
        key = cv2.waitKey(1) & 0xFF

        # Toggle hole / foreground
        if key == ord('h'):
            drawing_hole = not drawing_hole
            current_points.clear()
            image_display[:] = image_bgr.copy()

            mode = "HOLE" if drawing_hole else "FOREGROUND"
            print(f"Mode switched to {mode}")

            color = HOLE_COLOR if drawing_hole else FG_COLOR
            cv2.setMouseCallback(
                "Draw Polygon",
                draw_polygon,
                param=[image_display, current_points, color]
            )

        # Close polygon
        elif key == ord('c') and len(current_points) >= 3:
            pts = np.array(current_points, np.int32)

            # Close visual contour
            cv2.line(image_display, current_points[-1], current_points[0],
                     HOLE_COLOR if drawing_hole else FG_COLOR, 1)

            if drawing_hole:
                cv2.fillPoly(mask, [pts], 0)
                print("Hole polygon applied")
            else:
                cv2.fillPoly(mask, [pts], 255)
                print("Foreground polygon applied")

            current_points.clear()

        # Undo last vertex
        elif key == ord('z') and len(current_points) > 0:
            current_points.pop()
            image_display[:] = image_bgr.copy()

            color = HOLE_COLOR if drawing_hole else FG_COLOR
            for i, p in enumerate(current_points):
                cv2.circle(image_display, p, 3, color, -1)
                if i > 0:
                    cv2.line(image_display, current_points[i-1], p, color, 1)

        # Reset everything
        elif key == ord('r'):
            current_points.clear()
            mask[:] = 0
            image_display[:] = image_bgr.copy()
            drawing_hole = False
            print("Reset")

        # Save & exit
        elif key == ord('s'):
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap='gray')
            ax.set_title("Final Binary Mask")
            ax.axis("off")
            plt.show()

            cv2.destroyAllWindows()
            if scale != 1.0:
                mask_full_res = cv2.resize(mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            else:
                mask_full_res = mask

            return mask_full_res

        elif key == ord('q'):
            print("Exited without saving")
            break

    cv2.destroyAllWindows()
