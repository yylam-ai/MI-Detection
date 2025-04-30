import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
# from skimage.measure import approximate_polygon # Not strictly needed now but keep if desired later
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates, gaussian_filter as gaussian_filter_ndimage # Use alias to avoid name clash

# --- START: Copied from Script 1 ---
def find_max_intensity_anchor_points(image, start_pt_xy, end_pt_xy, apex_pt_xy,
                                     num_anchors_per_side=7,
                                     search_length=30, # Pixels perpendicular to search
                                     num_samples_along_search=50, # Samples along search line
                                     blur_sigma=1.5): # Optional blurring
    """
    Finds anchor points on the LV wall based on maximum intensity.
    EXPECTS AND RETURNS (x, y) coordinates.

    Args:
        image (np.ndarray): Grayscale input image.
        start_pt_xy (tuple): (x, y) coordinates of the start point.
        end_pt_xy (tuple): (x, y) coordinates of the end point.
        apex_pt_xy (tuple): (x, y) coordinates of the LV apex.
        num_anchors_per_side (int): Number of anchor points to find on each side.
        search_length (int): Total length of the perpendicular search line.
        num_samples_along_search (int): Number of points to sample along each search line.
        blur_sigma (float): Sigma for Gaussian blur. Set to 0 or None to disable.

    Returns:
        np.ndarray: An array of shape (2 * num_anchors_per_side, 2) containing
                    the (x, y) coordinates of the found anchor points, or None if error.
                    Points are ordered from start-side to end-side.
    """
    if blur_sigma is not None and blur_sigma > 0:
        image_processed = gaussian_filter_ndimage(image.astype(float), sigma=blur_sigma)
    else:
        image_processed = image.astype(float)

    start_pt = np.array(start_pt_xy, dtype=float)
    end_pt = np.array(end_pt_xy, dtype=float)
    apex_pt = np.array(apex_pt_xy, dtype=float)
    height, width = image_processed.shape

    found_anchors_xy = [] # Store as (x, y)

    # --- Process Left Side (Start -> Apex) ---
    vec_sa = apex_pt - start_pt
    len_sa = np.linalg.norm(vec_sa)
    if len_sa < 1e-6:
        print("Warning: Start and Apex points are too close.")
        unit_vec_sa = np.array([0.0, -1.0]) # Assume vertical if coincident (adjust based on typical apex location if needed)
    else:
        unit_vec_sa = vec_sa / len_sa

    perp_vec_sa = np.array([-unit_vec_sa[1], unit_vec_sa[0]])

    # print("Searching left side anchors...") # Optional debug print
    for i in range(1, num_anchors_per_side + 1):
        fraction = i / (num_anchors_per_side + 1)
        base_pt = start_pt + fraction * vec_sa
        search_start = base_pt - (search_length / 2.0) * perp_vec_sa
        search_end = base_pt + (search_length / 2.0) * perp_vec_sa
        x_coords = np.linspace(search_start[0], search_end[0], num_samples_along_search)
        y_coords = np.linspace(search_start[1], search_end[1], num_samples_along_search)

        try:
            # map_coordinates expects (row, col) i.e., (y, x)
            intensities = map_coordinates(image_processed,
                                          [y_coords, x_coords],
                                          order=1, mode='nearest', prefilter=False)
        except Exception as e:
            print(f" Error sampling intensities on left side, anchor {i}: {e}")
            continue

        if len(intensities) == 0: continue
        max_idx = np.argmax(intensities)
        anchor_pt_xy = np.array([x_coords[max_idx], y_coords[max_idx]])
        found_anchors_xy.append(anchor_pt_xy)

    # --- Process Right Side (Apex -> End) ---
    vec_ae = end_pt - apex_pt
    len_ae = np.linalg.norm(vec_ae)
    if len_ae < 1e-6:
        print("Warning: Apex and End points are too close.")
        unit_vec_ae = np.array([0.0, 1.0]) # Assume vertical if coincident
    else:
        unit_vec_ae = vec_ae / len_ae

    perp_vec_ae = np.array([-unit_vec_ae[1], unit_vec_ae[0]])

    # print("Searching right side anchors...") # Optional debug print
    for i in range(1, num_anchors_per_side + 1):
        fraction = i / (num_anchors_per_side + 1)
        base_pt = apex_pt + fraction * vec_ae
        search_start = base_pt - (search_length / 2.0) * perp_vec_ae
        search_end = base_pt + (search_length / 2.0) * perp_vec_ae
        x_coords = np.linspace(search_start[0], search_end[0], num_samples_along_search)
        y_coords = np.linspace(search_start[1], search_end[1], num_samples_along_search)

        try:
             # map_coordinates expects (row, col) i.e., (y, x)
            intensities = map_coordinates(image_processed,
                                          [y_coords, x_coords],
                                          order=1, mode='nearest', prefilter=False)
        except Exception as e:
            print(f" Error sampling intensities on right side, anchor {i}: {e}")
            continue

        if len(intensities) == 0: continue
        max_idx = np.argmax(intensities)
        anchor_pt_xy = np.array([x_coords[max_idx], y_coords[max_idx]])
        found_anchors_xy.append(anchor_pt_xy) # Add to the main list

    if len(found_anchors_xy) != 2 * num_anchors_per_side:
        print(f"Warning: Expected {2*num_anchors_per_side} anchors, found {len(found_anchors_xy)}. May indicate issues.")
        if len(found_anchors_xy) == 0:
             return None # Return None if no points found

    return np.array(found_anchors_xy) # Return as (x, y)
# --- END: Copied from Script 1 ---


# --- Global variable for point selection ---
points = []
fig_select = None
ax_select = None
img_select = None

# --- Helper Function for Manual Point Selection ---
def onclick(event):
    global points, ax_select
    if event.inaxes == ax_select:
        # Store as (x, y) internally during selection for consistency with plot events
        px, py = int(event.xdata), int(event.ydata)
        print(f'Point selected: x={px}, y={py}')
        points.append((px, py)) # Store as (x, y) -> (col, row) in the list
        ax_select.plot(px, py, 'r+', markersize=10)
        fig_select.canvas.draw()
        if len(points) == 3:
            print("START, APEX, END points selected. Close the selection window.")
            plt.close(fig_select)

def get_points(image):
    """Displays the image and waits for 3 clicks to define START, APEX, END."""
    global points, fig_select, ax_select, img_select
    points = [] # Reset points
    img_select = image
    print("Please click 3 points on the image in this order: START, APEX, END")
    fig_select, ax_select = plt.subplots(figsize=(8, 8))
    ax_select.imshow(image, cmap='gray')
    ax_select.set_title("Click START, APEX, END points (then close window)")
    fig_select.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True) # Wait until the window is closed

    if len(points) == 3:
        # Convert selected (x, y) points to (row, col) for internal use in Script 2
        start_pt_rc = (points[0][1], points[0][0]) # (y, x) -> (row, col)
        apex_pt_rc = (points[1][1], points[1][0])  # (y, x) -> (row, col)
        end_pt_rc = (points[2][1], points[2][0])    # (y, x) -> (row, col)
        return start_pt_rc, apex_pt_rc, end_pt_rc
    else:
        print("Error: Did not select 3 points.")
        return None, None, None

# --- NEW Function to create Initial Contour using Anchors ---
def create_initial_contour_with_anchors(image, start_pt_rc, apex_pt_rc, end_pt_rc,
                                        num_anchors_per_side=7, search_length=40,
                                        num_samples_along_search=60, blur_sigma=1.0):
    """
    Creates an initial contour by finding max intensity anchors and connecting them.

    Args:
        image (np.ndarray): Grayscale input image.
        start_pt_rc (tuple): (row, col) coordinates of the start point.
        apex_pt_rc (tuple): (row, col) coordinates of the apex point.
        end_pt_rc (tuple): (row, col) coordinates of the end point.
        num_anchors_per_side, search_length, etc: Parameters for anchor finding.

    Returns:
        np.ndarray: An array of shape (N, 2) with (row, col) coordinates for the
                    initial contour, or None if anchor finding fails.
    """
    print("Finding max intensity anchor points for initial contour...")

    # --- Convert points to (x, y) for find_max_intensity_anchor_points ---
    start_pt_xy = (start_pt_rc[1], start_pt_rc[0])
    apex_pt_xy = (apex_pt_rc[1], apex_pt_rc[0])
    end_pt_xy = (end_pt_rc[1], end_pt_rc[0])

    # --- Find anchor points ---
    anchor_points_xy = find_max_intensity_anchor_points(
        image, start_pt_xy, end_pt_xy, apex_pt_xy,
        num_anchors_per_side=num_anchors_per_side,
        search_length=search_length,
        num_samples_along_search=num_samples_along_search,
        blur_sigma=blur_sigma
    )

    if anchor_points_xy is None or len(anchor_points_xy) != 2 * num_anchors_per_side:
        print("Warning: Anchor point finding failed or returned unexpected number of points. Falling back to triangle.")
        # Fallback to simple triangle if anchor finding fails
        return create_initial_triangle(start_pt_rc, apex_pt_rc, end_pt_rc, num_points_per_side=20) # Use fewer points for triangle

    # --- Convert anchor points back to (row, col) ---
    # anchor_points_xy is shape (2*num_anchors, 2) with columns (x, y)
    # We want shape (2*num_anchors, 2) with columns (row, col) which is (y, x)
    anchor_points_rc = anchor_points_xy[:, ::-1] # Swap columns to get (y, x) -> (row, col)

    # --- Split anchors into left and right sides ---
    left_anchors_rc = anchor_points_rc[:num_anchors_per_side]
    right_anchors_rc = anchor_points_rc[num_anchors_per_side:]

    # --- Assemble the full contour in (row, col) format ---
    # Ensure points are numpy arrays with shape (1, 2) or (N, 2) for vstack
    start_pt_rc_arr = np.array(start_pt_rc).reshape(1, 2)
    apex_pt_rc_arr = np.array(apex_pt_rc).reshape(1, 2)
    end_pt_rc_arr = np.array(end_pt_rc).reshape(1, 2)

    # Order: start -> left anchors -> apex -> right anchors -> end
    # Left anchors are already ordered start->apex direction by find_max_intensity...
    # Right anchors are already ordered apex->end direction by find_max_intensity...
    initial_contour_points = [start_pt_rc_arr]
    if len(left_anchors_rc) > 0:
        initial_contour_points.append(left_anchors_rc)
    initial_contour_points.append(apex_pt_rc_arr)
    if len(right_anchors_rc) > 0:
        initial_contour_points.append(right_anchors_rc)
    initial_contour_points.append(end_pt_rc_arr)

    initial_contour = np.vstack(initial_contour_points)

    print(f"Initial contour created with {len(initial_contour)} points (including start, apex, end, and anchors).")
    return initial_contour


# --- Function to create Initial Triangle Contour (as fallback) ---
def create_initial_triangle(start_pt, apex_pt, end_pt, num_points_per_side=50):
    """Creates a denser contour along the triangle defined by the points."""
    # Linear interpolation between points (row, col format)
    side1_rows = np.linspace(start_pt[0], apex_pt[0], num_points_per_side)
    side1_cols = np.linspace(start_pt[1], apex_pt[1], num_points_per_side)
    side2_rows = np.linspace(apex_pt[0], end_pt[0], num_points_per_side)
    side2_cols = np.linspace(apex_pt[1], end_pt[1], num_points_per_side)

    # Combine sides (remove duplicate apex)
    rows = np.concatenate((side1_rows[:-1], side2_rows))
    cols = np.concatenate((side1_cols[:-1], side2_cols))

    initial_contour = np.array([rows, cols]).T
    return initial_contour

# --- Active Contour Evolution (Simplified Constraint) ---
def evolve_active_contour(image, initial_contour,
                          alpha=0.015, beta=10, gamma=0.001, # Parameters to TUNE
                          w_line=0, w_edge=1, # Added weights here for clarity
                          max_iterations=2500, convergence=0.1):
    """Evolves the active contour starting from the initial shape."""
    # Ensure image is float in [0, 1] range if not already
    if image.dtype != float:
        img_float = image.astype(float)
        # Check if normalization is needed
        img_max = np.max(img_float)
        if img_max > 1.0:
             img_float = img_float / img_max # Normalize to [0, 1]
    else:
         img_float = image
         # Also check float images in case they are not in [0, 1] range
         img_max = np.max(img_float)
         if img_max > 1.0:
             print("Warning: Input float image max value > 1.0. Normalizing.")
             img_float = img_float / img_max

    # Pre-smoothing the image is often beneficial for active contour
    img_smooth = gaussian(img_float, sigma=1, preserve_range=False)

    try:
        # REMOVED the coordinates='rc' argument
        snake = active_contour(img_smooth, # Use smoothed image
                               initial_contour,
                               alpha=alpha, beta=beta, gamma=gamma, # Contour properties
                               w_line=w_line, w_edge=w_edge,       # Image energy weights
                               max_num_iter=max_iterations, convergence=convergence)
        return snake
    except Exception as e:
        print(f"Error during active contour evolution: {e}")
        # Consider visualizing initial_contour and image properties here if debugging
        # plt.figure()
        # plt.imshow(img_smooth, cmap='gray') # Show smoothed image used
        # plt.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-')
        # plt.title("Initial Contour at Error")
        # plt.show()
        return None


# --- Fit Polynomials ---
def fit_polynomials(contour_points, apex_pt_orig_rc, poly_order=4):
    """
    Splits contour near the original apex and fits polynomials.
    Fits X = P(Y) i.e., Col = P(Row). Expects (row, col) points.
    """
    if contour_points is None or len(contour_points) < 10:
        print("Error: Not enough contour points for fitting.")
        return None, None, None, None

    # Find point on contour closest to the manually selected apex (in row, col)
    distances = np.sqrt(np.sum((contour_points - np.array(apex_pt_orig_rc))**2, axis=1))
    apex_index = np.argmin(distances)

    # Avoid splitting exactly at the ends
    if apex_index == 0 and len(contour_points) > 1 : apex_index = 1
    if apex_index == len(contour_points) - 1 and len(contour_points) > 1: apex_index = len(contour_points) - 2

    # Split into two segments: start->apex and apex->end
    points_left = contour_points[0:apex_index+1]  # Includes the contour point closest to apex
    points_right = contour_points[apex_index:]    # Also includes the contour point closest to apex

    # Ensure segments have enough points for the polynomial order
    min_points_needed = poly_order + 1
    if len(points_left) < min_points_needed or len(points_right) < min_points_needed:
        print(f"Error: Not enough points in left ({len(points_left)}) or right ({len(points_right)}) segments for polyfit (order {poly_order}). Need {min_points_needed}.")
        # Optionally, try reducing poly_order dynamically here if possible
        return None, None, points_left, points_right # Return original points even if fit fails

    coeffs_left = None
    coeffs_right = None

    try:
        # Fit Col = P(Row) -> polyfit(rows, cols, order)
        coeffs_left = np.polyfit(points_left[:, 0], points_left[:, 1], poly_order)
        coeffs_right = np.polyfit(points_right[:, 0], points_right[:, 1], poly_order)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error during polynomial fitting: {e}.")
        # Return original points even if fit fails
        return None, None, points_left, points_right

    return coeffs_left, coeffs_right, points_left, points_right


# --- Segment the Fitted Polynomials ---
def segment_polynomials(coeffs, side_points_rc, num_colored_segments=3):
    """
    Generates points along the fitted polynomial (Col=P(Row)) and divides them into segments.
    Returns num_colored_segments + 1 segments, where the first segment
    corresponds to the apex region (intended to be white).
    Ensures points are ordered APEX -> BASAL for segmentation.

    Args:
        coeffs (np.ndarray): Polynomial coefficients (for col = P(row)).
        side_points_rc (np.ndarray): The original contour points for this side (row, col).
        num_colored_segments (int): Number of functional segments (e.g., 3 for basal/mid/apical).

    Returns:
        list: A list of arrays, where each array contains the (row, col) points
              for one segment along the smooth polynomial curve. The list will
              contain num_colored_segments + 1 arrays. The first array is the apex segment.
    """
    if coeffs is None or len(side_points_rc) < 2:
        print("Warning: Cannot segment polynomials - invalid input.")
        return []

    num_total_segments = num_colored_segments + 1 # Add 1 for the white apex segment

    # Determine the range of rows from the original side points
    min_row = np.min(side_points_rc[:, 0])
    max_row = np.max(side_points_rc[:, 0])

    if max_row <= min_row: # Handle case of horizontal segment or single point
        print("Warning: Cannot generate smooth curve - row range too small.")
        # Fallback: return the original points split somehow? Or just empty.
        # Let's try returning the original points as a single segment.
        return [side_points_rc]

    # Generate points along the polynomial P(Row)
    smooth_rows = np.linspace(min_row, max_row, 200) # Generate sufficient points for smoothness
    smooth_cols = np.polyval(coeffs, smooth_rows)
    smooth_curve_points = np.vstack((smooth_rows, smooth_cols)).T # Combine into (row, col) format

    # --- CRITICAL: Ensure points are ordered from APEX outwards for segmentation ---
    # The 'apex' end of this side_points segment is the one closest to the overall apex_index.
    # The 'basal' end is the one closest to the start/end points.
    # Let's determine which end of side_points_rc corresponds to the apex.
    # The apex point on the contour for this side is likely side_points_rc[0] or side_points_rc[-1]
    # depending on how the split occurred. Let's assume the polynomial fit preserves this.
    # We need to order smooth_curve_points such that index 0 is near the apex end.

    # Find which end of the *original* side_points_rc is the apex end
    # (This relies on how fit_polynomials split the contour)
    # Heuristic: Usually, the split point is included in both left/right.
    # For points_left, apex is likely at index -1. For points_right, apex is likely at index 0.
    # A more robust way: Find which end of smooth_curve_points is closer to side_points_rc's apex end.
    # Find the apex point within side_points_rc (should be the shared point from the split)
    shared_apex_point_on_side = side_points_rc[0] if np.array_equal(side_points_rc[0], side_points_rc[1]) else side_points_rc[-1]
    # Find which end of smooth_curve_points is closer to this shared apex point
    dist_start_to_apex = np.linalg.norm(smooth_curve_points[0] - shared_apex_point_on_side)
    dist_end_to_apex = np.linalg.norm(smooth_curve_points[-1] - shared_apex_point_on_side)

    if dist_start_to_apex > dist_end_to_apex:
         # The generated curve starts near the base and ends near the apex. Reverse it.
         smooth_curve_points = smooth_curve_points[::-1]
    # Now smooth_curve_points[0] should be the point closest to the apex end of this side's fit

    # --- Segment the APEX -> BASAL ordered curve ---
    total_smooth_points = len(smooth_curve_points)
    segments = []

    if total_smooth_points < num_total_segments:
        print(f"Warning: Not enough points ({total_smooth_points}) generated to create {num_total_segments} segments. Returning fewer segments.")
        # Fallback: create as many single-point segments as possible? Or just return what we have?
        # Let's distribute points somewhat evenly.
        indices = np.linspace(0, total_smooth_points, num_total_segments + 1, dtype=int)
        for i in range(num_total_segments):
             seg = smooth_curve_points[indices[i]:indices[i+1]]
             if len(seg) > 0:
                  segments.append(seg)

    elif total_smooth_points > 0 :
         # Use integer division, distribute remainder points possibly to basal segments
         points_per_segment = total_smooth_points // num_total_segments
         remainder = total_smooth_points % num_total_segments

         start_idx = 0
         for i in range(num_total_segments):
             # Add one extra point to the first 'remainder' segments (which are the apex ones)
             current_segment_length = points_per_segment + (1 if i < remainder else 0)
             end_idx = start_idx + current_segment_length

             # Handle edge case if calculation leads to index out of bounds (shouldn't happen with proper remainder logic)
             end_idx = min(end_idx, total_smooth_points)

             segment_points = smooth_curve_points[start_idx:end_idx]

             if len(segment_points) > 0:
                segments.append(segment_points)
             elif start_idx < total_smooth_points: # Append empty segment if expected but no points left
                segments.append(np.array([]).reshape(0,2))


             start_idx = end_idx

             # Safety break if somehow start_idx exceeds total points
             if start_idx >= total_smooth_points and i < num_total_segments - 1:
                 print(f"Warning: Ran out of points unexpectedly during segmentation at segment {i+1}")
                 # Add empty segments for the remainder
                 while len(segments) < num_total_segments:
                     segments.append(np.array([]).reshape(0,2))
                 break

    # Ensure we always return num_total_segments lists, even if some are empty
    while len(segments) < num_total_segments:
       segments.append(np.array([]).reshape(0,2))

    # segments[0] should be apex (white)
    # segments[1] should be apical (e.g., lime/yellow)
    # segments[2] should be mid (e.g., red/blue)
    # segments[3] should be basal (e.g., cyan/magenta)
    return segments


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Image ---
    try:
        # image_path = 'complete_HMC_QU/A4C/folds/fold_0/inference_data/ES0001 _4CH_1.npy'
        image_path = 'complete_HMC_QU/A2C/folds/fold_0/inference_data/ES0001_CH2_1.npy'
        image_data = np.load(image_path, allow_pickle=True).item()['X']
        # Adjust based on actual structure of your .npy file
        if isinstance(image_data, np.ndarray):
            # Assuming it's just the image array directly or needs reshaping
            if image_data.ndim == 3 and image_data.shape[-1] in [1, 3]: # e.g. (H, W, C)
                 img_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if image_data.shape[-1] == 3 else image_data[:,:,0]
            elif image_data.ndim == 2: # Already grayscale
                 img_gray = image_data
            else: # Attempt reshape if it looks like (C, H, W) or similar
                 print("shape:", image_data.shape)
                 img_gray = image_data.reshape(-1, 224, 224)[0] # get first frame
        elif isinstance(image_data, dict): # Like the original script assumed
             img_gray = image_data.item()['X'].reshape(-1, 224, 224)[0] # MODIFY 224, 224 if needed
        else:
             raise ValueError("Unsupported .npy file structure")

        # Normalize image to float [0, 1] which skimage often prefers
        if img_gray.dtype != float:
             img_gray = img_gray.astype(float)
        img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray) + 1e-6) # Normalize

    except FileNotFoundError:
         print(f"Error: Image file not found at {image_path}")
         exit()
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        exit()


    # --- 1. Get Key Points ---
    # Returns points as (row, col)
    start_pt_rc, apex_pt_rc, end_pt_rc = get_points(img_gray)
    if start_pt_rc is None:
        print("Exiting.")
        exit()

    # --- Define Parameters for Anchor Finding ---
    anchor_params = {
        'num_anchors_per_side': 7,
        'search_length': 40, # Adjust based on LV size/image resolution
        'num_samples_along_search': 60,
        'blur_sigma': 1.0 # Gaussian blur sigma for anchor search (can be None)
    }

    # --- 2. Create Initial Contour using Anchors ---
    # Pass (row, col) points; function handles conversion for anchor finding
    initial_contour = create_initial_contour_with_anchors(
        img_gray, start_pt_rc, apex_pt_rc, end_pt_rc,
        **anchor_params
    )

    if initial_contour is None or len(initial_contour) < 3:
         print("Error: Failed to create a valid initial contour. Exiting.")
         exit()

    # --- 3. Evolve Active Contour ---
    print("Evolving Active Contour... (This may take a moment)")
    # --- !!! PARAMETER TUNING IS CRITICAL HERE !!! ---
    # Adjust alpha, beta, gamma, w_line, w_edge based on image contrast and desired smoothness
    evolved_contour = evolve_active_contour(img_gray, initial_contour,
                                            alpha=0.01,      # Membrane stiffness (higher = smoother)
                                            beta=5.0,       # Balloon force (higher = more expansion/contraction based on image gradient)
                                            gamma=0.001,     # Step size for iteration
                                            # w_line=0,        # Intensity attraction (set to 0 if using edges)
                                            # w_edge=1,        # Edge attraction (gradient magnitude)
                                            max_iterations=1500,# Max steps
                                            convergence=0.5) # Stop if change is small

    if evolved_contour is None:
        print("Active contour failed. Exiting.")
        # Keep the plot open showing the initial contour if debugging needed
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_gray, cmap='gray')
        ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=5, label='Start')
        ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=5, label='Apex')
        ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=5, label='End')
        # Plot anchor points if found, convert back to (x,y) for plotting
        # Note: anchor_points_xy might not be available here if fallback occurred
        # We would need to modify create_initial_contour_with_anchors to return them if needed for viz
        ax.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-', lw=1.5, label='Initial Contour (Anchor-based/Fallback)')
        ax.set_title("Active Contour Failed - Showing Initial Contour")
        ax.legend()
        plt.show()
        exit()

    # --- 4. Fit Polynomials ---
    # Pass the originally selected apex point (row, col) to help find the split
    coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(initial_contour, apex_pt_rc, poly_order=4) # poly_order can be tuned


    # --- 5. Segment Polynomials (now returns 4 segments per side) ---
    segments_left = []
    segments_right = []
    num_colored_segments_def = 3 # Define the number of *colored* segments needed (apical, mid, basal)

    if coeffs_l is not None and points_l_orig is not None:
        segments_left = segment_polynomials(coeffs_l, points_l_orig, num_colored_segments=num_colored_segments_def)
    else:
         print("Warning: Cannot segment left side - polynomial fit failed.")

    if coeffs_r is not None and points_r_orig is not None:
        segments_right = segment_polynomials(coeffs_r, points_r_orig, num_colored_segments=num_colored_segments_def)
    else:
         print("Warning: Cannot segment right side - polynomial fit failed.")


    # --- 6. Visualization (Mimicking Figure 3) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5)) # Adjusted size slightly
    plt.suptitle("Endocardial Boundary Extraction Stages", fontsize=16)

    # --- Panel 1: Input ---
    ax = axes[0]
    ax.imshow(img_gray, cmap='gray')
    ax.set_title("Input Echocardiography")
    ax.axis('off')

    # --- Panel 2: Initial Contour (Now Anchor-Based) ---
    ax = axes[1]
    ax.imshow(img_gray, cmap='gray')
    # Plot points in (x, y) format: point[1] is col (x), point[0] is row (y)
    ax.plot(start_pt_rc[1], start_pt_rc[0], 'ro', markersize=5)
    ax.text(start_pt_rc[1], start_pt_rc[0] + 15, 'START', color='white', ha='center')
    ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'ro', markersize=5)
    ax.text(apex_pt_rc[1], apex_pt_rc[0] - 15, 'APEX', color='white', ha='center', va='bottom')
    ax.plot(end_pt_rc[1], end_pt_rc[0], 'ro', markersize=5)
    ax.text(end_pt_rc[1], end_pt_rc[0] + 15, 'END', color='white', ha='center')
    # Plot initial contour (row, col) -> (y, x)
    ax.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-', lw=1.5, label='Initial Contour (Anchors)')
    ax.set_title("Initial Contour (Anchor-Based)")
    ax.axis('off')

    # --- Panel 3: Evolved Active Contour ---
    # ax = axes[2]
    # ax.imshow(img_gray, cmap='gray')
    # # Plot evolved contour (row, col) -> (y, x)
    # ax.plot(evolved_contour[:, 1], evolved_contour[:, 0], 'r-', lw=2, label='Evolved Contour')
    # # Plot key points again for reference
    # ax.plot(start_pt_rc[1], start_pt_rc[0], 'bo', markersize=5)
    # # ax.text(start_pt_rc[1], start_pt_rc[0] + 15, 'START', color='white', ha='center') # Text optional here
    # ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'bo', markersize=5)
    # # ax.text(apex_pt_rc[1], apex_pt_rc[0] - 15, 'APEX', color='white', ha='center', va='bottom')
    # ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=5)
    # # ax.text(end_pt_rc[1], end_pt_rc[0] + 15, 'END', color='white', ha='center')
    # ax.set_title("Evolved Active Contour")
    # ax.axis('off')

    # --- Panel 4: Active Polynomials (Segmented with White Apex) ---
    ax = axes[2]
    ax.imshow(img_gray, cmap='gray')

    # Define colors for the 3 functional segments (Apical, Mid, Basal)
    # Order: colors[0]=Apical, colors[1]=Mid, colors[2]=Basal
    # These correspond to segments[1], segments[2], segments[3] respectively. segments[0] is white.
    colors_left_functional = ['cyan', 'lime', 'red']    # Basal, Mid, Apical
    colors_right_functional = ['blue', 'yellow', 'magenta'] # Apical, Mid, Basal
    expected_num_segments = num_colored_segments_def + 1

    # Plot Left Segments
    if len(segments_left) == expected_num_segments:
        for i, seg in enumerate(segments_left):
            if seg is not None and len(seg) > 1:
                # seg is (row, col), plot as (col, row) -> (x, y)
                if i == 0: # First segment is apex
                    color = 'white'
                else: # Map functional segments (index 1, 2, 3) to colors (index 0, 1, 2)
                    color_index = i - 1
                    if color_index < len(colors_left_functional):
                         color = colors_left_functional[color_index]
                    else: color = 'gray' # Fallback
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
            # else: print(f"Left segment {i} is empty or too short.") # Debug print
    elif len(segments_left) > 0: # Fallback if segmentation didn't produce exactly expected parts
         print(f"Warning: Plotting left side with {len(segments_left)} segments instead of {expected_num_segments}.")
         # Just plot all available segments in sequence (no explicit white apex assumed)
         for i, seg in enumerate(segments_left):
              if seg is not None and len(seg) > 1:
                    color = colors_left_functional[i % len(colors_left_functional)] # Cycle through colors
                    ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)

    # Plot Right Segments
    if len(segments_right) == expected_num_segments:
        for i, seg in enumerate(segments_right):
            if seg is not None and len(seg) > 1:
                # seg is (row, col), plot as (col, row) -> (x, y)
                if i == 3: # Last segment is apex
                    color = 'white'
                else: # Map functional segments (index 1, 2, 3) to colors (index 0, 1, 2)
                    color_index = i - 1
                    if color_index < len(colors_right_functional):
                         color = colors_right_functional[color_index]
                    else: color = 'gray' # Fallback
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
            # else: print(f"Right segment {i} is empty or too short.") # Debug print
    elif len(segments_right) > 0: # Fallback
         print(f"Warning: Plotting right side with {len(segments_right)} segments instead of {expected_num_segments}.")
         for i, seg in enumerate(segments_right):
              if seg is not None and len(seg) > 1:
                    color = colors_right_functional[i % len(colors_right_functional)] # Cycle through colors
                    ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)


    ax.set_title("Active Polynomials (Segmented)")
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()