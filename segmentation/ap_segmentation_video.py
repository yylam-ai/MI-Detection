import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
# from skimage.measure import approximate_polygon # Not strictly needed now but keep if desired later
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates, gaussian_filter as gaussian_filter_ndimage # Use alias to avoid name clash
import os # For checking file existence

# --- START: Copied from Script 1 (No changes needed here) ---
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
        # Ensure input is float for gaussian_filter
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
        # print("Warning: Start and Apex points are too close.")
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

        # Clip coordinates to image bounds before sampling
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)

        try:
            # map_coordinates expects (row, col) i.e., (y, x)
            intensities = map_coordinates(image_processed,
                                          [y_coords, x_coords],
                                          order=1, mode='nearest', prefilter=False)
        except Exception as e:
            # print(f" Error sampling intensities on left side, anchor {i}: {e}")
            continue # Skip this anchor if sampling fails

        if len(intensities) == 0: continue
        max_idx = np.argmax(intensities)
        anchor_pt_xy = np.array([x_coords[max_idx], y_coords[max_idx]])
        found_anchors_xy.append(anchor_pt_xy)

    # --- Process Right Side (Apex -> End) ---
    vec_ae = end_pt - apex_pt
    len_ae = np.linalg.norm(vec_ae)
    if len_ae < 1e-6:
        # print("Warning: Apex and End points are too close.")
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

        # Clip coordinates to image bounds before sampling
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)

        try:
             # map_coordinates expects (row, col) i.e., (y, x)
            intensities = map_coordinates(image_processed,
                                          [y_coords, x_coords],
                                          order=1, mode='nearest', prefilter=False)
        except Exception as e:
            # print(f" Error sampling intensities on right side, anchor {i}: {e}")
            continue # Skip this anchor if sampling fails

        if len(intensities) == 0: continue
        max_idx = np.argmax(intensities)
        anchor_pt_xy = np.array([x_coords[max_idx], y_coords[max_idx]])
        found_anchors_xy.append(anchor_pt_xy) # Add to the main list

    if len(found_anchors_xy) != 2 * num_anchors_per_side:
        # print(f"Warning: Expected {2*num_anchors_per_side} anchors, found {len(found_anchors_xy)}. May indicate issues.")
        if len(found_anchors_xy) == 0:
             return None # Return None if no points found

    return np.array(found_anchors_xy) # Return as (x, y)
# --- END: Copied from Script 1 ---

# --- Global variable for point selection (no changes needed) ---
points = []
fig_select = None
ax_select = None
img_select = None

# --- Helper Function for Manual Point Selection (no changes needed) ---
def onclick(event):
    global points, ax_select
    if event.inaxes == ax_select:
        px, py = int(event.xdata), int(event.ydata)
        print(f'Point selected: x={px}, y={py}')
        points.append((px, py))
        ax_select.plot(px, py, 'r+', markersize=10)
        fig_select.canvas.draw()
        if len(points) == 3:
            print("START, APEX, END points selected. Close the selection window.")
            plt.close(fig_select)

def get_points(image):
    """Displays the image and waits for 3 clicks to define START, APEX, END."""
    global points, fig_select, ax_select, img_select
    points = []
    img_select = image
    print("Please click 3 points on the image in this order: START, APEX, END")
    fig_select, ax_select = plt.subplots(figsize=(8, 8))
    ax_select.imshow(image, cmap='gray')
    ax_select.set_title("Click START, APEX, END points (then close window)")
    fig_select.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    if len(points) == 3:
        # Convert selected (x, y) points to (row, col) for internal use
        start_pt_rc = (points[0][1], points[0][0]) # (y, x) -> (row, col)
        apex_pt_rc = (points[1][1], points[1][0])  # (y, x) -> (row, col)
        end_pt_rc = (points[2][1], points[2][0])    # (y, x) -> (row, col)
        return start_pt_rc, apex_pt_rc, end_pt_rc
    else:
        print("Error: Did not select 3 points.")
        return None, None, None

# --- Function create_initial_contour_with_anchors (no changes needed) ---
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
    # print("Finding max intensity anchor points for initial contour...") # Less verbose

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

    if anchor_points_xy is None: # Simpler check
        print("Warning: Anchor point finding failed. Falling back to triangle.")
        return create_initial_triangle(start_pt_rc, apex_pt_rc, end_pt_rc, num_points_per_side=20)

    num_expected_anchors = 2 * num_anchors_per_side
    if len(anchor_points_xy) != num_expected_anchors:
         print(f"Warning: Found {len(anchor_points_xy)} anchors, expected {num_expected_anchors}. Using found points.")
         # Proceeding even with fewer anchors might be better than triangle fallback if some were found
         if len(anchor_points_xy) == 0:
              print("No anchors found. Falling back to triangle.")
              return create_initial_triangle(start_pt_rc, apex_pt_rc, end_pt_rc, num_points_per_side=20)
         # Adjust num_anchors_per_side based on what was found for splitting logic
         # This simple split assumes they are still ordered left then right,
         # which might be wrong if many failed on one side. A more robust split might be needed.
         num_found_left = min(num_anchors_per_side, len(anchor_points_xy)) # Approximate
         num_found_right = len(anchor_points_xy) - num_found_left
    else:
        num_found_left = num_anchors_per_side
        num_found_right = num_anchors_per_side


    # --- Convert anchor points back to (row, col) ---
    anchor_points_rc = anchor_points_xy[:, ::-1] # Swap columns to get (y, x) -> (row, col)

    # --- Split anchors into left and right sides ---
    # Use num_found_left determined above
    left_anchors_rc = anchor_points_rc[:num_found_left]
    right_anchors_rc = anchor_points_rc[num_found_left:]

    # --- Assemble the full contour in (row, col) format ---
    start_pt_rc_arr = np.array(start_pt_rc).reshape(1, 2)
    apex_pt_rc_arr = np.array(apex_pt_rc).reshape(1, 2)
    end_pt_rc_arr = np.array(end_pt_rc).reshape(1, 2)

    initial_contour_points = [start_pt_rc_arr]
    if len(left_anchors_rc) > 0:
        initial_contour_points.append(left_anchors_rc)
    initial_contour_points.append(apex_pt_rc_arr)
    if len(right_anchors_rc) > 0:
        initial_contour_points.append(right_anchors_rc)
    initial_contour_points.append(end_pt_rc_arr)

    initial_contour = np.vstack(initial_contour_points)

    # print(f"Initial contour created with {len(initial_contour)} points.") # Less verbose
    return initial_contour

# --- Function create_initial_triangle (no changes needed) ---
def create_initial_triangle(start_pt, apex_pt, end_pt, num_points_per_side=50):
    """Creates a denser contour along the triangle defined by the points."""
    side1_rows = np.linspace(start_pt[0], apex_pt[0], num_points_per_side)
    side1_cols = np.linspace(start_pt[1], apex_pt[1], num_points_per_side)
    side2_rows = np.linspace(apex_pt[0], end_pt[0], num_points_per_side)
    side2_cols = np.linspace(apex_pt[1], end_pt[1], num_points_per_side)
    rows = np.concatenate((side1_rows[:-1], side2_rows))
    cols = np.concatenate((side1_cols[:-1], side2_cols))
    initial_contour = np.array([rows, cols]).T
    return initial_contour

# --- Function evolve_active_contour (no changes needed) ---
def evolve_active_contour(image, initial_contour,
                          alpha=0.015, beta=10, gamma=0.001,
                          w_line=0, w_edge=1,
                          max_iterations=2500, convergence=0.1):
    """Evolves the active contour starting from the initial shape."""
    if image.dtype != float:
        img_float = image.astype(float)
        img_max = np.max(img_float)
        if img_max > 1.0: img_float = img_float / img_max
    else:
         img_float = image
         img_max = np.max(img_float)
         if img_max > 1.0: img_float = img_float / img_max

    img_smooth = gaussian(img_float, sigma=1, preserve_range=False) # sigma=1 might be too small, consider 2 or 3

    try:
        snake = active_contour(img_smooth,
                               initial_contour,
                               alpha=alpha, beta=beta, gamma=gamma,
                               w_line=w_line, w_edge=w_edge,
                               max_num_iter=max_iterations, convergence=convergence)
        return snake
    except ValueError as ve:
        # Catch specific error related to points outside image bounds
        if "Points are outside image bounds" in str(ve):
            print(f"Error during active contour evolution: {ve}")
            print("Initial contour points may be invalid or too close to edge.")
            # Optional: Visualize the problematic initial contour
            # fig_err, ax_err = plt.subplots()
            # ax_err.imshow(img_smooth, cmap='gray')
            # ax_err.plot(initial_contour[:, 1], initial_contour[:, 0], 'r.-', label='Initial Contour')
            # ax_err.plot(initial_contour[0, 1], initial_contour[0, 0], 'go', label='Start')
            # ax_err.plot(initial_contour[-1, 1], initial_contour[-1, 0], 'bo', label='End')
            # ax_err.set_xlim(0, img_smooth.shape[1])
            # ax_err.set_ylim(img_smooth.shape[0], 0)
            # ax_err.set_title("Active Contour Error: Invalid Initial Points?")
            # ax_err.legend()
            # plt.show()
        else:
             print(f"Unexpected error during active contour evolution: {ve}")
        return None # Return None on failure
    except Exception as e:
        print(f"Error during active contour evolution: {e}")
        return None

# --- Function fit_polynomials (no changes needed) ---
def fit_polynomials(contour_points, apex_pt_orig_rc, poly_order=4):
    """
    Splits contour near the original apex and fits polynomials.
    Fits X = P(Y) i.e., Col = P(Row). Expects (row, col) points.
    """
    if contour_points is None or len(contour_points) < 10:
        # print("Error: Not enough contour points for fitting.") # Less verbose
        return None, None, None, None

    # Find point on contour closest to the manually selected apex (in row, col)
    distances = np.sqrt(np.sum((contour_points - np.array(apex_pt_orig_rc))**2, axis=1))
    apex_index = np.argmin(distances)

    # Adjust index slightly away from ends if needed to ensure enough points on each side
    min_points_needed = poly_order + 1
    if apex_index < min_points_needed // 2 :
        apex_index = min(min_points_needed // 2, len(contour_points) - 1)
    if apex_index > len(contour_points) - 1 - (min_points_needed // 2):
        apex_index = max(len(contour_points) - 1 - (min_points_needed // 2) , 0)


    # Split into two segments: start->apex and apex->end
    points_left = contour_points[0:apex_index+1]
    points_right = contour_points[apex_index:]

    # Ensure segments have enough points for the polynomial order
    if len(points_left) < min_points_needed or len(points_right) < min_points_needed:
        # print(f"Warning: Not enough points in left ({len(points_left)}) or right ({len(points_right)}) segments for polyfit (order {poly_order}). Need {min_points_needed}.")
        # Try fitting with lower order? Or just return None. Let's return None.
        return None, None, points_left, points_right # Return original points even if fit fails

    coeffs_left = None
    coeffs_right = None

    try:
        # Fit Col = P(Row) -> polyfit(rows, cols, order)
        # Check for near-vertical segments where polyfit(row, col) is unstable
        if np.ptp(points_left[:, 0]) < 1e-6: # Check range of row values
             print("Warning: Left segment near-vertical, cannot fit Col=P(Row).")
             return None, None, points_left, points_right # Indicate failure
        coeffs_left = np.polyfit(points_left[:, 0], points_left[:, 1], poly_order)

        if np.ptp(points_right[:, 0]) < 1e-6: # Check range of row values
             print("Warning: Right segment near-vertical, cannot fit Col=P(Row).")
             return None, None, points_left, points_right # Indicate failure
        coeffs_right = np.polyfit(points_right[:, 0], points_right[:, 1], poly_order)

    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error during polynomial fitting: {e}.")
        return None, None, points_left, points_right

    return coeffs_left, coeffs_right, points_left, points_right

# --- Function segment_polynomials (no changes needed) ---
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
    if coeffs is None or side_points_rc is None or len(side_points_rc) < 2:
        # print("Warning: Cannot segment polynomials - invalid input.") # Less verbose
        return []

    num_total_segments = num_colored_segments + 1 # Add 1 for the white apex segment

    # Determine the range of rows from the original side points
    min_row = np.min(side_points_rc[:, 0])
    max_row = np.max(side_points_rc[:, 0])

    # Check for very small row range (near horizontal or single point)
    if max_row - min_row < 1e-6:
        # print("Warning: Cannot generate smooth curve - row range too small.")
        # Fallback: Distribute original points into segments as best as possible
        total_orig_points = len(side_points_rc)
        segments = []
        indices = np.linspace(0, total_orig_points, num_total_segments + 1, dtype=int)
        for i in range(num_total_segments):
             seg = side_points_rc[indices[i]:indices[i+1]]
             if len(seg) > 0:
                  segments.append(seg)
             else: # Ensure correct number of segments even if empty
                  segments.append(np.array([]).reshape(0,2))
        while len(segments) < num_total_segments:
             segments.append(np.array([]).reshape(0,2))
        # Need to figure out apex orientation still - this fallback is basic
        # Assume side_points_rc[0] or [-1] is apex end based on context (hard without knowing left/right)
        # For simplicity in fallback, just return the split points. Orientation might be wrong.
        return segments


    # Generate points along the polynomial P(Row)
    num_smooth_points = max(50, len(side_points_rc) * 2) # Generate more points than original
    smooth_rows = np.linspace(min_row, max_row, num_smooth_points)
    smooth_cols = np.polyval(coeffs, smooth_rows)
    smooth_curve_points = np.vstack((smooth_rows, smooth_cols)).T # Combine into (row, col) format

    # --- Ensure points are ordered from APEX outwards for segmentation ---
    # The apex point on this side *should* be either side_points_rc[0] or side_points_rc[-1]
    # due to the splitting logic in fit_polynomials.
    # Find which end of smooth_curve_points is closer to the ends of side_points_rc.
    dist_smooth_start_to_side_start = np.linalg.norm(smooth_curve_points[0] - side_points_rc[0])
    dist_smooth_start_to_side_end = np.linalg.norm(smooth_curve_points[0] - side_points_rc[-1])

    # Assume the apex point on this side is side_points_rc[-1] if it's the 'left' segment fit,
    # and side_points_rc[0] if it's the 'right' segment fit (based on split logic).
    # This requires knowing if we are processing left or right, which isn't passed directly.
    # Alternative Heuristic: Assume the apex end is the one with the minimum row value (usually true for A4C)
    apex_end_index_in_side = np.argmin(side_points_rc[:, 0]) # Index of lowest row value point

    # Find which end of the smooth curve is closer to this estimated apex end
    apex_point_on_side = side_points_rc[apex_end_index_in_side]
    dist_smooth_start_to_apex_end = np.linalg.norm(smooth_curve_points[0] - apex_point_on_side)
    dist_smooth_end_to_apex_end = np.linalg.norm(smooth_curve_points[-1] - apex_point_on_side)

    if dist_smooth_start_to_apex_end > dist_smooth_end_to_apex_end:
         # The generated curve starts near the base and ends near the apex. Reverse it.
         smooth_curve_points = smooth_curve_points[::-1]
    # Now smooth_curve_points[0] should be the point closest to the apex end of this side's fit

    # --- Segment the APEX -> BASAL ordered curve ---
    total_smooth_points = len(smooth_curve_points)
    segments = []

    if total_smooth_points < num_total_segments:
        # print(f"Warning: Not enough points ({total_smooth_points}) generated to create {num_total_segments} segments. Returning fewer segments.")
        # Fallback: Distribute points somewhat evenly.
        indices = np.linspace(0, total_smooth_points, num_total_segments + 1, dtype=int)
        for i in range(num_total_segments):
             seg = smooth_curve_points[indices[i]:indices[i+1]]
             if len(seg) > 0: segments.append(seg)
             else: segments.append(np.array([]).reshape(0,2)) # Add empty segment placeholder
    elif total_smooth_points > 0 :
         points_per_segment = total_smooth_points // num_total_segments
         remainder = total_smooth_points % num_total_segments
         start_idx = 0
         for i in range(num_total_segments):
             current_segment_length = points_per_segment + (1 if i < remainder else 0)
             end_idx = start_idx + current_segment_length
             end_idx = min(end_idx, total_smooth_points) # Ensure valid index
             segment_points = smooth_curve_points[start_idx:end_idx]
             if len(segment_points) > 0: segments.append(segment_points)
             else: segments.append(np.array([]).reshape(0,2)) # Add empty segment placeholder
             start_idx = end_idx
             if start_idx >= total_smooth_points and i < num_total_segments - 1:
                 # print(f"Warning: Ran out of points unexpectedly during segmentation at segment {i+1}")
                 break # Exit loop early if no more points

    # Ensure we always return num_total_segments lists
    while len(segments) < num_total_segments:
       segments.append(np.array([]).reshape(0,2))

    return segments


# --- NEW: KLT Tracking Function ---
def track_points_klt(prev_frame_gray, current_frame_gray, prev_points_rc,
                     lk_params, fb_error_threshold=1.0):
    """
    Tracks points from prev_frame to current_frame using KLT with forward-backward error check.

    Args:
        prev_frame_gray (np.ndarray): Previous grayscale frame (uint8).
        current_frame_gray (np.ndarray): Current grayscale frame (uint8).
        prev_points_rc (list): List of (row, col) tuples for points in the previous frame.
        lk_params (dict): Parameters for cv2.calcOpticalFlowPyrLK.
        fb_error_threshold (float): Maximum allowed distance between original point and
                                    backward-tracked point (in pixels).

    Returns:
        tuple: (current_points_rc, validity_mask)
            current_points_rc (list): List of tracked (row, col) tuples in the current frame.
                                      Points that failed tracking might be None or previous position.
            validity_mask (list): List of booleans indicating if tracking was successful
                                  and passed the FB error check for each point.
    """
    if not prev_points_rc:
        return [], []

    # --- Convert points to format required by OpenCV KLT: (N, 1, 2) float32 (x, y) ---
    # Input is [(r0, c0), (r1, c1), ...] -> [(c0, r0), (c1, r1), ...]
    prev_points_xy_list = [(p[1], p[0]) for p in prev_points_rc]
    p0_xy = np.array(prev_points_xy_list, dtype=np.float32).reshape(-1, 1, 2)

    # --- Forward Tracking ---
    p1_xy, st_forward, err_forward = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, p0_xy, None, **lk_params)

    # --- Backward Tracking (only for points successfully tracked forward) ---
    p0_xy_backward = None
    st_backward = np.zeros_like(st_forward) # Initialize status for backward track

    # Select only points that were successfully tracked forward
    good_forward_points = p1_xy[st_forward == 1].reshape(-1, 1, 2) # Points in current frame
    original_indices_of_good_forward = np.where(st_forward == 1)[0] # Original indices

    if len(good_forward_points) > 0:
        p0_xy_backward_dense, st_backward_dense, err_backward = cv2.calcOpticalFlowPyrLK(
            current_frame_gray, prev_frame_gray, good_forward_points, None, **lk_params
        )
        # Place the results back into the sparse structure corresponding to original points
        p0_xy_backward = np.zeros_like(p0_xy)
        p0_xy_backward[original_indices_of_good_forward] = p0_xy_backward_dense
        st_backward[original_indices_of_good_forward] = st_backward_dense

    # --- Calculate Forward-Backward Error and Determine Validity ---
    current_points_rc = []
    validity_mask = []

    for i in range(len(p0_xy)):
        tracked_successfully_forward = (st_forward[i] == 1)
        tracked_successfully_backward = (st_backward[i] == 1) # Check status from the backward pass

        point_valid = False
        current_point_rc = None # Default to None if tracking fails

        if tracked_successfully_forward and tracked_successfully_backward:
            # Calculate FB error: distance between original p0 and backward-tracked p0
            fb_error = np.linalg.norm(p0_xy[i] - p0_xy_backward[i])

            if fb_error < fb_error_threshold:
                point_valid = True
                # Use the forward-tracked point p1 (convert back to row, col)
                current_point_rc = (int(round(p1_xy[i, 0, 1])), int(round(p1_xy[i, 0, 0]))) # (y, x) -> (row, col)
            else:
                 # print(f"Point {i}: FB error {fb_error:.2f} > threshold {fb_error_threshold}") # Debug
                 pass # point_valid remains False
        # else: print(f"Point {i}: Tracking failed (fwd:{st_forward[i]}, bwd:{st_backward[i]})") # Debug

        # If point is not valid, decide what to do. Let's keep the previous position for now.
        if not point_valid:
            current_point_rc = prev_points_rc[i] # Use previous position as fallback
            # Or could set to None: current_point_rc = None

        current_points_rc.append(current_point_rc)
        validity_mask.append(point_valid) # Store actual validity status

    return current_points_rc, validity_mask


# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    # Input
    # image_path = 'complete_HMC_QU/A2C/folds/fold_0/inference_data/ES0001_CH2_1.npy'
    image_path = 'complete_HMC_QU/A4C/folds/fold_0/inference_data/ES0001 _4CH_1.npy'

    # Anchor Finding
    anchor_params = {
        'num_anchors_per_side': 7,
        'search_length': 40,
        'num_samples_along_search': 60,
        'blur_sigma': 1.0
    }

    # Active Contour
    ac_params = {
        'alpha': 0.01,       # Smoothness/Stiffness
        'beta': 5.0,         # Attraction to edges (higher = more contraction/expansion)
        'gamma': 0.001,      # Step size
        'w_line': 0,         # Attraction to intensity (usually 0 for edge-based)
        'w_edge': 1.0,       # Attraction to gradient/edges (usually 1)
        'max_iterations': 1500,
        'convergence': 0.5
    }

    # Polynomial Fitting
    poly_order = 4
    num_colored_segments_def = 3 # Apical, Mid, Basal (plus white apex segment)

    # KLT Tracking
    lk_params = dict(winSize=(21, 21),   # Search window size
                     maxLevel=3,         # Pyramid levels
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    fb_error_threshold = 1.0 # Max forward-backward error in pixels

        # --- Load Image Sequence ---
    try:
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"File not found: {image_path}")

        data = np.load(image_path, allow_pickle=True) # allow_pickle=True might not be necessary if it's purely numerical data, but often harmless

        # --- Adjust loading based on your specific .npy structure ---

        # Check if 'data' is the numpy array itself
        if isinstance(data, np.ndarray) and data.ndim == 3:
            # Check if the dimensions look like (num_frames, H, W)
            # Add checks based on expected H, W if known, e.g., > 50
            if data.shape[1] > 10 and data.shape[2] > 10: # Basic sanity check for H, W
                 print(f"Loaded data shape: {data.shape}. Assuming (num_frames, height, width).")
                 image_sequence_raw = data
            else:
                 # Could potentially be (H, W, C) if only 3 frames, needs more checks
                 # Or maybe (C, H, W)? For now, assume (N, H, W) is most likely for 3 dims.
                 raise ValueError(f"Loaded 3D array shape {data.shape} doesn't look like (num_frames, H, W). Adjust loading logic if needed.")

        # Example 1: If it's a dict like {'X': array(N, H, W)} - Keep this check for flexibility
        elif isinstance(data, np.ndarray) and data.dtype == 'object':
            data_dict = data.item()
            if 'X' in data_dict and isinstance(data_dict['X'], np.ndarray) and data_dict['X'].ndim == 4:
                 print(f"Loaded data from dictionary key 'X', shape: {data_dict['X'].shape}. Assuming (num_frames, height, width).")
                 image_sequence_raw = data_dict['X'].reshape(-1, 224, 224)
            else:
                 raise ValueError("Found dictionary in .npy, but key 'X' is missing, not a 3D numpy array, or has unexpected shape.")
        # Example 2: If it's just a numpy array (e.g., older format load) - This is redundant now with the first check
        # elif isinstance(data, np.ndarray): # This was the previous general case
        #     image_sequence_raw = data # Handled by the first check now

        else:
             raise ValueError(f"Unsupported .npy file structure or data type: {type(data)}")

        if not isinstance(image_sequence_raw, np.ndarray) or image_sequence_raw.ndim != 3:
             raise ValueError("Loaded data is not a 3D numpy array as expected.")

        # --- Frame Processing Loop (Should work correctly now) ---
        image_sequence_processed = []
        image_sequence_float_norm = [] # Keep float version for active contour

        # Iterate through the frames (slices along the first dimension)
        for frame_idx in range(image_sequence_raw.shape[0]):
            frame_data = image_sequence_raw[frame_idx] # This will be shape (H, W)

            # Should directly hit this condition now
            if frame_data.ndim == 2:
                img_gray = frame_data
            # Keep other checks just in case, though unlikely for (N,H,W) input
            elif frame_data.ndim == 3 and frame_data.shape[-1] in [1, 3]:
                img_gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY) if frame_data.shape[-1] == 3 else frame_data[:, :, 0]
            else:
                 raise ValueError(f"Frame {frame_idx} has unexpected shape: {frame_data.shape}")

            # Normalize to [0, 1] float for active contour
            if img_gray.dtype != float: img_gray_float = img_gray.astype(float)
            else: img_gray_float = img_gray.copy()
            min_val, max_val = np.min(img_gray_float), np.max(img_gray_float)
            if max_val > min_val:
                 img_gray_norm = (img_gray_float - min_val) / (max_val - min_val)
            else:
                 img_gray_norm = np.zeros_like(img_gray_float) # Avoid division by zero if flat image
            image_sequence_float_norm.append(img_gray_norm)

            # Convert to uint8 [0, 255] for KLT
            # Use the normalized float image for scaling to uint8 to handle different original ranges consistently
            img_gray_uint8 = (img_gray_norm * 255).astype(np.uint8)
            # # Alternative: Normalize original gray image directly
            # img_gray_uint8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_sequence_processed.append(img_gray_uint8)


        if not image_sequence_processed:
            raise ValueError("No valid frames processed from the input file.")

        print(f"Processed {len(image_sequence_processed)} frames.")

    except FileNotFoundError as fnf:
        print(f"Error: {fnf}")
        exit()
    except Exception as e:
        print(f"Error loading or processing image sequence: {e}")
        import traceback
        traceback.print_exc() # More detailed error for debugging complex loads
        exit()

    # ... rest of the script remains the same ...


    # --- Data Storage for Results ---
    all_tracked_points = [] # List to store [(start_rc, apex_rc, end_rc), ...] for each frame
    all_contours = []       # List to store the final segmented contour points for each frame


    # --- Process Frame 0 (Manual Initialization) ---
    print("\n--- Processing Frame 0 (Manual Initialization) ---")
    current_frame_idx = 0
    img_gray_f0_norm = image_sequence_float_norm[current_frame_idx]
    img_gray_f0_uint8 = image_sequence_processed[current_frame_idx]

    start_pt_rc, apex_pt_rc, end_pt_rc = get_points(img_gray_f0_norm) # Use normalized float for selection display
    if start_pt_rc is None:
        print("Exiting.")
        exit()

    # Store initial points
    current_points_rc = [start_pt_rc, apex_pt_rc, end_pt_rc]
    all_tracked_points.append(current_points_rc)

    # --- Run the pipeline for Frame 0 ---
    print("Generating contour for Frame 0...")
    initial_contour = create_initial_contour_with_anchors(
        img_gray_f0_norm, start_pt_rc, apex_pt_rc, end_pt_rc, **anchor_params
    )

    if initial_contour is None or len(initial_contour) < 3:
         print("Error: Failed to create initial contour for Frame 0. Exiting.")
         # Store None or empty results for this frame if needed
         all_contours.append(None)
         # Depending on desired behavior, you might exit or try to continue tracking points
         exit() # Exit for now

    evolved_contour = evolve_active_contour(img_gray_f0_norm, initial_contour, **ac_params)

    if evolved_contour is None:
        print("Active contour failed for Frame 0. Storing None.")
        all_contours.append(None)
        # Decide if you want to continue tracking points even if contour fails
        # For now, let's continue tracking using the last known good points
    else:
        coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(initial_contour, apex_pt_rc, poly_order=poly_order)

        segments_left = []
        if coeffs_l is not None and points_l_orig is not None:
            segments_left = segment_polynomials(coeffs_l, points_l_orig, num_colored_segments=num_colored_segments_def)
        else: print("Warning: Cannot segment left side Frame 0 - polynomial fit failed.")

        segments_right = []
        if coeffs_r is not None and points_r_orig is not None:
            segments_right = segment_polynomials(coeffs_r, points_r_orig, num_colored_segments=num_colored_segments_def)
        else: print("Warning: Cannot segment right side Frame 0 - polynomial fit failed.")

        # Store the segmented results (e.g., as a dictionary)
        all_contours.append({'left': segments_left, 'right': segments_right, 'evolved_raw': evolved_contour})


    # --- Process Subsequent Frames (Tracking + Pipeline) ---
    for frame_idx in range(1, len(image_sequence_processed)):
        print(f"\n--- Processing Frame {frame_idx} ---")
        prev_frame_uint8 = image_sequence_processed[frame_idx - 1]
        current_frame_uint8 = image_sequence_processed[frame_idx]
        current_frame_float_norm = image_sequence_float_norm[frame_idx]

        # Get points from the *previous* frame
        prev_points_rc = all_tracked_points[-1] # Get the list [start, apex, end] from last frame

        # Check if previous points are valid before tracking
        if None in prev_points_rc:
             print(f"Error: Cannot track points for frame {frame_idx}, previous points are invalid/None.")
             # Append None to results and potentially stop or skip frame
             all_tracked_points.append([None, None, None])
             all_contours.append(None)
             continue # Skip processing this frame


        # --- Track points using KLT ---
        print(f"Tracking points from frame {frame_idx-1} to {frame_idx}...")
        tracked_points_rc, validity = track_points_klt(
            prev_frame_uint8, current_frame_uint8, prev_points_rc, lk_params, fb_error_threshold
        )

        # Update points for the current frame
        start_pt_rc_tracked, apex_pt_rc_tracked, end_pt_rc_tracked = tracked_points_rc
        all_tracked_points.append([start_pt_rc_tracked, apex_pt_rc_tracked, end_pt_rc_tracked])

        print(f"  Tracked Points (R, C): Start={start_pt_rc_tracked}, Apex={apex_pt_rc_tracked}, End={end_pt_rc_tracked}")
        print(f"  Tracking Validity: Start={validity[0]}, Apex={validity[1]}, End={validity[2]}")

        # --- Check if tracking was successful for all points ---
        if not all(validity) or None in tracked_points_rc:
            print(f"Warning: KLT tracking failed for one or more points in frame {frame_idx}. Using previous positions where failed.")
            # The track_points_klt function already handles fallback to previous position if invalid
            # If a point became None (optional implementation), we need to handle it here.
            # For now, assume track_points_klt returns the previous position on failure.
            if None in tracked_points_rc: # Extra check if track_points_klt was modified to return None
                 print(f"Error: Critical point became None after tracking in frame {frame_idx}. Stopping pipeline for this frame.")
                 all_contours.append(None)
                 continue

        # --- Run the pipeline for the current frame using tracked points ---
        print(f"Generating contour for Frame {frame_idx}...")
        initial_contour = create_initial_contour_with_anchors(
            current_frame_float_norm, start_pt_rc_tracked, apex_pt_rc_tracked, end_pt_rc_tracked,
            **anchor_params
        )

        if initial_contour is None or len(initial_contour) < 3:
            print(f"Error: Failed to create initial contour for Frame {frame_idx}. Storing None.")
            all_contours.append(None)
            continue # Skip rest of pipeline for this frame

        evolved_contour = evolve_active_contour(current_frame_float_norm, initial_contour, **ac_params)

        if evolved_contour is None:
            print(f"Active contour failed for Frame {frame_idx}. Storing None.")
            all_contours.append(None)
            continue # Skip rest of pipeline for this frame
        else:
            # Use the *tracked* apex point for splitting the evolved contour
            coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(initial_contour, apex_pt_rc_tracked, poly_order=poly_order)

            segments_left = []
            if coeffs_l is not None and points_l_orig is not None:
                segments_left = segment_polynomials(coeffs_l, points_l_orig, num_colored_segments=num_colored_segments_def)
            else: print(f"Warning: Cannot segment left side Frame {frame_idx} - polynomial fit failed.")

            segments_right = []
            if coeffs_r is not None and points_r_orig is not None:
                segments_right = segment_polynomials(coeffs_r, points_r_orig, num_colored_segments=num_colored_segments_def)
            else: print(f"Warning: Cannot segment right side Frame {frame_idx} - polynomial fit failed.")

            # Store the segmented results
            all_contours.append({'left': segments_left, 'right': segments_right, 'evolved_raw': evolved_contour})


    # --- Visualization (Show results for ALL processed frames sequentially) ---
    print(f"\n--- Displaying Results Sequentially for All Processed Frames ---")

    # Define colors outside the loop (consistent for all frames)
    # colors[0]=Apical, colors[1]=Mid, colors[2]=Basal
    colors_left_functional = ['red', 'lime', 'cyan']      # Apical, Mid, Basal
    colors_right_functional = ['magenta', 'yellow', 'blue'] # Apical, Mid, Basal
    expected_num_segments = num_colored_segments_def + 1 # Includes white apex

    # Store average features per frame if needed later
    all_frames_left_features = []
    all_frames_right_features = []

    for frame_idx in range(len(image_sequence_processed)):
        print(f"\n--- Preparing Visualization for Frame {frame_idx} ---")

        img_to_display = image_sequence_float_norm[frame_idx]
        points_to_display = all_tracked_points[frame_idx]
        contour_data_to_display = all_contours[frame_idx]

        # --- Check if data for this frame is valid ---
        if points_to_display is None or None in points_to_display:
            print(f"Skipping Frame {frame_idx}: Tracked points are missing or invalid.")
            # Append placeholder for features if tracking failed
            all_frames_left_features.append(None)
            all_frames_right_features.append(None)
            continue # Skip to the next frame

        # Retrieve tracked points
        start_pt_rc_disp, apex_pt_rc_disp, end_pt_rc_disp = points_to_display

        # --- Create Figure for this Frame ---
        # Create new figure and axes for each frame to avoid plotting over previous ones
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.suptitle(f"Endocardial Boundary Extraction - Frame {frame_idx}", fontsize=16)

        # --- Panel 1: Input + Tracked Points ---
        ax = axes[0]
        ax.imshow(img_to_display, cmap='gray')
        ax.plot(start_pt_rc_disp[1], start_pt_rc_disp[0], 'go', markersize=7, label='Tracked Start')
        ax.text(start_pt_rc_disp[1], start_pt_rc_disp[0] + 15, 'START', color='lime', ha='center')
        ax.plot(apex_pt_rc_disp[1], apex_pt_rc_disp[0], 'yo', markersize=7, label='Tracked Apex')
        ax.text(apex_pt_rc_disp[1], apex_pt_rc_disp[0] - 15, 'APEX', color='yellow', ha='center', va='bottom')
        ax.plot(end_pt_rc_disp[1], end_pt_rc_disp[0], 'bo', markersize=7, label='Tracked End')
        ax.text(end_pt_rc_disp[1], end_pt_rc_disp[0] + 15, 'END', color='cyan', ha='center')
        ax.set_title("Input + Tracked Key Points")
        ax.axis('off')
        ax.legend()

        # --- Panel 2: Active Polynomials (Segmented) ---
        ax = axes[1]
        ax.imshow(img_to_display, cmap='gray')

        current_frame_left_features = None
        current_frame_right_features = None

        if contour_data_to_display is None:
            print(f"Displaying Frame {frame_idx}: Contour generation failed. Showing only tracked points.")
            ax.set_title("Contour Failed")
            # Optionally add the tracked points again to the second panel for context
            ax.plot(start_pt_rc_disp[1], start_pt_rc_disp[0], 'go', markersize=5)
            ax.plot(apex_pt_rc_disp[1], apex_pt_rc_disp[0], 'yo', markersize=5)
            ax.plot(end_pt_rc_disp[1], end_pt_rc_disp[0], 'bo', markersize=5)
            ax.axis('off') # Keep axis off even if contour failed
        else:
            # Proceed with plotting the segmented contour
            segments_left = contour_data_to_display.get('left', []) # Use .get for safety
            segments_right = contour_data_to_display.get('right', [])

            # Plot Left Segments (Apex -> Basal Order)
            # segments_left[0]=apex(white), [1]=apical, [2]=mid, [3]=basal
            valid_left_segments = []
            if segments_left and len(segments_left) > 0: # Check if list is not empty
                if len(segments_left) == expected_num_segments:
                    for i, seg in enumerate(segments_left):
                        if seg is not None and len(seg) > 1:
                             valid_left_segments.append(seg) # Keep track for feature calculation
                             if i == 0: color = 'white'
                             else:
                                 color_index = i - 1 # Map segment index (1,2,3) to color index (0,1,2)
                                 if color_index < len(colors_left_functional): color = colors_left_functional[color_index]
                                 else: color = 'gray' # Fallback color
                             ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
                else: # Fallback plot if segmentation returned different number
                    print(f"  Warning: Plotting left side with {len(segments_left)} segments (expected {expected_num_segments}). Using fallback colors.")
                    for i, seg in enumerate(segments_left):
                         if seg is not None and len(seg) > 1:
                               valid_left_segments.append(seg)
                               color = colors_left_functional[i % len(colors_left_functional)]
                               ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)

            # Plot Right Segments (Apex -> Basal Order)
            # segments_right[0]=apex(white), [1]=apical, [2]=mid, [3]=basal
            valid_right_segments = []
            if segments_right and len(segments_right) > 0: # Check if list is not empty
                if len(segments_right) == expected_num_segments:
                     for i, seg in enumerate(segments_right):
                         if seg is not None and len(seg) > 1:
                             valid_right_segments.append(seg)
                             if i == 0: color = 'white'
                             else:
                                 color_index = i - 1 # Map segment index (1,2,3) to color index (0,1,2)
                                 if color_index < len(colors_right_functional): color = colors_right_functional[color_index]
                                 else: color = 'gray'
                             ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
                else: # Fallback plot
                    print(f"  Warning: Plotting right side with {len(segments_right)} segments (expected {expected_num_segments}). Using fallback colors.")
                    for i, seg in enumerate(segments_right):
                         if seg is not None and len(seg) > 1:
                               valid_right_segments.append(seg)
                               color = colors_right_functional[i % len(colors_right_functional)]
                               ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)

            ax.set_title("Active Polynomials (Segmented)")
            ax.axis('off')

            # --- Calculate and Print Features for this frame (if segments valid) ---
            # Calculate only if segments were actually plotted and valid
            if valid_left_segments:
                # Ensure we skip the apex segment (index 0) if present and expected
                segments_to_avg_left = valid_left_segments
                if len(valid_left_segments) == expected_num_segments:
                    segments_to_avg_left = valid_left_segments[1:] # Skip white apex segment
                # Check if segments_to_avg_left is not empty after skipping apex
                if segments_to_avg_left:
                    current_frame_left_features = [
                        [sum(col) / len(col) for col in zip(*matrix)] if matrix.size > 0 else [np.nan, np.nan] # Handle empty matrix
                        for matrix in segments_to_avg_left # Use the filtered list
                    ]
                    print(f'  Frame {frame_idx} Left Features (Avg R,C): {current_frame_left_features}')
                else:
                    print(f'  Frame {frame_idx} Left Features: No valid non-apex segments found.')
                    current_frame_left_features = None

            if valid_right_segments:
                segments_to_avg_right = valid_right_segments
                if len(valid_right_segments) == expected_num_segments:
                    segments_to_avg_right = valid_right_segments[1:] # Skip white apex segment
                 # Check if segments_to_avg_right is not empty after skipping apex
                if segments_to_avg_right:
                    # Reverse the order for right side features if needed (Basal -> Mid -> Apical order often desired)
                    segments_to_avg_right = segments_to_avg_right[::-1]
                    current_frame_right_features = [
                        [sum(col) / len(col) for col in zip(*matrix)] if matrix.size > 0 else [np.nan, np.nan] # Handle empty matrix
                        for matrix in segments_to_avg_right
                    ]
                    print(f'  Frame {frame_idx} Right Features (Avg R,C): {current_frame_right_features}')
                else:
                    print(f'  Frame {frame_idx} Right Features: No valid non-apex segments found.')
                    current_frame_right_features = None


        # Store features for the frame
        all_frames_left_features.append(current_frame_left_features)
        all_frames_right_features.append(current_frame_right_features)

        # --- Finalize and Show Plot for Current Frame ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        print(f"--> Displaying plot for Frame {frame_idx}. Close the plot window to view the next frame...")
        plt.show() # Show the plot for the current frame and block until closed

    print("\n--- Finished displaying all frames ---")

    # Optional: Print all collected features at the end
    # print("\nCollected Left Features per Frame:")
    # for i, feat in enumerate(all_frames_left_features):
    #     print(f" Frame {i}: {feat}")
    # print("\nCollected Right Features per Frame:")
    # for i, feat in enumerate(all_frames_right_features):
    #     print(f" Frame {i}: {feat}")