import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
# from skimage.measure import approximate_polygon # Not strictly needed now but keep if desired later
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates, gaussian_filter as gaussian_filter_ndimage # Use alias to avoid name clash
import os # For file path manipulation
import argparse # For command-line arguments

# Default directory for saving annotations
ANNOTATION_DIR = "annotations"

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
        # --- Boundary Check ---
        if not (0 <= anchor_pt_xy[0] < width and 0 <= anchor_pt_xy[1] < height):
            print(f"Warning: Left anchor {i} ({anchor_pt_xy}) is out of bounds. Skipping.")
            continue
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
        # --- Boundary Check ---
        if not (0 <= anchor_pt_xy[0] < width and 0 <= anchor_pt_xy[1] < height):
            print(f"Warning: Right anchor {i} ({anchor_pt_xy}) is out of bounds. Skipping.")
            continue
        found_anchors_xy.append(anchor_pt_xy) # Add to the main list

    # Check *after* collecting valid points
    if len(found_anchors_xy) != 2 * num_anchors_per_side:
        print(f"Warning: Expected {2*num_anchors_per_side} anchors (within bounds), found {len(found_anchors_xy)}. May indicate issues.")
        if len(found_anchors_xy) == 0:
             return None

    return np.array(found_anchors_xy) # Return as (x, y)
# --- END: Copied from Script 1 ---


# --- Global variables for point selection ---
points = []
fig_select = None
ax_select = None
img_select = None
cid_click = None # Connection ID for click events
cid_key = None   # Connection ID for key press events
points_confirmed = False # Flag to track if 's' was pressed

# --- Helper Function for Manual Point Selection ---
def onclick(event):
    """Handles mouse clicks for point selection."""
    global points, ax_select, fig_select, points_confirmed
    if event.inaxes != ax_select: return # Ignore clicks outside axes
    if len(points) >= 3: return # Don't add more than 3 points
    points_confirmed = False # Reset confirmation if clicking new points after reset

    # Store as (x, y) internally during selection for consistency with plot events
    px, py = int(event.xdata), int(event.ydata)
    print(f'Point {len(points)+1} selected: x={px}, y={py}')
    points.append((px, py)) # Store as (x, y) -> (col, row) in the list
    ax_select.plot(px, py, 'r+', markersize=10)
    update_title() # Update title with progress
    fig_select.canvas.draw()

    if len(points) == 3:
        print("START, APEX, END points selected. Press 'r' to reset, or CLOSE the window to proceed.")
        update_title()

def onkey(event):
    """Handles key presses during point selection (only reset needed now)."""
    global points, ax_select, fig_select, img_select, points_confirmed
    if event.key == 'r' or event.key == 'R':
        print("Resetting point selection.")
        points = [] # Clear points
        points_confirmed = False # Reset confirmation
        # Clear plot and redraw
        ax_select.cla()
        ax_select.imshow(img_select, cmap='gray')
        update_title()
        fig_select.canvas.draw()
    # REMOVED 's' key logic here - confirmation happens implicitly by closing window

def update_title():
    """Updates the plot title during point selection."""
    global ax_select, points
    num_needed = 3 - len(points)
    if len(points) == 3:
        ax_select.set_title("Got 3 points. Press 'r' to reset, or CLOSE window to use.", color='orange')
    elif num_needed > 0:
        point_names = ["START", "APEX", "END"]
        next_point = point_names[len(points)]
        ax_select.set_title(f"Click {next_point} ({num_needed} remaining). Press 'r' to reset.", color='white')
    else: # Start state
        ax_select.set_title("Click START (3 remaining). Press 'r' to reset.", color='white')
    if fig_select and fig_select.canvas: # Check if figure exists
        fig_select.canvas.draw_idle()

def get_points_interactive(image):
    """
    Displays the image and waits for 3 clicks (START, APEX, END).
    Allows resetting with 'r' key. Points are confirmed by closing the window.

    Args:
        image (np.ndarray): Grayscale image to display.

    Returns:
        tuple: (start_pt_rc, apex_pt_rc, end_pt_rc) as (row, col), or (None, None, None) if unsuccessful.
    """
    global points, fig_select, ax_select, img_select, cid_click, cid_key, points_confirmed
    points = [] # Reset points
    points_confirmed = False # Reset flag
    img_select = image

    print("\n--- Interactive Point Selection ---")
    print("Please click 3 points on the image in this order: START, APEX, END")
    print("Press 'r' anytime to RESET the selection.")
    print("After 3 points are selected, CLOSE the window to proceed with these points.")
    print("Closing the window before 3 points are selected will cancel.")

    # Ensure matplotlib backend allows interactive plotting
    # print(f"Using Matplotlib backend: {plt.get_backend()}") # Optional debug
    # plt.switch_backend('TkAgg') # Or 'Qt5Agg' if preferred and installed

    fig_select, ax_select = plt.subplots(figsize=(8, 8))
    ax_select.imshow(image, cmap='gray')
    update_title() # Initial title

    # Connect events
    cid_click = fig_select.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig_select.canvas.mpl_connect('key_press_event', onkey)

    plt.show(block=True) # Wait until the window is closed

    # Disconnect events after window is closed
    # Check if fig_select still exists and has a canvas
    if fig_select and hasattr(fig_select, 'canvas') and fig_select.canvas:
        if cid_click: fig_select.canvas.mpl_disconnect(cid_click)
        if cid_key: fig_select.canvas.mpl_disconnect(cid_key)
    cid_click = cid_key = None # Reset connection IDs
    ax_select = None # Prevent potential issues if function is called again
    fig_select = None


    if len(points) == 3:
        print("Point selection window closed with 3 points.")
        # Convert selected (x, y) points to (row, col) for internal use
        start_pt_rc = (points[0][1], points[0][0]) # (y, x) -> (row, col)
        apex_pt_rc = (points[1][1], points[1][0])  # (y, x) -> (row, col)
        end_pt_rc = (points[2][1], points[2][0])    # (y, x) -> (row, col)
        print("Selected points (row, col):", start_pt_rc, apex_pt_rc, end_pt_rc)
        # Saving decision will be made *after* the final plot
        return start_pt_rc, apex_pt_rc, end_pt_rc
    else:
        print("Error: Window closed before 3 points were selected.")
        return None, None, None

# --- NEW Function to create Initial Contour using Anchors ---
# [ This function remains unchanged ]
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
    start_pt_xy = (start_pt_rc[1], start_pt_rc[0])
    apex_pt_xy = (apex_pt_rc[1], apex_pt_rc[0])
    end_pt_xy = (end_pt_rc[1], end_pt_rc[0])
    anchor_points_xy = find_max_intensity_anchor_points(
        image, start_pt_xy, end_pt_xy, apex_pt_xy,
        num_anchors_per_side=num_anchors_per_side,
        search_length=search_length,
        num_samples_along_search=num_samples_along_search,
        blur_sigma=blur_sigma
    )
    num_expected_anchors = 2 * num_anchors_per_side
    if anchor_points_xy is not None and len(anchor_points_xy) > 0:
         print(f"Found {len(anchor_points_xy)} valid anchor points (expected {num_expected_anchors}).")
         if len(anchor_points_xy) != num_expected_anchors:
              print(f"Warning: Found {len(anchor_points_xy)} anchors, expected {num_expected_anchors}. Falling back to triangle initial contour for robustness.")
              return create_initial_triangle(start_pt_rc, apex_pt_rc, end_pt_rc, num_points_per_side=20)
         anchor_points_rc = anchor_points_xy[:, ::-1]
         left_anchors_rc = anchor_points_rc[:num_anchors_per_side]
         right_anchors_rc = anchor_points_rc[num_anchors_per_side:]
    elif anchor_points_xy is None or len(anchor_points_xy) == 0:
        print("Warning: Anchor point finding failed or returned no points. Falling back to triangle.")
        return create_initial_triangle(start_pt_rc, apex_pt_rc, end_pt_rc, num_points_per_side=20)
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
    print(f"Initial contour created with {len(initial_contour)} points (including start, apex, end, and anchors).")
    return initial_contour

# --- Function to create Initial Triangle Contour (as fallback) ---
# [ This function remains unchanged ]
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

# --- Active Contour Evolution (Simplified Constraint) ---
# [ This function remains unchanged ]
def evolve_active_contour(image, initial_contour,
                          alpha=0.015, beta=10, gamma=0.001, # Parameters to TUNE
                          w_line=0, w_edge=1, # Added weights here for clarity
                          max_iterations=2500, convergence=0.1):
    """Evolves the active contour starting from the initial shape."""
    if image.dtype != float:
        img_float = image.astype(float)
        img_max = np.max(img_float)
        if img_max > 1.0 and img_max > 0: img_float = img_float / img_max
    else:
         img_float = image
         img_max = np.max(img_float)
         if img_max > 1.0:
             print("Warning: Input float image max value > 1.0. Normalizing.")
             img_float = img_float / img_max
    img_smooth = gaussian(img_float, sigma=1, preserve_range=False)
    try:
        snake = active_contour(img_smooth,
                               initial_contour,
                               alpha=alpha, beta=beta, gamma=gamma,
                               w_line=w_line, w_edge=w_edge,
                               max_num_iter=max_iterations, convergence=convergence)
        return snake
    except Exception as e:
        print(f"Error during active contour evolution: {e}")
        return None

# --- Fit Polynomials ---
# [ This function remains unchanged ]
def fit_polynomials(contour_points, apex_pt_orig_rc, poly_order=4):
    """
    Splits contour near the original apex and fits polynomials.
    Fits X = P(Y) i.e., Col = P(Row). Expects (row, col) points.
    """
    if contour_points is None or len(contour_points) < 10:
        print("Error: Not enough contour points for fitting.")
        return None, None, None, None
    distances = np.sqrt(np.sum((contour_points - np.array(apex_pt_orig_rc))**2, axis=1))
    apex_index = np.argmin(distances)
    if apex_index == 0 and len(contour_points) > 1 : apex_index = 1
    if apex_index == len(contour_points) - 1 and len(contour_points) > 1: apex_index = len(contour_points) - 2
    if not (0 < apex_index < len(contour_points) - 1):
         print(f"Warning: Could not find suitable apex split point (index {apex_index} invalid for length {len(contour_points)}). Signalling failure.")
         return None, None, None, None
    points_left = contour_points[0:apex_index+1]
    points_right = contour_points[apex_index:]
    min_points_needed = poly_order + 1
    if len(points_left) < min_points_needed or len(points_right) < min_points_needed:
        print(f"Error: Not enough points in left ({len(points_left)}) or right ({len(points_right)}) segments for polyfit (order {poly_order}). Need {min_points_needed}.")
        return None, None, points_left, points_right
    coeffs_left = None
    coeffs_right = None
    try:
        if np.ptp(points_left[:, 0]) < 1e-6:
            print("Warning: Left segment is nearly vertical. Polynomial fit Col=P(Row) may be unstable.")
        else:
             coeffs_left = np.polyfit(points_left[:, 0], points_left[:, 1], poly_order)
        if np.ptp(points_right[:, 0]) < 1e-6:
            print("Warning: Right segment is nearly vertical. Polynomial fit Col=P(Row) may be unstable.")
        else:
            coeffs_right = np.polyfit(points_right[:, 0], points_right[:, 1], poly_order)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error during polynomial fitting: {e}.")
        return None, None, points_left, points_right
    return coeffs_left, coeffs_right, points_left, points_right

# --- Segment the Fitted Polynomials ---
# [ This function remains unchanged ]
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
              contain num_colored_segments + 1 arrays. The first array is the apex segment. Returns empty list on error.
    """
    if coeffs is None or side_points_rc is None or len(side_points_rc) < 2:
        # print("Warning: Cannot segment polynomials - invalid input.") # Reduced verbosity
        return []
    num_total_segments = num_colored_segments + 1
    min_row = np.min(side_points_rc[:, 0])
    max_row = np.max(side_points_rc[:, 0])
    if max_row <= min_row or np.isclose(max_row, min_row):
        # print("Warning: Cannot generate smooth curve for segmentation - row range too small or zero.") # Reduced verbosity
        return []
    smooth_rows = np.linspace(min_row, max_row, 200)
    smooth_cols = np.polyval(coeffs, smooth_rows)
    smooth_curve_points = np.vstack((smooth_rows, smooth_cols)).T
    apex_end_point_guess = side_points_rc[np.argmin(side_points_rc[:,0])]
    dist_start_to_apex_guess = np.linalg.norm(smooth_curve_points[0] - apex_end_point_guess)
    dist_end_to_apex_guess = np.linalg.norm(smooth_curve_points[-1] - apex_end_point_guess)
    if dist_start_to_apex_guess > dist_end_to_apex_guess:
         smooth_curve_points = smooth_curve_points[::-1]
    total_smooth_points = len(smooth_curve_points)
    segments = []
    if total_smooth_points == 0:
         return [np.array([]).reshape(0,2)] * num_total_segments
    if total_smooth_points < num_total_segments:
        # print(f"Warning: Not enough points ({total_smooth_points}) generated to create {num_total_segments} segments. Returning fewer segments.") # Reduced verbosity
        indices = np.linspace(0, total_smooth_points, total_smooth_points + 1, dtype=int)
        for i in range(total_smooth_points):
            seg = smooth_curve_points[indices[i]:indices[i+1]]
            if len(seg) > 0: segments.append(seg)
        while len(segments) < num_total_segments: segments.append(np.array([]).reshape(0,2))
        return segments
    else:
         points_per_segment = total_smooth_points // num_total_segments
         remainder = total_smooth_points % num_total_segments
         start_idx = 0
         for i in range(num_total_segments):
             current_segment_length = points_per_segment + (1 if i < remainder else 0)
             end_idx = start_idx + current_segment_length
             end_idx = min(end_idx, total_smooth_points)
             segment_points = smooth_curve_points[start_idx:end_idx]
             segments.append(segment_points if len(segment_points) > 0 else np.array([]).reshape(0,2))
             start_idx = end_idx
             if start_idx >= total_smooth_points and i < num_total_segments - 1:
                 # print(f"Warning: Ran out of points unexpectedly during segmentation at segment {i+1}/{num_total_segments}") # Reduced verbosity
                 while len(segments) < num_total_segments: segments.append(np.array([]).reshape(0,2))
                 break
    while len(segments) < num_total_segments: segments.append(np.array([]).reshape(0,2))
    if len(segments) > num_total_segments: segments = segments[:num_total_segments]
    return segments


# --- Helper function to generate annotation file path ---
def get_annotation_path(image_file_path, base_annotation_dir=ANNOTATION_DIR):
    """
    Generates the corresponding .npy path for annotations inside a base directory.
    """
    image_filename = os.path.basename(image_file_path)
    base, _ = os.path.splitext(image_filename)
    annotation_filename = base + "_points.npy"
    # Ensure the base annotation directory exists
    os.makedirs(base_annotation_dir, exist_ok=True)
    return os.path.join(base_annotation_dir, annotation_filename)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Extract and segment LV endocardial boundary.")
    # Make image_path optional, if not provided, use default.
    parser.add_argument("--image_path", default='complete_HMC_QU/A2C/folds/fold_0/inference_data/ES0001_CH2_1.npy',
                        help="Path to the input image (.npy file)")
    # REMOVED --annotate flag
    parser.add_argument("--poly_order", type=int, default=4, help="Order of polynomial for fitting.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Active contour membrane stiffness.")
    parser.add_argument("--beta", type=float, default=5.0, help="Active contour balloon force.")
    parser.add_argument("--gamma", type=float, default=0.001, help="Active contour step size.")
    parser.add_argument("--w_edge", type=float, default=1.0, help="Active contour edge weight.")
    parser.add_argument("--w_line", type=float, default=0.0, help="Active contour line weight.")
    parser.add_argument("--max_iter", type=int, default=1500, help="Active contour max iterations.")
    parser.add_argument("--convergence", type=float, default=0.5, help="Active contour convergence threshold.")
    parser.add_argument("--force_reannotate", action="store_true",
                        help="Force manual annotation even if an annotation file exists.")

    args = parser.parse_args()

    image_path = args.image_path
    pts_path = get_annotation_path(image_path) # Path will now be inside ANNOTATION_DIR

    # --- Load Image ---
    try:
        # Simplified loading assuming the structure used before
        data = np.load(image_path, allow_pickle=True)
        image_data = data.item()['X'].reshape(-1, 224, 224)[0] # Adjust shape as needed

        if image_data.ndim == 3: # H, W, C? Assume grayscale needed
            img_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if image_data.shape[-1] == 3 else image_data[:,:,0]
        elif image_data.ndim == 2:
            img_gray = image_data
        else:
             raise ValueError(f"Unsupported image data shape after reshape: {image_data.shape}")

        # Normalize image to float [0, 1]
        if img_gray.dtype == np.uint8:
             img_gray_norm = img_gray.astype(float) / 255.0
        else:
             min_val, max_val = np.min(img_gray), np.max(img_gray)
             if max_val > min_val:
                 img_gray_norm = (img_gray.astype(float) - min_val) / (max_val - min_val)
             else:
                 img_gray_norm = np.zeros_like(img_gray, dtype=float)

    except FileNotFoundError:
         print(f"Error: Image file not found at {image_path}")
         exit()
    except Exception as e:
        print(f"Error loading or processing image '{image_path}': {e}")
        exit()


    # --- 1. Get Key Points (Load or Force Annotate/Re-annotate) ---
    start_pt_rc, apex_pt_rc, end_pt_rc = None, None, None
    annotation_loaded = False

    # Check if annotation exists and force_reannotate is OFF
    if os.path.exists(pts_path) and not args.force_reannotate:
        try:
            loaded_pts = np.load(pts_path, allow_pickle=True).item()
            start_pt_rc = tuple(loaded_pts['start_rc'])
            apex_pt_rc = tuple(loaded_pts['apex_rc'])
            end_pt_rc = tuple(loaded_pts['end_rc'])
            print(f"Loaded points from: {pts_path}")
            print(f"  Start: {start_pt_rc}, Apex: {apex_pt_rc}, End: {end_pt_rc}")
            annotation_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load points file '{pts_path}': {e}")
            print("Manual point selection required.")
    else:
        if args.force_reannotate:
             print("Forcing re-annotation...")
        else: # File doesn't exist
            print(f"Annotation file not found: {pts_path}")
        print("Manual point selection required.")

    # If points couldn't be loaded or re-annotation is forced, get them interactively
    if not annotation_loaded:
        start_pt_rc, apex_pt_rc, end_pt_rc = get_points_interactive(img_gray_norm)

    # Exit if points were not obtained
    if start_pt_rc is None or apex_pt_rc is None or end_pt_rc is None:
        print("Error: Could not obtain START, APEX, END points. Exiting.")
        exit()

    # --- Now proceed with the rest of the pipeline ---

    # --- Define Parameters for Anchor Finding ---
    anchor_params = {
        'num_anchors_per_side': 7,
        'search_length': 40,
        'num_samples_along_search': 60,
        'blur_sigma': 1.0
    }

    # --- 2. Create Initial Contour using Anchors ---
    initial_contour = create_initial_contour_with_anchors(
        img_gray_norm, start_pt_rc, apex_pt_rc, end_pt_rc,
        **anchor_params
    )

    if initial_contour is None or len(initial_contour) < 3:
         print("Error: Failed to create a valid initial contour. Plotting key points and exiting.")
         # Show just the key points if contour fails
         fig, ax = plt.subplots(figsize=(6, 6))
         ax.imshow(img_gray_norm, cmap='gray')
         ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=6, label='Start')
         ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=6, label='Apex')
         ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=6, label='End')
         ax.set_title("Initial Contour Creation Failed")
         ax.legend()
         plt.show()
         exit()

    # --- 3. Evolve Active Contour ---
    print("Evolving Active Contour...")
    evolved_contour = evolve_active_contour(img_gray_norm, initial_contour,
                                            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                            w_line=args.w_line, w_edge=args.w_edge,
                                            max_iterations=args.max_iter, convergence=args.convergence)

    if evolved_contour is None:
        print("Active contour evolution failed. Plotting initial contour and exiting.")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_gray_norm, cmap='gray')
        ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=5, label='Start')
        ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=5, label='Apex')
        ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=5, label='End')
        ax.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-', lw=1.5, label='Initial Contour')
        ax.set_title("Active Contour Failed - Showing Initial Contour")
        ax.legend()
        plt.show()
        exit()

    # --- 4. Fit Polynomials ---
    print(f"Fitting polynomials of order {args.poly_order}...")
    coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(initial_contour, apex_pt_rc, poly_order=args.poly_order)

    # Check if fitting failed significantly
    if coeffs_l is None and coeffs_r is None:
        print("Error: Polynomial fitting failed for both sides. Cannot segment.")
        # Show evolved contour and key points before exiting
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_gray_norm, cmap='gray')
        ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=5, label='Start')
        ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=5, label='Apex')
        ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=5, label='End')
        ax.plot(evolved_contour[:, 1], evolved_contour[:, 0], 'm-', lw=1.5, label='Evolved Contour (Fit Failed)')
        ax.set_title("Polynomial Fitting Failed")
        ax.legend()
        plt.show()
        exit()


    # --- 5. Segment Polynomials ---
    print("Segmenting polynomial fits...")
    segments_left = []
    segments_right = []
    num_colored_segments_def = 3 # Apical, Mid, Basal

    if coeffs_l is not None and points_l_orig is not None:
        segments_left = segment_polynomials(coeffs_l, points_l_orig, num_colored_segments=num_colored_segments_def)
    else:
         print("Warning: Cannot segment left side - polynomial fit failed or missing.")

    if coeffs_r is not None and points_r_orig is not None:
        segments_right = segment_polynomials(coeffs_r, points_r_orig, num_colored_segments=num_colored_segments_def)
    else:
         print("Warning: Cannot segment right side - polynomial fit failed or missing.")


    # --- 6. Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig_title = f"Endocardial Boundary Stages: {os.path.basename(image_path)}"
    # No longer adding (Annotation Mode) here
    plt.suptitle(fig_title, fontsize=16)

    # Panel 1: Input Image + Key Points
    ax = axes[0]
    ax.imshow(img_gray_norm, cmap='gray')
    ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=6, label='Start')
    ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=6, label='Apex')
    ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=6, label='End')
    ax.set_title("Input + Key Points")
    ax.legend()
    ax.axis('off')

    # Panel 2: Initial Contour
    ax = axes[1]
    ax.imshow(img_gray_norm, cmap='gray')
    ax.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-', lw=1.5, label='Initial Contour')
    ax.plot(start_pt_rc[1], start_pt_rc[0], 'go', markersize=4)
    ax.plot(apex_pt_rc[1], apex_pt_rc[0], 'yo', markersize=4)
    ax.plot(end_pt_rc[1], end_pt_rc[0], 'bo', markersize=4)
    ax.set_title("Initial Contour")
    ax.legend()
    ax.axis('off')

    # Panel 3: Active Polynomials (Segmented)
    ax = axes[2]
    ax.imshow(img_gray_norm, cmap='gray')
    colors_left = ['white', 'lime', 'yellow', 'cyan']     # Apex, Apical, Mid, Basal
    colors_right = ['white', 'red', 'magenta', 'blue']  # Apex, Apical, Mid, Basal
    segment_names = ["Apex", "Apical", "Mid", "Basal"]
    expected_num_segments = num_colored_segments_def + 1
    plotted_legend_labels = set()

    # Plot Left Segments
    valid_left_segments = [s for s in segments_left if s is not None and len(s) > 0]
    if len(valid_left_segments) == expected_num_segments:
        # print("Plotting left segments:") # Debug
        for i, seg in enumerate(segments_left): # Iterate original list to get correct index/color
            if seg is not None and len(seg) > 0:
                color = colors_left[i % len(colors_left)]
                label = f"L-{segment_names[i]}" if segment_names[i] not in plotted_legend_labels else ""
                if label: plotted_legend_labels.add(segment_names[i])
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3, label=label)
    elif len(valid_left_segments) > 0:
         print(f"Warning: Plotting left side with {len(valid_left_segments)} valid segments instead of {expected_num_segments}.")
         for i, seg in enumerate(valid_left_segments):
             color = colors_left[i % len(colors_left)]
             ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)

    # Plot Right Segments
    valid_right_segments = [s for s in segments_right if s is not None and len(s) > 0]
    if len(valid_right_segments) == expected_num_segments:
        # print("Plotting right segments:") # Debug
        for i, seg in enumerate(segments_right): # Iterate original list to get correct index/color
             if seg is not None and len(seg) > 0:
                color = colors_right[i % len(colors_right)]
                label = f"R-{segment_names[i]}" if segment_names[i] not in plotted_legend_labels else ""
                if label: plotted_legend_labels.add(segment_names[i])
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3, label=label)
    elif len(valid_right_segments) > 0:
         print(f"Warning: Plotting right side with {len(valid_right_segments)} valid segments instead of {expected_num_segments}.")
         for i, seg in enumerate(valid_right_segments):
              color = colors_right[i % len(colors_right)]
              ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)

    ax.set_title("Active Polynomials (Segmented)")
    ax.legend()
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # Keep the plot window open until the user closes it or responds to the prompt
    plt.show(block=False) # Show plot but don't block execution

    # --- 7. Ask to Save Points (After Visualization) ---
    # Only ask if the points were manually annotated in this run (i.e., not loaded)
    # OR if force_reannotate was used.
    save_prompt = False
    if not annotation_loaded or args.force_reannotate:
        save_prompt = True
        # If forcing re-annotation, check if file exists to ask about overwriting
        if args.force_reannotate and os.path.exists(pts_path):
            prompt_text = f"Annotation file '{pts_path}' exists.\nOverwrite with the currently selected points? [y/N]: "
        else: # Annotation was done because file didn't exist initially
            prompt_text = f"Save the selected START/APEX/END points to '{pts_path}'? [y/N]: "

    if save_prompt:
        try:
            # Make sure plot is drawn before input blocks
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.1) # Short pause for plot update

            user_input = input(f"\n{prompt_text}").strip().lower()

            if user_input == 'y':
                try:
                    # Ensure ANNOTATION_DIR exists (get_annotation_path does this, but double-check)
                    os.makedirs(ANNOTATION_DIR, exist_ok=True)
                    annotation_data = {
                        'start_rc': start_pt_rc,
                        'apex_rc': apex_pt_rc,
                        'end_rc': end_pt_rc
                    }
                    np.save(pts_path, annotation_data)
                    print(f"Annotation saved successfully to: {pts_path}")
                except Exception as e:
                    print(f"Error: Could not save annotation file '{pts_path}': {e}")
            else:
                print("Annotation not saved.")
        except Exception as e:
             print(f"Error during input prompt: {e}. Annotation not saved.")

    else: # Annotation was loaded and not forced to re-annotate
        print("\nAnnotation points were loaded from file. No save prompt needed.")
        # Keep plot window open briefly or until closed manually
        # input("Press Enter to close the plot and exit...") # Optionally wait

    print("\nProcessing finished.")
    # Explicitly close the plot window now if it's still open
    plt.close('all')