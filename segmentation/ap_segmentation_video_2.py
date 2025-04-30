import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
# from skimage.measure import approximate_polygon # Not currently used
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob # Keep for potential future use, but not needed for single .npy

# --- Global variable for point selection ---
points = []
fig_select = None
ax_select = None
img_select = None

# --- Helper Function for Manual Point Selection (Unchanged) ---
def onclick(event):
    global points, ax_select
    if event.inaxes == ax_select:
        print(f'Point selected: x={event.xdata:.0f}, y={event.ydata:.0f}')
        points.append((int(event.xdata), int(event.ydata))) # Store as (x, y) -> (col, row)
        ax_select.plot(event.xdata, event.ydata, 'r+', markersize=10)
        fig_select.canvas.draw()
        if len(points) == 3:
            print("START, APEX, END points selected. Close the selection window.")
            plt.close(fig_select)

# --- Get Manual Points (Unchanged, returns points as (row, col)) ---
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
    plt.show(block=True) # Wait until the window is closed

    if len(points) == 3:
        start_pt = (points[0][1], points[0][0])
        apex_pt = (points[1][1], points[1][0])
        end_pt = (points[2][1], points[2][0])
        return start_pt, apex_pt, end_pt
    else:
        print("Error: Did not select 3 points.")
        return None, None, None

# --- Function to create Initial Triangle Contour (Unchanged) ---
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

# --- Active Contour Evolution (Unchanged) ---
def evolve_active_contour(image, initial_contour,
                          alpha=0.015, beta=10, gamma=0.001, # Parameters to TUNE
                          max_iterations=2500, convergence=0.1):
    """Evolves the active contour starting from the initial shape."""
    if image.dtype != np.float64 and image.dtype != np.float32:
         img_float = image.astype(np.float64) / np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else image.astype(np.float64)
    else:
         img_float = image.astype(np.float64)
    img_smooth = gaussian(img_float, sigma=1, preserve_range=False)

    try:
        snake = active_contour(img_smooth,
                               initial_contour,
                               alpha=alpha, beta=beta, gamma=gamma,
                               max_num_iter=max_iterations, convergence=convergence)
        return snake
    except Exception as e:
        print(f"Error during active contour evolution: {e}")
        # Check for out-of-bounds initial contour
        if initial_contour[:, 0].min() < 0 or initial_contour[:, 0].max() >= image.shape[0] or \
           initial_contour[:, 1].min() < 0 or initial_contour[:, 1].max() >= image.shape[1]:
            print("Error Detail: Initial contour appears to be outside image bounds.")
        return None

# --- Fit Polynomials (Unchanged) ---
def fit_polynomials(contour_points, apex_pt_orig, poly_order=4):
    """
    Splits contour near the original apex and fits polynomials.
    Fits X = P(Y) i.e., Col = P(Row).
    """
    if contour_points is None or len(contour_points) < 10:
        # print("Debug: Not enough contour points for fitting.") # Less verbose
        return None, None, None, None

    apex_pt_orig_arr = np.array(apex_pt_orig)
    distances = np.sqrt(np.sum((contour_points - apex_pt_orig_arr)**2, axis=1))
    apex_index = np.argmin(distances)

    if apex_index == 0 and len(contour_points) > 1: apex_index = 1
    if apex_index == len(contour_points) - 1 and len(contour_points) > 1: apex_index = len(contour_points) - 2
    if len(contour_points) <= 2: return None, None, contour_points, None

    points_left = contour_points[0:apex_index+1]
    points_right = contour_points[apex_index:]

    min_pts_for_fit = poly_order + 1
    valid_left = len(points_left) >= min_pts_for_fit and (np.ptp(points_left[:, 0]) > 1e-6)
    valid_right = len(points_right) >= min_pts_for_fit and (np.ptp(points_right[:, 0]) > 1e-6)

    coeffs_left = None
    coeffs_right = None

    try:
        if valid_left:
            coeffs_left = np.polyfit(points_left[:, 0], points_left[:, 1], poly_order)
        else:
            # print(f"Debug: Insufficient points/span for left fit ({len(points_left)} pts)") # Less verbose
            points_left = None # Indicate failure

        if valid_right:
             coeffs_right = np.polyfit(points_right[:, 0], points_right[:, 1], poly_order)
        else:
             # print(f"Debug: Insufficient points/span for right fit ({len(points_right)} pts)") # Less verbose
             points_right = None # Indicate failure

    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error during polynomial fitting: {e}.")
        if not valid_left: coeffs_left = None
        if not valid_right: coeffs_right = None

    return coeffs_left, coeffs_right, points_left, points_right

# --- Segment Polynomials (Unchanged, uses apex_pt_orig_side for ordering) ---
def segment_polynomials(coeffs, side_points, apex_pt_orig_side, num_colored_segments=3):
    """
    Generates points along the fitted polynomial and divides them into segments.
    Returns num_colored_segments + 1 segments, ordered from APEX outwards.
    """
    if coeffs is None or side_points is None or len(side_points) < 2:
        return []

    num_total_segments = num_colored_segments + 1
    min_row = np.min(side_points[:, 0])
    max_row = np.max(side_points[:, 0])

    if max_row <= min_row:
        if len(side_points) > 0:
             apex_like_point = side_points[np.argmin(np.sum((side_points - apex_pt_orig_side)**2, axis=1))]
             return [np.array([apex_like_point])] * num_total_segments # Fallback
        else: return []

    smooth_rows = np.linspace(min_row, max_row, 200)
    smooth_cols = np.polyval(coeffs, smooth_rows)
    smooth_curve_points = np.vstack((smooth_rows, smooth_cols)).T

    dist_from_apex = np.sqrt(np.sum((smooth_curve_points - np.array(apex_pt_orig_side))**2, axis=1))
    if len(dist_from_apex) > 1 and dist_from_apex[0] > dist_from_apex[-1]:
         smooth_curve_points = smooth_curve_points[::-1]

    total_smooth_points = len(smooth_curve_points)
    if total_smooth_points == 0: return []

    num_valid_segments = num_total_segments
    if total_smooth_points < num_total_segments:
        # print(f"Debug: Not enough points ({total_smooth_points}) for {num_total_segments} segments.") # Less verbose
        points_per_segment = 1
        num_valid_segments = total_smooth_points
    else:
        points_per_segment = total_smooth_points // num_total_segments

    segments = []
    start_idx = 0
    for i in range(num_valid_segments):
        if i == num_total_segments - 1: # Last intended segment gets remainder
            end_idx = total_smooth_points
        else:
            end_idx = start_idx + points_per_segment

        if start_idx >= total_smooth_points: break
        end_idx = min(end_idx, total_smooth_points)

        segment_points = smooth_curve_points[start_idx:end_idx]
        if len(segment_points) > 0: segments.append(segment_points)
        start_idx = end_idx

    return segments

# --- KLT Tracking Function (Unchanged) ---
def track_points_klt(prev_img, current_img, prev_pts_tuples,
                     win_size=(21, 21), max_level=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     bidirectional_error_threshold=1.0):
    """Tracks points using KLT with bidirectional error check."""
    if prev_img is None or current_img is None or not prev_pts_tuples: return None, False

    # Ensure uint8 grayscale
    if prev_img.dtype != np.uint8: prev_img = cv2.normalize(prev_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if current_img.dtype != np.uint8: current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(prev_img.shape) > 2: prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    if len(current_img.shape) > 2: current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    prev_pts_klt = np.array([[pt[1], pt[0]] for pt in prev_pts_tuples], dtype=np.float32).reshape(-1, 1, 2)

    try: # Forward tracking
        next_pts_klt, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(prev_img, current_img, prev_pts_klt, None, winSize=win_size, maxLevel=max_level, criteria=criteria)
    except cv2.error as e: print(f"OpenCV Error (Fwd KLT): {e}"); return None, False
    except Exception as e: print(f"Error (Fwd KLT): {e}"); return None, False
    if next_pts_klt is None or status_fwd is None: print("KLT forward failed (None)."); return None, False

    good_fwd = status_fwd.flatten() == 1
    if not np.all(good_fwd): print(f"KLT forward failed for points idx: {np.where(~good_fwd)[0]}"); return None, False

    prev_pts_klt_good = prev_pts_klt[good_fwd]
    next_pts_klt_good = next_pts_klt[good_fwd]

    try: # Backward tracking
        prev_pts_klt_bwd, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(current_img, prev_img, next_pts_klt_good, None, winSize=win_size, maxLevel=max_level, criteria=criteria)
    except cv2.error as e: print(f"OpenCV Error (Bwd KLT): {e}"); return None, False
    except Exception as e: print(f"Error (Bwd KLT): {e}"); return None, False
    if prev_pts_klt_bwd is None or status_bwd is None: print("KLT backward failed (None)."); return None, False

    good_bwd = status_bwd.flatten() == 1
    if not np.all(good_bwd): print(f"KLT backward failed for points idx: {np.where(~good_bwd)[0]}"); return None, False

    # Bidirectional Error
    original_points_for_comparison = prev_pts_klt_good[good_bwd]
    backward_tracked_points = prev_pts_klt_bwd[good_bwd]
    bidirectional_error = np.sqrt(np.sum((original_points_for_comparison - backward_tracked_points)**2, axis=2)).flatten()

    valid_tracking = bidirectional_error < bidirectional_error_threshold
    if not np.all(valid_tracking):
        print(f"KLT bidirectional error too high for points idx: {np.where(~valid_tracking)[0]}. Errors: {bidirectional_error[~valid_tracking]}")
        return None, False

    tracked_pts_tuples = [(int(pt[0][1]), int(pt[0][0])) for pt in next_pts_klt_good] # (row,col)
    # print(f"KLT Tracking successful. Bidirectional errors: {bidirectional_error}") # Less verbose
    return tracked_pts_tuples, True


# --- REVISED: Load Image Sequence from Single .npy ---
def load_image_sequence(npy_file_path):
    """
    Loads an image sequence from a single .npy file.
    Assumes the structure after data = np.load(...).item() is {'X': array},
    and the array shape is (frame_size, 1, height, width).

    Args:
        npy_file_path (str): Path to the .npy file.

    Returns:
        list: A list of NumPy arrays (uint8, 2D grayscale), where each array is a frame.
              Returns empty list on failure.
    """
    frames = []
    if not npy_file_path.endswith('.npy'):
        print(f"Error: Input path '{npy_file_path}' is not a .npy file.")
        return []

    print(f"Loading sequence from single .npy file: {npy_file_path}")
    try:
        # Load the dictionary containing the data
        data_dict = np.load(npy_file_path, allow_pickle=True)

        # Check if loading resulted in ndarray instead of dict (if allow_pickle=False used during save)
        if isinstance(data_dict, np.ndarray):
             print("Warning: Loaded data is an ndarray directly. Assuming it's the sequence.")
             raw_frames_data = data_dict.item()['X']
             raw_frames_data = raw_frames_data.reshape(-1, 1, 224, 224)
        elif isinstance(data_dict, np.lib.npyio.NpzFile):
             print("Warning: Loaded data is an NpzFile. Trying to access default key 'arr_0'.")
             # Or adapt if you know the key name used during np.savez
             if 'arr_0' in data_dict:
                  raw_frames_data = data_dict['arr_0']
             else:
                   print(f"Error: Cannot find default data array in NpzFile: {npy_file_path}")
                   return []
        else:
             # Assume it's a dictionary loaded via allow_pickle=True
             data_dict = data_dict.item() # Convert 0-dim array object to dict
             if 'X' not in data_dict:
                 print(f"Error: Key 'X' not found in the dictionary loaded from {npy_file_path}")
                 return []
             raw_frames_data = data_dict['X']


        # Validate shape: expecting (frame_size, 1, height, width)
        if raw_frames_data.ndim != 4 or raw_frames_data.shape[1] != 1:
            print(f"Error: Unexpected data shape in {npy_file_path}.")
            print(f"Expected (frame_size, 1, height, width), but got {raw_frames_data.shape}")
            return []

        num_frames = raw_frames_data.shape[0]
        height = raw_frames_data.shape[2]
        width = raw_frames_data.shape[3]
        print(f"Detected shape: ({num_frames}, 1, {height}, {width})")

        for i in range(num_frames):
            # Extract frame: shape becomes (1, height, width)
            frame_data = raw_frames_data[i]
            # Remove the channel dimension: shape becomes (height, width)
            frame_2d = frame_data.squeeze(axis=0) # Or frame_data[0]

            # --- IMPORTANT: Normalize and Convert to uint8 ---
            # KLT works best with uint8, and ensures consistency
            if frame_2d.dtype != np.uint8:
                # Normalize frame to 0-255 range before converting to uint8
                frame_2d_normalized = cv2.normalize(frame_2d, None, 0, 255, cv2.NORM_MINMAX)
                frame_2d_uint8 = frame_2d_normalized.astype(np.uint8)
            else:
                frame_2d_uint8 = frame_2d # Already uint8

            frames.append(frame_2d_uint8)

        print(f"Successfully processed and extracted {len(frames)} frames.")
        return frames

    except FileNotFoundError:
        print(f"Error: File not found at {npy_file_path}")
        return []
    except Exception as e:
        print(f"Error loading or processing the .npy file {npy_file_path}: {e}")
        return []


# --- Main Execution (Adjusted for single .npy input) ---
if __name__ == "__main__":
    # --- Specify the path to your SINGLE .npy file ---
    image_sequence_path = 'complete_HMC_QU/A2C/folds/fold_0/inference_data/ES0001_CH2_1.npy' # MODIFY THIS PATH

    # --- Load Image Sequence ---
    image_sequence = load_image_sequence(image_sequence_path)

    if not image_sequence:
        print("Error: Could not load image sequence. Exiting.")
        exit()

    print(f"Loaded {len(image_sequence)} frames for processing.")

    # --- Parameters (Keep or adjust as needed) ---
    ac_alpha = 0.01
    ac_beta = 5.0
    ac_gamma = 0.005
    ac_max_iter = 1000
    ac_convergence = 0.5
    klt_win_size = (25, 25)
    klt_max_level = 3
    klt_bidir_thresh = 2.0
    poly_order_fit = 4
    num_colored_segments_def = 3 # Basal/Mid/Apical

    # --- Process Frames ---
    results = [] # Store results for each frame
    prev_frame = None
    current_start_pt, current_apex_pt, current_end_pt = None, None, None

    for i, current_frame in enumerate(image_sequence):
        print(f"\n--- Processing Frame {i} ---")

        # --- 1. Get Key Points ---
        if i == 0:
            start_pt_manual, apex_pt_manual, end_pt_manual = get_points(current_frame)
            if start_pt_manual is None: print("Initial point selection failed. Exiting."); exit()
            current_start_pt, current_apex_pt, current_end_pt = start_pt_manual, apex_pt_manual, end_pt_manual
            print(f"Frame 0 Manual Points: S={current_start_pt}, A={current_apex_pt}, E={current_end_pt}")
        else:
            if prev_frame is None or current_start_pt is None:
                 print("Error: Missing previous frame data for tracking. Stopping."); break
            # print(f"Tracking points from Frame {i-1} to {i}...") # Less verbose
            prev_pts_list = [current_start_pt, current_apex_pt, current_end_pt]
            tracked_pts_list, track_status = track_points_klt(prev_frame, current_frame, prev_pts_list, win_size=klt_win_size, max_level=klt_max_level, bidirectional_error_threshold=klt_bidir_thresh)

            if not track_status or tracked_pts_list is None or len(tracked_pts_list) != 3:
                print(f"!!! Point tracking failed for Frame {i}. Stopping sequence processing. !!!")
                results.append({'frame_index': i, 'status': 'tracking_failed'})
                break # Stop processing
            else:
                current_start_pt, current_apex_pt, current_end_pt = tracked_pts_list[0], tracked_pts_list[1], tracked_pts_list[2]
                print(f"Frame {i} Tracked Points: S={current_start_pt}, A={current_apex_pt}, E={current_end_pt}")

        # --- Steps 2-5: Run segmentation pipeline ---
        frame_results = {'frame_index': i, 'status': 'processing'}
        try:
            # --- 2. Initial Contour ---
            initial_contour = create_initial_triangle(current_start_pt, current_apex_pt, current_end_pt)
            frame_results['initial_contour'] = initial_contour
            frame_results['start_pt'] = current_start_pt
            frame_results['apex_pt'] = current_apex_pt
            frame_results['end_pt'] = current_end_pt

            # --- 3. Evolve Active Contour ---
            # print("Evolving Active Contour...") # Less verbose
            evolved_contour = evolve_active_contour(current_frame, initial_contour, alpha=ac_alpha, beta=ac_beta, gamma=ac_gamma, max_iterations=ac_max_iter, convergence=ac_convergence)
            if evolved_contour is None:
                print(f"Active contour failed for Frame {i}.")
                frame_results['status'] = 'contour_failed'; frame_results['error'] = 'Contour evolution None'
                results.append(frame_results)
                prev_frame = current_frame # Still update prev_frame
                continue # Skip rest of pipeline for this frame

            frame_results['evolved_contour'] = evolved_contour

            # --- 4. Fit Polynomials ---
            # print("Fitting Polynomials...") # Less verbose
            coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(evolved_contour, current_apex_pt, poly_order=poly_order_fit)
            frame_results['coeffs_left'] = coeffs_l; frame_results['coeffs_right'] = coeffs_r
            frame_results['points_left_orig'] = points_l_orig; frame_results['points_right_orig'] = points_r_orig

            # --- 5. Segment Polynomials ---
            # print("Segmenting Polynomials...") # Less verbose
            segments_left = []
            segments_right = []
            if coeffs_l is not None and points_l_orig is not None:
                segments_left = segment_polynomials(coeffs_l, points_l_orig, current_apex_pt, num_colored_segments=num_colored_segments_def)
            if coeffs_r is not None and points_r_orig is not None:
                segments_right = segment_polynomials(coeffs_r, points_r_orig, current_apex_pt, num_colored_segments=num_colored_segments_def)
            frame_results['segments_left'] = segments_left; frame_results['segments_right'] = segments_right
            frame_results['status'] = 'success'

        except Exception as e:
             print(f"!!! Error during pipeline processing for Frame {i}: {e} !!!")
             frame_results['status'] = 'pipeline_error'; frame_results['error'] = str(e)

        results.append(frame_results)
        prev_frame = current_frame.copy() # Use copy!
        break

    # --- 6. Visualization (Show last successfully processed frame) ---
    last_successful_frame_index = -1
    for idx in range(len(results) - 1, -1, -1):
         if results[idx].get('status') == 'success':
              last_successful_frame_index = idx; break

    if last_successful_frame_index != -1:
        print(f"\n--- Displaying Results for Frame {last_successful_frame_index} ---")
        res = results[last_successful_frame_index]
        img_to_show = image_sequence[last_successful_frame_index]
        start_pt_disp, apex_pt_disp, end_pt_disp = res['start_pt'], res['apex_pt'], res['end_pt']
        init_contour_disp, evolved_contour_disp = res['initial_contour'], res['evolved_contour']
        segments_l_disp, segments_r_disp = res['segments_left'], res['segments_right']

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        plt.suptitle(f"Endocardial Boundary Extraction Stages (Frame {last_successful_frame_index})", fontsize=16)

        # Panel 1: Input
        ax = axes[0]; ax.imshow(img_to_show, cmap='gray'); ax.set_title("Input Echo"); ax.axis('off')

        # Panel 2: Points & Initial Contour
        ax = axes[1]; ax.imshow(img_to_show, cmap='gray')
        point_marker = 'go' if last_successful_frame_index > 0 else 'ro'
        label_suffix = " (Tracked)" if last_successful_frame_index > 0 else " (Manual)"
        ax.plot(start_pt_disp[1], start_pt_disp[0], point_marker, markersize=6)
        ax.text(start_pt_disp[1]+2, start_pt_disp[0] + 15, 'S'+label_suffix, color='white', ha='left', fontsize=8)
        ax.plot(apex_pt_disp[1], apex_pt_disp[0], point_marker, markersize=6)
        ax.text(apex_pt_disp[1], apex_pt_disp[0] - 15, 'A'+label_suffix, color='white', ha='center', va='bottom', fontsize=8)
        ax.plot(end_pt_disp[1], end_pt_disp[0], point_marker, markersize=6)
        ax.text(end_pt_disp[1]-2, end_pt_disp[0] + 15, 'E'+label_suffix, color='white', ha='right', fontsize=8)
        ax.plot(init_contour_disp[:, 1], init_contour_disp[:, 0], 'r-', lw=1.5)
        ax.set_title("Points & Initial Contour"); ax.axis('off')

        # Panel 3: Evolved Active Contour
        ax = axes[2]; ax.imshow(img_to_show, cmap='gray')
        if evolved_contour_disp is not None: ax.plot(evolved_contour_disp[:, 1], evolved_contour_disp[:, 0], 'r-', lw=2)
        ax.plot(start_pt_disp[1], start_pt_disp[0], point_marker, markersize=5)
        ax.plot(apex_pt_disp[1], apex_pt_disp[0], point_marker, markersize=5)
        ax.plot(end_pt_disp[1], end_pt_disp[0], point_marker, markersize=5)
        ax.set_title("Evolved Active Contour"); ax.axis('off')

        # Panel 4: Active Polynomials (Segmented)
        ax = axes[3]; ax.imshow(img_to_show, cmap='gray')
        colors_left = ['lime', 'yellow', 'cyan'] # Apical, Mid, Basal
        colors_right = ['red', 'blue', 'magenta'] # Apical, Mid, Basal
        num_expected_segments = num_colored_segments_def + 1

        # Plot Left Segments
        if segments_l_disp and len(segments_l_disp) >= 1:
             if len(segments_l_disp) != num_expected_segments: print(f"Warn L: {len(segments_l_disp)} segs")
             for i, seg in enumerate(segments_l_disp):
                  if seg is not None and len(seg) > 0:
                      color = 'white' if i == 0 else (colors_left[i-1] if i-1 < len(colors_left) else 'gray')
                      if len(seg) > 1: ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
                      else: ax.plot(seg[0, 1], seg[0, 0], '.', color=color, markersize=3) # Plot single point

        # Plot Right Segments
        if segments_r_disp and len(segments_r_disp) >= 1:
             if len(segments_r_disp) != num_expected_segments: print(f"Warn R: {len(segments_r_disp)} segs")
             for i, seg in enumerate(segments_r_disp):
                  if seg is not None and len(seg) > 0:
                      color = 'white' if i == 0 else (colors_right[i-1] if i-1 < len(colors_right) else 'gray')
                      if len(seg) > 1: ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
                      else: ax.plot(seg[0, 1], seg[0, 0], '.', color=color, markersize=3) # Plot single point

        ax.set_title("Segmented Polynomials"); ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    else:
        print("\nNo frames were successfully processed through the entire pipeline.")
        # Optional: Show the first frame's initial points
        if results and 'start_pt' in results[0]:
             print("Showing initial points selected on Frame 0.")
             fig, ax = plt.subplots(figsize=(8, 8))
             ax.imshow(image_sequence[0], cmap='gray')
             ax.plot(results[0]['start_pt'][1], results[0]['start_pt'][0], 'ro', ms=5); ax.text(results[0]['start_pt'][1], results[0]['start_pt'][0]+15, 'START', c='w', ha='c')
             ax.plot(results[0]['apex_pt'][1], results[0]['apex_pt'][0], 'ro', ms=5); ax.text(results[0]['apex_pt'][1], results[0]['apex_pt'][0]-15, 'APEX', c='w', ha='c', va='b')
             ax.plot(results[0]['end_pt'][1], results[0]['end_pt'][0], 'ro', ms=5); ax.text(results[0]['end_pt'][1], results[0]['end_pt'][0]+15, 'END', c='w', ha='c')
             ax.set_title("Frame 0 - Initial Manual Points"); ax.axis('off'); plt.show()