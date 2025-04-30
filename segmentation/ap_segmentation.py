import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.measure import approximate_polygon # To simplify contour points if needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- (Keep onclick, get_points, create_initial_triangle, ---
# ---  evolve_active_contour, fit_polynomials) ---
# --- Global variable for point selection ---
points = []
fig_select = None
ax_select = None
img_select = None

# --- Helper Function for Manual Point Selection ---
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
        # Convert (col, row) points to (row, col) for skimage functions
        start_pt = (points[0][1], points[0][0])
        apex_pt = (points[1][1], points[1][0])
        end_pt = (points[2][1], points[2][0])
        return start_pt, apex_pt, end_pt
    else:
        print("Error: Did not select 3 points.")
        return None, None, None

# --- Function to create Initial Triangle Contour ---
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
                          max_iterations=2500, convergence=0.1):
    """Evolves the active contour starting from the initial shape."""
    img_smooth = gaussian(image, sigma=1, preserve_range=False)

    try:
        # REMOVED the 'coordinates' argument if it caused errors previously
        snake = active_contour(img_smooth,
                               initial_contour,
                               alpha=alpha, beta=beta, gamma=gamma,
                               max_num_iter=max_iterations, convergence=convergence)
        return snake
    except Exception as e:
        print(f"Error during active contour evolution: {e}")
        return None

# --- Fit Polynomials ---
def fit_polynomials(contour_points, apex_pt_orig, poly_order=4):
    """
    Splits contour near the original apex and fits polynomials.
    Fits X = P(Y) i.e., Col = P(Row).
    """
    if contour_points is None or len(contour_points) < 10:
        print("Error: Not enough contour points for fitting.")
        return None, None, None, None

    distances = np.sqrt(np.sum((contour_points - np.array(apex_pt_orig))**2, axis=1))
    apex_index = np.argmin(distances)

    if apex_index == 0: apex_index = 1
    if apex_index == len(contour_points) - 1: apex_index = len(contour_points) - 2

    points_left = contour_points[0:apex_index+1]
    points_right = contour_points[apex_index:]

    if len(points_left) < poly_order + 1 or len(points_right) < poly_order + 1:
        print("Error: Not enough points in left or right segments for polyfit.")
        return None, None, points_left, points_right

    coeffs_left = None
    coeffs_right = None

    try:
        coeffs_left = np.polyfit(points_left[:, 0], points_left[:, 1], poly_order)
        coeffs_right = np.polyfit(points_right[:, 0], points_right[:, 1], poly_order)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error during polynomial fitting: {e}.")
        return None, None, points_left, points_right

    return coeffs_left, coeffs_right, points_left, points_right


# --- MODIFIED Segment the Fitted Polynomials ---
def segment_polynomials(coeffs, side_points, num_colored_segments=3):
    """
    Generates points along the fitted polynomial and divides them into segments.
    Returns num_colored_segments + 1 segments, where the first segment
    corresponds to the apex region (intended to be white).

    Args:
        coeffs (np.ndarray): Polynomial coefficients (for col = P(row)).
        side_points (np.ndarray): The original points for this side (row, col).
        num_colored_segments (int): Number of functional segments (e.g., 3 for basal/mid/apical).

    Returns:
        list: A list of arrays, where each array contains the (row, col) points
              for one segment along the smooth polynomial curve. The list will
              contain num_colored_segments + 1 arrays. The first array is the apex segment.
    """
    if coeffs is None or len(side_points) < 2:
        return []

    num_total_segments = num_colored_segments + 1 # Add 1 for the white apex segment

    min_row = np.min(side_points[:, 0])
    max_row = np.max(side_points[:, 0])

    if max_row <= min_row:
        smooth_rows = np.array([min_row])
    else:
        smooth_rows = np.linspace(min_row, max_row, 200) # Generate enough points

    smooth_cols = np.polyval(coeffs, smooth_rows)
    smooth_curve_points = np.vstack((smooth_rows, smooth_cols)).T

    # --- CRITICAL: Ensure points are ordered from APEX outwards ---
    # We want the *first* segment to be the one near the apex.
    # Check distance from the *apex point* of the original side_points
    apex_orig_side = side_points[-1] if np.linalg.norm(side_points[-1] - side_points[0]) > np.linalg.norm(side_points[0] - side_points[-1]) else side_points[0] # Heuristic: apex is usually further from start
    # A better apex point might be needed if this heuristic fails
    # Use the last point of the side_points as the reference for apex end of the segment
    apex_end_ref_point = side_points[-1]

    dist_from_apex_end = np.sqrt(np.sum((smooth_curve_points - apex_end_ref_point)**2, axis=1))

    # If the point generated at index 0 is further from the apex-end than the point at index 1,
    # it means the curve was generated basal->apical, so reverse it to be apical->basal.
    if len(dist_from_apex_end) > 1 and dist_from_apex_end[0] > dist_from_apex_end[1]:
         smooth_curve_points = smooth_curve_points[::-1]
    # Now smooth_curve_points[0] should be the point closest to the apex end of this side's fit

    total_smooth_points = len(smooth_curve_points)
    if total_smooth_points < num_total_segments:
        print(f"Warning: Not enough points ({total_smooth_points}) generated to create {num_total_segments} segments. Returning fewer segments.")
        # Fallback: return segments based on available points
        points_per_segment = 1
        num_total_segments = total_smooth_points if total_smooth_points > 0 else 0


    elif total_smooth_points > 0 :
         # Use integer division, handle remainder for last segment (basal)
        points_per_segment = total_smooth_points // num_total_segments


    segments = []
    start_idx = 0
    if total_smooth_points > 0:
        for i in range(num_total_segments):
            end_idx = start_idx + points_per_segment
            # For the last segment (most basal), include all remaining points
            if i == num_total_segments - 1:
                end_idx = total_smooth_points

            # Handle edge cases
            if start_idx >= total_smooth_points:
                break
            end_idx = min(end_idx, total_smooth_points)

            segment_points = smooth_curve_points[start_idx:end_idx]
            if len(segment_points) > 0:
               segments.append(segment_points)
            start_idx = end_idx

            if start_idx >= total_smooth_points and i < num_total_segments - 1:
                 print(f"Warning: Ran out of points for segmentation at segment {i+1}")
                 break

    # segments[0] is apex (white)
    # segments[1] is apical (e.g., lime/yellow)
    # segments[2] is mid (e.g., red/blue)
    # segments[3] is basal (e.g., cyan/magenta)
    return segments


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Image ---
    image_path = 'complete_HMC_QU/A4C/folds/fold_0/inference_data/ES0001 _4CH_1.npy'

    image_orig = np.load(image_path, allow_pickle=True).item()['X']
    image_gray = image_orig.reshape(-1, 224, 224)[0]
    if image_orig is None:
        print(f"Error: Could not load image at {image_path}")
        exit()

    # --- 1. Get Key Points ---
    start_pt, apex_pt, end_pt = get_points(image_gray)
    if start_pt is None:
        print("Exiting.")
        exit()

    # --- 2. Create Initial Contour ---
    initial_contour = create_initial_triangle(start_pt, apex_pt, end_pt)

    # --- 3. Evolve Active Contour ---
    print("Evolving Active Contour... (This may take a moment)")
    # --- !!! PARAMETER TUNING IS CRITICAL HERE !!! ---
    evolved_contour = evolve_active_contour(image_gray, initial_contour,
                                            alpha=0.01,
                                            beta=5.0,
                                            gamma=0.005,
                                            max_iterations=1000,
                                            convergence=0.5)

    if evolved_contour is None:
        print("Active contour failed. Exiting.")
        exit()

    # --- 4. Fit Polynomials ---
    coeffs_l, coeffs_r, points_l_orig, points_r_orig = fit_polynomials(evolved_contour, apex_pt)

    # --- 5. Segment Polynomials (now returns 4 segments) ---
    segments_left = []
    segments_right = []
    num_colored_segments_def = 3 # Define the number of *colored* segments needed
    if coeffs_l is not None:
        segments_left = segment_polynomials(coeffs_l, points_l_orig, num_colored_segments=num_colored_segments_def)
    if coeffs_r is not None:
        segments_right = segment_polynomials(coeffs_r, points_r_orig, num_colored_segments=num_colored_segments_def)

    # --- 6. Visualization (Mimicking Figure 3) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.suptitle("Endocardial Boundary Extraction Stages", fontsize=16)

    # --- Panels 1, 2, 3 are unchanged ---
    # Panel 1: Input
    ax = axes[0]
    ax.imshow(image_gray, cmap='gray')
    ax.set_title("Input Echocardiography")
    ax.axis('off')
    # Panel 2: Initial Contour
    ax = axes[1]
    ax.imshow(image_gray, cmap='gray')
    ax.plot(start_pt[1], start_pt[0], 'ro', markersize=5)
    ax.text(start_pt[1], start_pt[0] + 15, 'START', color='white', ha='center')
    ax.plot(apex_pt[1], apex_pt[0], 'ro', markersize=5)
    ax.text(apex_pt[1], apex_pt[0] - 15, 'APEX', color='white', ha='center', va='bottom')
    ax.plot(end_pt[1], end_pt[0], 'ro', markersize=5)
    ax.text(end_pt[1], end_pt[0] + 15, 'END', color='white', ha='center')
    ax.plot(initial_contour[:, 1], initial_contour[:, 0], 'r-', lw=1.5, label='Initial Contour')
    ax.set_title("Initial Contour (Triangle)")
    ax.axis('off')
    # Panel 3: Evolved Active Contour
    ax = axes[2]
    ax.imshow(image_gray, cmap='gray')
    ax.plot(evolved_contour[:, 1], evolved_contour[:, 0], 'r-', lw=2, label='Evolved Contour')
    ax.plot(start_pt[1], start_pt[0], 'bo', markersize=5)
    ax.text(start_pt[1], start_pt[0] + 15, 'START', color='white', ha='center')
    ax.plot(apex_pt[1], apex_pt[0], 'bo', markersize=5)
    ax.text(apex_pt[1], apex_pt[0] - 15, 'APEX', color='white', ha='center', va='bottom')
    ax.plot(end_pt[1], end_pt[0], 'bo', markersize=5)
    ax.text(end_pt[1], end_pt[0] + 15, 'END', color='white', ha='center')
    ax.set_title("Evolved Active Contour")
    ax.axis('off')


    # --- MODIFIED Panel 4: Active Polynomials (Segmented with White Apex) ---
    ax = axes[3]
    ax.imshow(image_gray, cmap='gray')

    # Define colors for the 3 functional segments (Apical, Mid, Basal)
    # Order matters: colors[0]=Apical, colors[1]=Mid, colors[2]=Basal
    colors_left_functional = ['red', 'lime', 'cyan']    # Apical, Mid, Basal
    colors_right_functional = ['blue', 'yellow', 'magenta'] # Apical, Mid, Basal

    # Plot Left Segments (expecting 4 segments: white, apical, mid, basal)
    if len(segments_left) == num_colored_segments_def + 1:
        for i, seg in enumerate(segments_left):
            if len(seg) > 1:
                if i == 3: # Last segment is apex
                    color = 'white'
                else: # Map to functional colors (index 1->0, 2->1, 3->2)
                    color_index = i - 1
                    if color_index < len(colors_left_functional):
                         color = colors_left_functional[color_index]
                    else: # Fallback if something went wrong
                         color = 'gray'
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
    elif len(segments_left) > 0: # Fallback if segmentation didn't produce 4 parts
         print("Warning: Plotting left side with fewer than expected segments.")
         # Just plot all available segments in sequence with colors (no explicit white)
         for i, seg in enumerate(segments_left):
              if len(seg) > 1:
                    color = colors_left_functional[i % len(colors_left_functional)]
                    ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)


    # Plot Right Segments (expecting 4 segments: white, apical, mid, basal)
    if len(segments_right) == num_colored_segments_def + 1:
        for i, seg in enumerate(segments_right):
            if len(seg) > 3:
                if i == 3: # Last segment is apex
                    color = 'white'
                else: # Map to functional colors (index 1->0, 2->1, 3->2)
                    color_index = i - 1
                    if color_index < len(colors_right_functional):
                         color = colors_right_functional[color_index]
                    else: # Fallback
                         color = 'gray'
                ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)
    elif len(segments_right) > 0: # Fallback
         print("Warning: Plotting right side with fewer than expected segments.")
         for i, seg in enumerate(segments_right):
              if len(seg) > 1:
                    color = colors_right_functional[i % len(colors_right_functional)]
                    ax.plot(seg[:, 1], seg[:, 0], '-', color=color, lw=3)


    ax.set_title("Active Polynomials (Segmented)")
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()