import numpy as np
import os
import sys # For potentially adding hmc_load path
import argparse
from sklearn.model_selection import KFold

# --- Dynamically find hmc_load ---
# Add parent directory of 'segmentation' to path if hmc_load is inside it
# Adjust this logic based on your actual project structure if needed
try:
    # Assuming data_split.py is at the same level as the 'segmentation' folder
    # or inside it. Adjust if needed.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level if the script is inside 'segmentation'
    if os.path.basename(script_dir) == 'segmentation':
        parent_dir = os.path.dirname(script_dir)
    else:
        parent_dir = script_dir # Assume 'segmentation' is a subdir here

    segmentation_module_path = os.path.join(parent_dir, 'segmentation')
    if segmentation_module_path not in sys.path:
         sys.path.insert(0, parent_dir) # Add parent to allow 'import segmentation.hmc_load'

    # Now import
    from segmentation import hmc_load

except ImportError as e:
     print(f"Error importing hmc_load. Make sure 'hmc_load.py' is accessible.")
     print(f"Current sys.path: {sys.path}")
     print(f"Attempted import path: {segmentation_module_path if 'segmentation_module_path' in locals() else 'N/A'}")
     print(f"Original error: {e}")
     sys.exit(1)
except ModuleNotFoundError as e:
     print(f"Error: Could not find the 'segmentation' module or 'hmc_load' within it.")
     print(f"Make sure your project structure allows the import.")
     print(f"Current sys.path: {sys.path}")
     print(f"Original error: {e}")
     sys.exit(1)
# --- End dynamic import ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess HMC dataset for time series analysis with K-Fold split. "
                    "Saves each video as a separate .npy file within fold/train|test directories."
    )
    parser.add_argument('-d', '--data_root', type=str, default='Complete_HMC_QU',
                        help='Root directory containing the HMC dataset (e.g., A4C.xlsx, HMC-QU/, etc.)')
    parser.add_argument('-o', '--output_dir', type=str, default='hmc_kfold_time_series',
                        help='Root directory where the fold folders (fold_0, fold_1, ...) will be saved.')
    parser.add_argument('--seed', type=int, default=9999, help='Random seed for KFold shuffling.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-Fold splits.')
    parser.add_argument('--img_dtype', type=str, default='float32', help='NumPy dtype for saving video frames (e.g., float32, uint8)')
    parser.add_argument('--mask_dtype', type=str, default='uint8', help='NumPy dtype for saving segmentation masks (e.g., uint8, int32)')
    parser.add_argument('--axis', type=str, default='A4C', help='Which axis view to process (e.g., A4C, A2C)')
    parser.add_argument('--frame_size', type=int, default=224, help='Target size (height and width) for frames.')
    parser.add_argument('--use_cache', action='store_true', help='Use cached processed tensors from HMCDataset if available.')


    args = parser.parse_args()

    print("--- Configuration ---")
    print(f"Data Root:       {args.data_root}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Axis View:       {args.axis}")
    print(f"Frame Size:      {args.frame_size}")
    print(f"Num Splits:      {args.n_splits}")
    print(f"Random Seed:     {args.seed}")
    print(f"Use HMCDataset Cache: {args.use_cache}")
    print(f"Output Image Dtype: {args.img_dtype}")
    print(f"Output Mask Dtype:  {args.mask_dtype}")
    print("---------------------")

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load the dataset definition ---
    print(f"\nLoading HMC dataset metadata for axis '{args.axis}'...")
    try:
        hmc = hmc_load.HMCDataset(root=args.data_root,
                                  axis=args.axis,
                                  frame_size=args.frame_size,
                                  use_cache=args.use_cache)
        if len(hmc) == 0:
            print("Error: HMCDataset loaded successfully but contains 0 items. Check A4C.xlsx and data paths.")
            sys.exit(1)
        print(f"Dataset contains {len(hmc)} items (videos) for axis '{args.axis}'.")
    except (FileNotFoundError, ValueError, IOError, IndexError) as e:
         print(f"Error initializing HMCDataset: {e}")
         print("Please ensure the data root path is correct and A4C.xlsx exists and is valid.")
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during dataset initialization: {e}")
        sys.exit(1)
    # --- End Dataset Loading ---


    # --- Prepare KFold ---
    total_items = len(hmc)
    idxs = np.arange(total_items)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    print(f"\nPreparing {args.n_splits}-Fold split...")
    # --- End KFold Prep ---


    # --- Process and Save Folds ---
    processed_video_count = 0
    failed_video_count = 0

    for i, (train_indices, test_indices) in enumerate(kf.split(idxs)):
        print(f"\n--- Processing Fold {i} ---")

        # Create directories for the current fold
        fold_dir = os.path.join(args.output_dir, f'fold_{i}')
        train_dir = os.path.join(fold_dir, 'train')
        test_dir = os.path.join(fold_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        print(f"  Created directory: {fold_dir}")

        # --- Process Training Data for Fold i ---
        print(f"  Processing {len(train_indices)} training videos...")
        fold_train_processed = 0
        fold_train_failed = 0
        for vid_idx in train_indices:
            try:
                # Get original filename from dataset info
                item_info = hmc.sub_df_info[vid_idx]
                fn = item_info['filename'] # Original base filename (without extension)

                # Get video tensor and segmentation tensor
                # hmc[idx] returns (raw_vid_tensor, seg_vid_tensor) or (None, None)
                # raw_vid_tensor shape: (C, F, H, W)
                # seg_vid_tensor shape: (F, H, W)
                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]

                # Check if loading was successful
                if raw_vid_tensor is None or seg_vid_tensor is None:
                     print(f"    Warning: Skipping video index {vid_idx} (file: {fn}) due to loading error in HMCDataset.")
                     fold_train_failed += 1
                     continue

                # Convert tensors to NumPy arrays with specified dtypes
                # Use .cpu() if tensors might be on GPU, and .detach() if they have gradients
                video_np = raw_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.img_dtype))
                mask_np = seg_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.mask_dtype))

                # Create the dictionary to save
                data_to_save = {'X': video_np, 'y': mask_np}

                # Construct the output path and save
                output_npy_path = os.path.join(train_dir, f"{fn}.npy")
                np.save(output_npy_path, data_to_save) # np.save handles dicts (pickles them)
                fold_train_processed += 1

            except Exception as e:
                # Catch errors during processing a specific video
                print(f"    Error processing training video index {vid_idx} (File: {fn if 'fn' in locals() else 'unknown'}): {e}")
                fold_train_failed += 1
                continue # Skip this video

        print(f"  Finished processing training data for fold {i}. Processed: {fold_train_processed}, Failed: {fold_train_failed}")

        # --- Process Test Data for Fold i ---
        print(f"  Processing {len(test_indices)} test videos...")
        fold_test_processed = 0
        fold_test_failed = 0
        for vid_idx in test_indices:
            try:
                item_info = hmc.sub_df_info[vid_idx]
                fn = item_info['filename']

                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]

                if raw_vid_tensor is None or seg_vid_tensor is None:
                     print(f"    Warning: Skipping video index {vid_idx} (file: {fn}) due to loading error in HMCDataset.")
                     fold_test_failed += 1
                     continue

                video_np = raw_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.img_dtype))
                mask_np = seg_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.mask_dtype))

                data_to_save = {'X': video_np, 'y': mask_np}

                output_npy_path = os.path.join(test_dir, f"{fn}.npy")
                np.save(output_npy_path, data_to_save)
                fold_test_processed += 1

            except Exception as e:
                print(f"    Error processing test video index {vid_idx} (File: {fn if 'fn' in locals() else 'unknown'}): {e}")
                fold_test_failed += 1
                continue

        print(f"  Finished processing test data for fold {i}. Processed: {fold_test_processed}, Failed: {fold_test_failed}")

        processed_video_count += (fold_train_processed + fold_test_processed)
        failed_video_count += (fold_train_failed + fold_test_failed)
    # --- End Fold Processing ---


    print("\n--- Finished ---")
    print(f"Successfully processed and saved data for {processed_video_count} video instances across {args.n_splits} folds.")
    if failed_video_count > 0:
        print(f"Warning: Failed to process {failed_video_count} video instances. Check logs above for details.")
    print(f"Data is saved in: {args.output_dir}")