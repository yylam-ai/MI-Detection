# data_split.py (modified to save all data to a single folder)
import numpy as np
import os
import sys
import argparse
# KFold is no longer needed
# from sklearn.model_selection import KFold
from segmentation import hmc_load_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess HMC dataset (including videos without masks). "
                    "Saves each video as a separate .npy file containing {'X': video_array, 'y': mask_array_or_None} into a single directory."
    )
    parser.add_argument('-d', '--data_root', type=str, default='Complete_HMC_QU',
                        help='Root directory containing the HMC dataset (e.g., A4C.xlsx, HMC-QU/, etc.)')
    parser.add_argument('-o', '--output_dir', type=str, default='complete_HMC_QU/A2C/processed_data', # Modified default
                        help='Directory where the processed .npy files will be saved.')
    # Arguments for KFold (seed, n_splits) are removed
    parser.add_argument('--img_dtype', type=str, default='float32', help='NumPy dtype for saving video frames (e.g., float32, uint8)')
    parser.add_argument('--mask_dtype', type=str, default='uint8', help='NumPy dtype for saving segmentation masks (e.g., uint8, int32). Ignored if mask is absent.')
    parser.add_argument('--axis', type=str, default='A2C', help='Which axis view to process (e.g., A4C, A2C)')
    parser.add_argument('--frame_size', type=int, default=224, help='Target size (height and width) for frames.')
    parser.add_argument('--use_cache', action='store_true', help='Use cached processed tensors from HMCDataset if available.')

    args = parser.parse_args()

    print("--- Configuration ---")
    print(f"Data Root:       {args.data_root}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Axis View:       {args.axis}")
    print(f"Frame Size:      {args.frame_size}")
    # Removed n_splits and seed from config print
    print(f"Use HMCDataset Cache: {args.use_cache}")
    print(f"Output Image Dtype: {args.img_dtype}")
    print(f"Output Mask Dtype:  {args.mask_dtype}")
    print("---------------------")

    # Create base output directory (this will be the single folder for all .npy files)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load the dataset definition ---
    print(f"\nLoading HMC dataset metadata for axis '{args.axis}' (including items without masks)...")
    try:
        hmc = hmc_load_raw.HMCDataset(root=args.data_root,
                                  axis=args.axis,
                                  frame_size=args.frame_size,
                                  use_cache=args.use_cache)
        if len(hmc) == 0:
            print(f"Error: HMCDataset loaded but contains 0 usable items. Check {args.axis}.xlsx, data paths, and start/end frames.")
            sys.exit(1)
        print(f"Dataset definition loaded. Found {len(hmc)} potential items for axis '{args.axis}'.")
    except (FileNotFoundError, ValueError, IOError, IndexError) as e:
         print(f"Error initializing HMCDataset: {e}")
         print(f"Please ensure the data root path ('{args.data_root}') is correct, {args.axis}.xlsx exists and is valid.")
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during dataset initialization: {e}")
        sys.exit(1)
    # --- End Dataset Loading ---

    # KFold preparation is removed

    # --- Process and Save Data ---
    print(f"\nProcessing {len(hmc)} items to be saved in '{args.output_dir}'...")
    total_processed_count = 0
    total_failed_count = 0
    total_mask_saved_count = 0
    total_no_mask_saved_count = 0

    # Iterate through all items in the dataset
    for vid_idx in range(len(hmc)):
        fn = "unknown" # Default in case info retrieval fails
        try:
            # Get original filename from dataset info (for saving)
            item_info = hmc.sub_df_info[vid_idx]
            fn = item_info['filename'] # Original base filename (without extension)
            has_mask_expected = item_info['has_mask']

            # Get video tensor and segmentation tensor (mask can be None)
            raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]

            # Check if video loading was successful
            if raw_vid_tensor is None:
                 print(f"    Warning: Skipping item index {vid_idx} (file: {fn}) due to video loading error in HMCDataset.")
                 total_failed_count += 1
                 continue

            # --- Prepare Video Data ---
            video_np = raw_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.img_dtype))

            # --- Prepare Mask Data (handle None) ---
            mask_np = None # Default to None
            if seg_vid_tensor is not None:
                mask_np = seg_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.mask_dtype))
                total_mask_saved_count += 1
            elif has_mask_expected:
                # Mask was expected, but HMCDataset returned None for it
                print(f"    Warning: Item {vid_idx} (file: {fn}) had an expected mask, but it failed to load. Saving with mask=None.")
                total_no_mask_saved_count += 1
            else:
                # Mask was not expected, seg_vid_tensor is correctly None
                total_no_mask_saved_count += 1

            # --- Create the dictionary to save ---
            data_to_save = {'X': video_np, 'y': mask_np} # mask_np can be None

            # Construct the output path (directly into args.output_dir) and save
            output_npy_path = os.path.join(args.output_dir, f"{fn}.npy")
            np.save(output_npy_path, data_to_save) # np.save handles dicts with None via pickle
            total_processed_count += 1

            if (vid_idx + 1) % 50 == 0 or (vid_idx + 1) == len(hmc): # Print progress
                print(f"  Processed {vid_idx + 1}/{len(hmc)} items...")

        except Exception as e:
            # Catch errors during processing this specific video/item
            print(f"    Error processing item index {vid_idx} (File: {fn}): {e}")
            # For detailed debugging, uncomment the next two lines:
            # import traceback
            # traceback.print_exc()
            total_failed_count += 1
            continue # Skip this item

    print(f"Finished processing all items.")
    # --- End Data Processing ---


    print("\n--- Preprocessing Finished ---")
    print(f"Successfully processed {total_processed_count} items.")
    print(f"  - Items saved with masks: {total_mask_saved_count}")
    print(f"  - Items saved without masks: {total_no_mask_saved_count}")
    if total_failed_count > 0:
        print(f"Warning: Failed to process or skipped {total_failed_count} items. Check logs above for details.")
    print(f"Output data is saved in: {args.output_dir}")

    # Reminder about loading the saved .npy files
    print("\nNote: To load the saved data, use:")
    print("  data = np.load('your_file_path.npy', allow_pickle=True).item()")
    print("  video = data['X']")
    print("  mask = data['y']  # This will be a NumPy array or None")