# data_split.py
import numpy as np
import os
import sys
import argparse
from sklearn.model_selection import KFold
from segmentation import hmc_load_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess HMC dataset (including videos without masks) for K-Fold split. "
                    "Saves each video as a separate .npy file containing {'X': video_array, 'y': mask_array_or_None}."
    )
    # (Keep existing arguments)
    parser.add_argument('-d', '--data_root', type=str, default='Complete_HMC_QU',
                        help='Root directory containing the HMC dataset (e.g., A4C.xlsx, HMC-QU/, etc.)')
    parser.add_argument('-o', '--output_dir', type=str, default='complete_HMC_QU/A4C/folds', # Default A2C here, user should override
                        help='Root directory where the fold folders (fold_0, fold_1, ...) will be saved.')
    parser.add_argument('--seed', type=int, default=9999, help='Random seed for KFold shuffling.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-Fold splits.')
    parser.add_argument('--img_dtype', type=str, default='float32', help='NumPy dtype for saving video frames (e.g., float32, uint8)')
    parser.add_argument('--mask_dtype', type=str, default='uint8', help='NumPy dtype for saving segmentation masks (e.g., uint8, int32). Ignored if mask is absent.')
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


    # --- Prepare KFold ---
    total_items = len(hmc)
    idxs = np.arange(total_items)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    print(f"\nPreparing {args.n_splits}-Fold split for {total_items} items...")
    # --- End KFold Prep ---


    # --- Process and Save Folds ---
    total_processed_count = 0
    total_failed_count = 0
    total_mask_saved_count = 0
    total_no_mask_saved_count = 0

    for i, (train_indices, test_indices) in enumerate(kf.split(idxs)):
        print(f"\n--- Processing Fold {i} ---")

        fold_dir = os.path.join(args.output_dir, f'fold_{i}')
        train_dir = os.path.join(fold_dir, 'train')
        test_dir = os.path.join(fold_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        print(f"  Output directories created: {train_dir}, {test_dir}")

        # --- Process Training Data for Fold i ---
        print(f"  Processing {len(train_indices)} training items...")
        fold_train_processed = 0
        fold_train_failed = 0
        fold_train_mask_saved = 0
        fold_train_no_mask_saved = 0

        for vid_idx in train_indices:
            fn = "unknown" # Default in case info retrieval fails
            try:
                # Get original filename from dataset info (for saving)
                item_info = hmc.sub_df_info[vid_idx]
                fn = item_info['filename'] # Original base filename (without extension)
                has_mask_expected = item_info['has_mask']

                # Get video tensor and segmentation tensor (mask can be None)
                # hmc[idx] returns (raw_vid_tensor, seg_vid_tensor) or (None, None)
                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]

                # Check if video loading was successful
                if raw_vid_tensor is None:
                     print(f"    Warning: Skipping training item index {vid_idx} (file: {fn}) due to video loading error in HMCDataset.")
                     fold_train_failed += 1
                     continue

                # --- Prepare Video Data ---
                # Use .cpu() if tensors might be on GPU, and .detach() if they have gradients
                video_np = raw_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.img_dtype))

                # --- Prepare Mask Data (handle None) ---
                mask_np = None # Default to None
                mask_saved = False
                if seg_vid_tensor is not None:
                    mask_np = seg_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.mask_dtype))
                    mask_saved = True
                    fold_train_mask_saved += 1
                elif has_mask_expected:
                    # Mask was expected, but HMCDataset returned None for it (likely a .mat read error)
                    print(f"    Warning: Training item {vid_idx} (file: {fn}) had an expected mask, but it failed to load. Saving with mask=None.")
                    fold_train_no_mask_saved += 1 # Count as processed, but without mask
                else:
                    # Mask was not expected, seg_vid_tensor is correctly None
                    fold_train_no_mask_saved += 1 # Count as processed without mask

                # --- Create the dictionary to save ---
                data_to_save = {'X': video_np, 'y': mask_np} # mask_np can be None

                # Construct the output path and save
                output_npy_path = os.path.join(train_dir, f"{fn}.npy")
                np.save(output_npy_path, data_to_save) # np.save handles dicts with None via pickle
                fold_train_processed += 1

            except Exception as e:
                # Catch errors during processing this specific video/item
                print(f"    Error processing training item index {vid_idx} (File: {fn}): {e}")
                # print traceback if needed for debugging
                # import traceback
                # traceback.print_exc()
                fold_train_failed += 1
                continue # Skip this item

        print(f"  Finished training data for fold {i}. Processed: {fold_train_processed} (Masks: {fold_train_mask_saved}, No Mask: {fold_train_no_mask_saved}). Failed/Skipped: {fold_train_failed}")

        # --- Process Test Data for Fold i ---
        print(f"  Processing {len(test_indices)} test items...")
        fold_test_processed = 0
        fold_test_failed = 0
        fold_test_mask_saved = 0
        fold_test_no_mask_saved = 0

        for vid_idx in test_indices:
            fn = "unknown"
            try:
                item_info = hmc.sub_df_info[vid_idx]
                fn = item_info['filename']
                has_mask_expected = item_info['has_mask']

                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]

                if raw_vid_tensor is None:
                     print(f"    Warning: Skipping test item index {vid_idx} (file: {fn}) due to video loading error in HMCDataset.")
                     fold_test_failed += 1
                     continue

                video_np = raw_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.img_dtype))

                mask_np = None
                mask_saved = False
                if seg_vid_tensor is not None:
                    mask_np = seg_vid_tensor.cpu().detach().numpy().astype(getattr(np, args.mask_dtype))
                    mask_saved = True
                    fold_test_mask_saved += 1
                elif has_mask_expected:
                    print(f"    Warning: Test item {vid_idx} (file: {fn}) had an expected mask, but it failed to load. Saving with mask=None.")
                    fold_test_no_mask_saved +=1
                else:
                    fold_test_no_mask_saved +=1


                data_to_save = {'X': video_np, 'y': mask_np}

                output_npy_path = os.path.join(test_dir, f"{fn}.npy")
                np.save(output_npy_path, data_to_save)
                fold_test_processed += 1

            except Exception as e:
                print(f"    Error processing test item index {vid_idx} (File: {fn}): {e}")
                # import traceback
                # traceback.print_exc()
                fold_test_failed += 1
                continue

        print(f"  Finished test data for fold {i}. Processed: {fold_test_processed} (Masks: {fold_test_mask_saved}, No Mask: {fold_test_no_mask_saved}). Failed/Skipped: {fold_test_failed}")

        # Update total counts
        total_processed_count += (fold_train_processed + fold_test_processed)
        total_failed_count += (fold_train_failed + fold_test_failed)
        total_mask_saved_count += (fold_train_mask_saved + fold_test_mask_saved)
        total_no_mask_saved_count += (fold_train_no_mask_saved + fold_test_no_mask_saved)
    # --- End Fold Processing ---


    print("\n--- Preprocessing Finished ---")
    print(f"Successfully processed {total_processed_count} items across {args.n_splits} folds.")
    print(f"  - Items saved with masks: {total_mask_saved_count}")
    print(f"  - Items saved without masks: {total_no_mask_saved_count}")
    if total_failed_count > 0:
        print(f"Warning: Failed to process or skipped {total_failed_count} items. Check logs above for details.")
    print(f"Output data is saved in: {args.output_dir}")

    # Reminder about loading the saved .npy files
    print("\nNote: To load the saved data, use:")
    print("  data = np.load(your_file_path.npy, allow_pickle=True).item()")
    print("  video = data['X']")
    print("  mask = data['y']  # This will be a NumPy array or None")