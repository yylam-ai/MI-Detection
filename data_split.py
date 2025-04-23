import numpy as np
import os

from segmentation import hmc_load
from sklearn.model_selection import KFold
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HMC dataset with K-Fold split and save as NumPy arrays.")
    parser.add_argument('-d', '--data_root', type=str, default='Complete_HMC_QU',
                        help='Root directory containing the HMC dataset (e.g., A4C.xlsx, HMC-QU/, etc.)')
    parser.add_argument('-o', '--output_dir', type=str, default='hmc_kfold_numpy',
                        help='Directory where the NumPy array files will be saved.')
    parser.add_argument('--seed', type=int, default=9999, help='Random seed for KFold shuffling.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-Fold splits.')
    # Add arguments for desired numpy data types (optional)
    parser.add_argument('--img_dtype', type=str, default='float32', help='NumPy dtype for raw images (e.g., float32, uint8)')
    parser.add_argument('--mask_dtype', type=str, default='uint8', help='NumPy dtype for segmentation masks (e.g., uint8, int32)')


    args = parser.parse_args()

    print(f"Data Root: {args.data_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Num Splits: {args.n_splits}")
    print(f"Random Seed: {args.seed}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset definition (replace with your actual class)
    print("Loading HMC dataset metadata...")
    hmc = hmc_load.HMCDataset(root=args.data_root)
    print(f"Dataset contains {len(hmc)} items (videos).")

    # Prepare KFold
    # Assuming 109 is the total number of videos/items in your dataset
    total_items = len(hmc)
    idxs = np.arange(total_items)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # Lists to hold the data for each fold
    # Each element in these lists will be a NumPy array containing all frames for that fold/split
    all_folds_X_train = []
    all_folds_y_train = []
    all_folds_X_test = []
    all_folds_y_test = []

    print(f"Processing {args.n_splits} folds...")
    # Loop through the KFold splits
    for i, (train_indices, test_indices) in enumerate(kf.split(idxs)):
        print(f"\n--- Processing Fold {i} ---")

        # --- Process Training Data for Fold i ---
        fold_train_raw_frames = []
        fold_train_seg_frames = []
        print(f"  Processing {len(train_indices)} training videos...")
        for vid_idx in train_indices:
            # Get raw video tensor and segmentation tensor for the current video index
            # Assuming hmc[idx] returns (raw_vid_tensor, seg_vid_tensor)
            # raw_vid_tensor shape: (Channels, Frames, H, W)
            # seg_vid_tensor shape: (Frames, H, W)
            try:
                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]
            except Exception as e:
                print(f"    Error loading data for index {vid_idx}: {e}")
                continue # Skip this video if loading fails

            num_frames = seg_vid_tensor.shape[0] # Get number of frames from seg tensor

            # Iterate through frames of the current video
            for frame_idx in range(num_frames):
                # Extract the raw image frame: (Channels, H, W)
                # Need to move channel axis if needed later (e.g., for libraries expecting H, W, C)
                # Keep as (C, H, W) for now, consistent with typical PyTorch input
                raw_img_tensor = raw_vid_tensor[:, frame_idx, :, :]
                # Extract the segmentation mask frame: (H, W)
                seg_img_tensor = seg_vid_tensor[frame_idx, :, :]

                # Convert tensors to NumPy arrays with specified dtypes
                # Use .detach() if tensors might have gradients attached
                raw_img_np = raw_img_tensor.detach().numpy().astype(getattr(np, args.img_dtype))
                seg_img_np = seg_img_tensor.detach().numpy().astype(getattr(np, args.mask_dtype))

                fold_train_raw_frames.append(raw_img_np)
                fold_train_seg_frames.append(seg_img_np)

        # Stack all frames for the current fold's training set into single NumPy arrays
        if fold_train_raw_frames: # Check if list is not empty
             fold_X_train = np.stack(fold_train_raw_frames, axis=0) # Shape: (TotalTrainFrames, C, H, W)
             fold_y_train = np.stack(fold_train_seg_frames, axis=0) # Shape: (TotalTrainFrames, H, W)
             print(f"  Train Fold {i}: X shape={fold_X_train.shape}, y shape={fold_y_train.shape}")
             all_folds_X_train.append(fold_X_train)
             all_folds_y_train.append(fold_y_train)
        else:
             print(f"  Warning: No training data processed for fold {i}.")
             # Append empty arrays or handle as needed if this case is possible/problematic
             all_folds_X_train.append(np.array([]))
             all_folds_y_train.append(np.array([]))


        # --- Process Test Data for Fold i ---
        fold_test_raw_frames = []
        fold_test_seg_frames = []
        print(f"  Processing {len(test_indices)} test videos...")
        for vid_idx in test_indices:
            try:
                raw_vid_tensor, seg_vid_tensor = hmc[vid_idx]
            except Exception as e:
                print(f"    Error loading data for index {vid_idx}: {e}")
                continue

            num_frames = seg_vid_tensor.shape[0]

            for frame_idx in range(num_frames):
                raw_img_tensor = raw_vid_tensor[:, frame_idx, :, :]
                seg_img_tensor = seg_vid_tensor[frame_idx, :, :]

                raw_img_np = raw_img_tensor.detach().numpy().astype(getattr(np, args.img_dtype))
                seg_img_np = seg_img_tensor.detach().numpy().astype(getattr(np, args.mask_dtype))

                fold_test_raw_frames.append(raw_img_np)
                fold_test_seg_frames.append(seg_img_np)

        # Stack all frames for the current fold's test set
        if fold_test_raw_frames:
            fold_X_test = np.stack(fold_test_raw_frames, axis=0) # Shape: (TotalTestFrames, C, H, W)
            fold_y_test = np.stack(fold_test_seg_frames, axis=0) # Shape: (TotalTestFrames, H, W)
            print(f"  Test Fold {i}: X shape={fold_X_test.shape}, y shape={fold_y_test.shape}")
            all_folds_X_test.append(fold_X_test)
            all_folds_y_test.append(fold_y_test)
        else:
            print(f"  Warning: No testing data processed for fold {i}.")
            all_folds_X_test.append(np.array([]))
            all_folds_y_test.append(np.array([]))

    # --- Save the collected data ---
    # The lists all_folds_X_train etc. now contain N NumPy arrays, where N = n_splits
    # We save these lists directly. NumPy handles saving lists/tuples of arrays using pickling.
    print("\n--- Saving data to .npy files ---")

    # Define output file paths
    x_train_file = os.path.join(args.output_dir, 'X_train_kfold.npy')
    y_train_file = os.path.join(args.output_dir, 'y_train_kfold.npy')
    x_test_file = os.path.join(args.output_dir, 'X_test_kfold.npy')
    y_test_file = os.path.join(args.output_dir, 'y_test_kfold.npy')

    # Save the lists of arrays
    # allow_pickle=True is necessary because we are saving a list of NumPy arrays,
    # which might have different shapes/sizes per fold.
    np.save(x_train_file, np.array(all_folds_X_train, dtype=object), allow_pickle=True)
    print(f"Saved X_train data for all folds to: {x_train_file}")
    np.save(y_train_file, np.array(all_folds_y_train, dtype=object), allow_pickle=True)
    print(f"Saved y_train data for all folds to: {y_train_file}")
    np.save(x_test_file, np.array(all_folds_X_test, dtype=object), allow_pickle=True)
    print(f"Saved X_test data for all folds to: {x_test_file}")
    np.save(y_test_file, np.array(all_folds_y_test, dtype=object), allow_pickle=True)
    print(f"Saved y_test data for all folds to: {y_test_file}")

    print("\n--- Finished ---")
    print(f"To load the data later, use np.load('path/to/file.npy', allow_pickle=True)")
    print("Example: loaded_x_train = np.load('{}', allow_pickle=True)".format(x_train_file))
    print("Access fold 0 training data: loaded_x_train[0]")