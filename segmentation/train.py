import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import os
import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import glob 
import argparse
from model.unet import Unet
from metrics import batch_metric, accuracy, precision, recall, f_score, specificity

# --- Custom Dataset for Loading Video Frames ---
class VideoFrameDataset(Dataset):
    """
    PyTorch Dataset for loading frames from video .npy files.
    Each .npy file is expected to contain a dictionary {'X': video_data, 'y': mask_data}.
    """
    def __init__(self, npy_file_paths, n_classes, img_dtype=torch.float32, mask_dtype=torch.long):
        """
        Args:
            npy_file_paths (list): List of paths to .npy files (each representing a video).
            n_classes (int): Number of segmentation classes.
            img_dtype (torch.dtype): Desired dtype for image tensors.
            mask_dtype (torch.dtype): Desired dtype for mask tensors (usually torch.long for CrossEntropy).
        """
        super().__init__()
        self.file_paths = npy_file_paths
        self.n_classes = n_classes
        self.img_dtype = img_dtype
        self.mask_dtype = mask_dtype

        self.frame_mapping = [] # List to store tuples: (file_path_index, frame_index_within_video)
        self.frames_per_video = [] # Store frame counts to avoid reloading just for len
        self.loaded_data_cache = {} # Optional: Cache loaded npy files if memory allows

        print(f"  Initializing dataset with {len(self.file_paths)} video files...")
        total_frames = 0
        for i, file_path in enumerate(tqdm.tqdm(self.file_paths, desc="  Scanning video files", leave=False)):
            try:
                # Load only once to get frame count and cache if needed (or just get count)
                if file_path not in self.loaded_data_cache:
                     # Load the dictionary, '.item()' retrieves the dict from the 0-d array wrapper
                     data = np.load(file_path, allow_pickle=True).item()
                     # --- Basic validation of loaded data ---
                     if 'X' not in data or 'y' not in data:
                         print(f"Warning: Skipping file {file_path} - Missing 'X' or 'y' key.")
                         continue
                     if not isinstance(data['X'], np.ndarray) or not isinstance(data['y'], np.ndarray):
                         print(f"Warning: Skipping file {file_path} - 'X' or 'y' is not a numpy array.")
                         continue
                     if data['X'].ndim < 3 or data['y'].ndim < 3: # Expecting C,F,H,W for X and F,H,W for y
                         print(f"Warning: Skipping file {file_path} - Insufficient dimensions (X:{data['X'].ndim}, y:{data['y'].ndim}).")
                         continue
                     # --- End basic validation ---

                     # Assuming 'y' shape is (F, H, W) or 'X' shape is (C, F, H, W)
                     num_frames = data.get('y', np.array([])).shape[0] # Prioritize 'y' for frame count
                     if num_frames == 0 and data.get('X', np.array([])).ndim > 1:
                         num_frames = data['X'].shape[1] # Fallback to X dim 1 if y is missing/empty

                     if num_frames == 0:
                         print(f"Warning: Skipping file {file_path} - Could not determine number of frames (shape y: {data.get('y', 'N/A').shape}, shape X: {data.get('X', 'N/A').shape}).")
                         continue

                     # Optional Caching (be mindful of memory)
                     # self.loaded_data_cache[file_path] = data
                else:
                    # data = self.loaded_data_cache[file_path] # Use cache if implementing
                    # num_frames = data['y'].shape[0] # Or get from cached data structure
                    # Temporarily re-load just for frame count if not caching full data
                    num_frames = np.load(file_path, allow_pickle=True).item()['y'].shape[0]


                self.frames_per_video.append(num_frames)
                for frame_idx in range(num_frames):
                    self.frame_mapping.append((i, frame_idx)) # Store index of file path and frame index
                total_frames += num_frames

            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}. Skipping.")
                self.frames_per_video.append(0) # Add 0 frames for skipped file to maintain index alignment

        print(f"  Dataset initialized. Total frames: {total_frames} from {len(self.file_paths) - self.frames_per_video.count(0)} valid videos.")
        if total_frames == 0 and len(self.file_paths) > 0:
             print("Error: No valid frames found in the provided video files.")
             # Consider raising an error if no frames are found

    def __len__(self):
        """Return the total number of frames across all videos."""
        return len(self.frame_mapping)

    def __getitem__(self, idx):
        """Return the idx-th frame and its corresponding mask."""
        if idx >= len(self.frame_mapping):
            raise IndexError("Index out of bounds")

        file_idx, frame_idx_in_video = self.frame_mapping[idx]
        file_path = self.file_paths[file_idx]

        try:
            # Load data for the specific file if not cached
            # if file_path in self.loaded_data_cache:
            #     data = self.loaded_data_cache[file_path]
            # else:
            data = np.load(file_path, allow_pickle=True).item()

            # Extract the specific frame
            # Assuming X shape is (C, F, H, W)
            img_frame_np = data['X'][:, frame_idx_in_video, :, :]
            # Assuming y shape is (F, H, W)
            mask_frame_np = data['y'][frame_idx_in_video, :, :]

            # Convert to PyTorch tensors with specified dtypes
            img_tensor = torch.from_numpy(img_frame_np).to(self.img_dtype)
            mask_tensor = torch.from_numpy(mask_frame_np).to(self.mask_dtype)

            return img_tensor, mask_tensor

        except Exception as e:
            print(f"Error loading frame {idx} (file: {file_path}, frame_in_vid: {frame_idx_in_video}): {e}")
            # Return dummy data or raise error? Returning dummy might hide issues.
            # Let's try returning None and handle in DataLoader collation or training loop.
            # Or raise the error to stop training. Raising is often better.
            raise RuntimeError(f"Failed to load frame {idx} from {file_path}") from e


# --- Training Function (Mostly unchanged, adjusted DataLoader part) ---
def train(datasets_per_fold, n_channels=1, n_classes=2, out_pth=None, model_name='unet', epochs=25, lr=1e-4, batch_size=4):
    """
    Trains the model using K-Fold data structure where each fold has its own Dataset objects.

    Args:
        datasets_per_fold (dict): Dict where keys are fold indices (0, 1, ...) and
                                   values are dicts {'train': Dataset, 'valid': Dataset}.
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        out_pth (str): Path to save model weights.
        model_name (str): Name of the model architecture.
        epochs (int): Number of training epochs per fold.
        lr (float): Learning rate.
        batch_size (int): Batch size for DataLoader.
    """

    if out_pth is None:
        out_pth = os.path.join(os.getcwd(), 'model_weights_kfold')

    os.makedirs(out_pth, exist_ok=True)

    metrics = {} # Store metrics per fold if needed later
    num_folds = len(datasets_per_fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting Training on {device} ---")
    print(f"Number of folds: {num_folds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Output path: {out_pth}")
    print(f"-----------------------------")


    for i in range(num_folds): # Loop through the folds provided in the dictionary keys

        if i not in datasets_per_fold or not datasets_per_fold[i]['train'] or not datasets_per_fold[i]['valid']:
             print(f"\n--- Skipping Fold {i} (Missing train or valid dataset) ---")
             continue

        print(f'\n--- Training Fold {i}/{num_folds-1} ---')

        fold_out_pth = os.path.join(out_pth, f'fold_{i}')
        os.makedirs(fold_out_pth, exist_ok=True)

        best_loss = float('inf')

        # --- Model Initialization ---
        if model_name.lower() == 'unet':
            model = Unet(n_channels, n_classes)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        model.to(device)
        # --- End Model Initialization ---

        # --- Optimizer and Loss ---
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Ensure masks are LongTensor for CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # --- End Optimizer and Loss ---

        # --- DataLoaders ---
        # Use the Dataset objects passed in datasets_per_fold
        try:
            dataloaders = {
                'train': DataLoader(datasets_per_fold[i]['train'], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4), # Adjust num_workers
                'valid': DataLoader(datasets_per_fold[i]['valid'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4) # No shuffle for valid
            }
            print(f"  Train samples (frames): {len(datasets_per_fold[i]['train'])}")
            print(f"  Valid samples (frames): {len(datasets_per_fold[i]['valid'])}")
            if len(datasets_per_fold[i]['train']) == 0 or len(datasets_per_fold[i]['valid']) == 0:
                 print(f"  Warning: Train or validation set has 0 frames for Fold {i}. Skipping epoch loop.")
                 continue # Skip to next fold if no data
        except Exception as e:
             print(f"  Error creating DataLoaders for Fold {i}: {e}")
             continue # Skip to next fold
        # --- End DataLoaders ---

        metrics[i] = [] # Store epoch metrics if needed

        for epoch in range(epochs):
            print(f'  Epoch {epoch+1}/{epochs}')
            print('  '+'-'*60)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                n_batches = 0
                total_samples = 0 # Count total frames processed in the epoch

                # Initialize metric accumulators for the epoch
                epoch_acc = 0.0
                epoch_pre = 0.0
                epoch_rec = 0.0
                epoch_f1s = 0.0
                epoch_spe = 0.0

                # Use tqdm context manager
                pbar = tqdm.tqdm(total=len(dataloaders[phase]), desc=f"    {phase.title():<5}", leave=False)

                for img_batch, gt_msk_batch in dataloaders[phase]:
                    img_batch = img_batch.to(device)     # Shape: (B, C, H, W)
                    gt_msk_batch = gt_msk_batch.to(device) # Shape: (B, H, W), dtype=long

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pr_msk_logits = model(img_batch) # Shape: (B, N_Classes, H, W)
                        loss = criterion(pr_msk_logits, gt_msk_batch)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # --- Calculate Metrics ---
                        # Apply Softmax and Argmax to get predicted class indices
                        pr_msk_indices = torch.argmax(F.softmax(pr_msk_logits, dim=1), dim=1) # Shape: (B, H, W)

                        # Convert to one-hot for metric functions (if they expect one-hot)
                        # Ensure masks are on CPU and detached for metric calculation if functions require numpy/non-gradient tensors
                        # Your batch_metric function seems to handle tensors directly, which is good.
                        # Make sure gt_msk_batch is also LongTensor if needed by F.one_hot
                        with torch.no_grad():
                             msk_pr_onehot = F.one_hot(pr_msk_indices, n_classes).permute(0, 3, 1, 2).float() # (B, N_Classes, H, W)
                             msk_gt_onehot = F.one_hot(gt_msk_batch, n_classes).permute(0, 3, 1, 2).float() # (B, N_Classes, H, W)

                             # Accumulate metrics (summing over batches)
                             # Note: batch_metric likely returns the sum/average metric *for the batch*. Need to aggregate correctly.
                             # Let's assume batch_metric returns the SUM of the metric across the batch items.
                             epoch_acc += batch_metric(msk_pr_onehot, msk_gt_onehot, accuracy)
                             epoch_pre += batch_metric(msk_pr_onehot, msk_gt_onehot, precision)
                             epoch_rec += batch_metric(msk_pr_onehot, msk_gt_onehot, recall)
                             epoch_f1s += batch_metric(msk_pr_onehot, msk_gt_onehot, f_score)
                             epoch_spe += batch_metric(msk_pr_onehot, msk_gt_onehot, specificity)
                        # --- End Metric Calculation ---

                        running_loss += loss.item() * img_batch.size(0)
                        n_batches += 1
                        total_samples += img_batch.size(0)

                        pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))
                        pbar.update()

                pbar.close() # Close tqdm bar for the phase

                if total_samples == 0:
                     print(f"    {phase.title()} phase skipped - no samples processed.")
                     continue # Skip epoch calculation if no samples

                epoch_loss = running_loss / total_samples
                # Average metrics over all BATCHES. If batch_metric returns sum, divide by n_batches.
                # If batch_metric returns average, need to weight by batch size -> total_metric / total_samples
                # Assuming batch_metric returns SUM for the batch, divide by n_batches for average per batch.
                # (Or better: modify batch_metric to return sum AND count, or adjust calculation here)
                # For now, assuming division by n_batches gives a reasonable average metric value. Revisit if metric definition is different.
                final_epoch_acc = epoch_acc / n_batches
                final_epoch_rec = epoch_rec / n_batches
                final_epoch_spe = epoch_spe / n_batches
                final_epoch_pre = epoch_pre / n_batches
                final_epoch_f1s = epoch_f1s / n_batches


                print(f'    {phase.title()} Loss: {epoch_loss:.4f} Acc: {final_epoch_acc:.4f} Sens: {final_epoch_rec:.4f} Spec: {final_epoch_spe:.4f} Prec: {final_epoch_pre:.4f} F1: {final_epoch_f1s:.4f}')

                # --- Save Best Model (based on validation loss) ---
                if phase == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_path = os.path.join(fold_out_pth, f'{model_name}_best_loss_fold{i}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"    --> Model improved! Saved to {best_model_path}")
                # --- End Save Best Model ---
        # --- End Epoch Loop ---
    # --- End Fold Loop ---

    print("\n--- Training Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet model using K-Fold data stored as individual video .npy files."
    )
    # --- Input Arguments ---
    parser.add_argument('--data_dir', type=str, default="complete_HMC_QU/hmc_kfold_time_series",
                        help='Root directory containing the fold folders (e.g., fold_0, fold_1, ...)')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model architecture (e.g., unet)')
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs per fold')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (number of frames per batch)')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes (including background)')
    # --- Output ---
    parser.add_argument('--output_dir', type=str, default='model_weights_kfold_video',
                        help='Directory to save trained model weights for each fold')
    # --- Validation Split ---
    parser.add_argument('--valid_size', type=float, default=0.15,
                        help='Proportion of the training *videos* per fold to use for validation')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for splitting train video files into train/validation sets within each fold')

    args = parser.parse_args()

    # --- Basic Validation ---
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        exit(1)

    # --- Find Fold Directories ---
    fold_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'fold_*')))
    num_folds = len(fold_dirs)

    if num_folds == 0:
        print(f"Error: No 'fold_*' directories found in {args.data_dir}")
        exit(1)

    print(f"Found {num_folds} fold directories in {args.data_dir}")

    # --- Prepare the 'datasets_per_fold' dictionary ---
    datasets_per_fold = {}
    print(f"\nPreparing dataset structure for {num_folds} folds...")
    print(f"Using validation split size: {args.valid_size}, random seed: {args.split_seed}")

    n_channels_determined = None # To store the channel count once found

    for i in range(num_folds):
        fold_path = os.path.join(args.data_dir, f'fold_{i}')
        train_data_path = os.path.join(fold_path, 'train')
        # test_data_path = os.path.join(fold_path, 'test') # Test data is not used during training loop

        print(f"\nProcessing Fold {i}...")
        if not os.path.isdir(train_data_path):
            print(f"  Warning: Train directory not found for fold {i}: {train_data_path}. Skipping fold.")
            datasets_per_fold[i] = {'train': None, 'valid': None} # Mark as unavailable
            continue

        # List all .npy files in the training directory for this fold
        all_train_video_files = sorted(glob.glob(os.path.join(train_data_path, '*.npy')))

        if not all_train_video_files:
            print(f"  Warning: No .npy files found in train directory for fold {i}: {train_data_path}. Skipping fold.")
            datasets_per_fold[i] = {'train': None, 'valid': None}
            continue

        print(f"  Found {len(all_train_video_files)} video files for training in fold {i}.")

        # --- Split video file list into train/validation ---
        if args.valid_size <= 0 or args.valid_size >= 1:
            print(f"  Validation size is {args.valid_size}. Using all video files for training.")
            train_files = all_train_video_files
            valid_files = []
        elif len(all_train_video_files) < 2 : # Need at least 2 files to split
             print(f"  Warning: Only {len(all_train_video_files)} video file(s) found. Cannot create validation split. Using all for training.")
             train_files = all_train_video_files
             valid_files = []
        else:
            try:
                train_files, valid_files = train_test_split(
                    all_train_video_files,
                    test_size=args.valid_size,
                    random_state=args.split_seed
                )
                print(f"  Split into {len(train_files)} train / {len(valid_files)} validation video files.")
            except Exception as e:
                print(f"  Error during train/valid file split for Fold {i}: {e}. Using all files for training.")
                train_files = all_train_video_files
                valid_files = []

        # --- Create Dataset objects ---
        # Make sure we have files before creating datasets
        train_dataset = None
        valid_dataset = None

        if train_files:
             print(f"  Creating training dataset for fold {i}...")
             train_dataset = VideoFrameDataset(train_files, n_classes=args.n_classes)
             if len(train_dataset) == 0:
                 print("  Warning: Training dataset created but contains 0 frames.")
                 train_dataset = None # Treat as unusable if empty

        if valid_files:
             print(f"  Creating validation dataset for fold {i}...")
             valid_dataset = VideoFrameDataset(valid_files, n_classes=args.n_classes)
             if len(valid_dataset) == 0:
                 print("  Warning: Validation dataset created but contains 0 frames.")
                 valid_dataset = None # Treat as unusable if empty

        # --- Determine n_channels (only once) ---
        if n_channels_determined is None:
             first_file_to_check = None
             if train_dataset and train_files:
                 first_file_to_check = train_files[0]
             elif valid_dataset and valid_files:
                 first_file_to_check = valid_files[0]

             if first_file_to_check:
                 try:
                     temp_data = np.load(first_file_to_check, allow_pickle=True).item()
                     n_channels_determined = temp_data['X'].shape[0] # Assuming C, F, H, W
                     print(f"\nDetermined n_channels = {n_channels_determined} from file {os.path.basename(first_file_to_check)}")
                 except Exception as e:
                     print(f"Warning: Could not determine n_channels from {first_file_to_check}: {e}. Using default 1.")
                     n_channels_determined = 1
             else:
                  # Handle case where both train/valid might be empty for the first fold processed
                  print(f"Warning: Could not determine n_channels for fold {i} as no valid files/datasets were found yet.")
                  # Will try again on the next fold or default later


        datasets_per_fold[i] = {'train': train_dataset, 'valid': valid_dataset}

    # --- Final Check for n_channels ---
    if n_channels_determined is None:
         n_channels_determined = 1 # Default if never determined
         print(f"\nWarning: Could not determine n_channels from any data files. Using default {n_channels_determined}.")

    # --- Call the training function ---
    train(
        datasets_per_fold=datasets_per_fold,
        n_channels=n_channels_determined,
        n_classes=args.n_classes,
        out_pth=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size # Pass batch_size to train function
    )

    print("\n--- Script Finished ---")