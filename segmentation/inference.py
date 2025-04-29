import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import glob
from model.unet import Unet


# --- Helper Dataset for processing frames WITHIN a single video ---
class SingleVideoFrameDataset(Dataset):
    """Dataset to serve frames from a single video's NumPy array."""
    def __init__(self, video_frames_np):
        """
        Args:
            video_frames_np (np.ndarray): NumPy array of video frames, shape (C, F, H, W).
        """
        # Expecting (C, F, H, W), we iterate over F (dim 1)
        if video_frames_np.ndim != 4:
             raise ValueError(f"Expected video_frames_np to have 4 dimensions (C, F, H, W), but got {video_frames_np.ndim}")
        self.video_frames_np = video_frames_np
        self.num_frames = video_frames_np.shape[1]
        self.num_channels = video_frames_np.shape[0]
        self.height = video_frames_np.shape[2]
        self.width = video_frames_np.shape[3]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_frames:
            raise IndexError(f"Index {idx} out of bounds for video with {self.num_frames} frames.")

        # Extract the frame: shape (C, H, W)
        img_frame_np = self.video_frames_np[:, idx, :, :]
        # Convert to PyTorch tensor (float32 is standard for input)
        img_tensor = torch.from_numpy(img_frame_np).float()
        return img_tensor

# --- Main Inference Function ---
def run_inference(args):
    """Loads model, iterates through test videos, performs inference, and saves predictions."""

    print("--- Starting Inference ---")
    print(f"Target Fold Index:   {args.fold_index}")
    print(f"Data Directory:      {args.data_dir}")
    print(f"Model Path:     {args.model_path}")
    print(f"Output Directory:    {args.output_dir}")
    print(f"Model Base Name:     {args.model_base_name}")
    print(f"Num Classes:         {args.n_classes}")
    print(f"Batch Size (frames): {args.batch_size}")
    print(f"Output Format:       npy (video masks)")
    print("-------------------------")

    # --- 1. Locate Test Data for the Fold ---
    fold_dir = os.path.join(args.data_dir, f'fold_{args.fold_index}')
    test_data_dir = os.path.join(fold_dir, 'inference_data')

    if not os.path.isdir(test_data_dir):
        print(f"Error: Test data directory not found for fold {args.fold_index}: {test_data_dir}")
        return

    # Find all video .npy files in the test directory
    test_video_files = sorted(glob.glob(os.path.join(test_data_dir, '*.npy')))

    if not test_video_files:
        print(f"Error: No .npy files found in {test_data_dir}")
        return

    print(f"Found {len(test_video_files)} test video files for fold {args.fold_index}.")

    # --- 2. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Load Model ---
    # Determine n_channels by loading the first video file
    try:
        first_video_data = np.load(test_video_files[0], allow_pickle=True).item()
        if 'X' not in first_video_data or first_video_data['X'].ndim != 4:
             raise ValueError("Key 'X' not found or has incorrect dimensions (expected C,F,H,W) in first video file.")
        n_channels = first_video_data['X'].shape[0]
        print(f"Inferred number of input channels: {n_channels} from {os.path.basename(test_video_files[0])}")
    except Exception as e:
        print(f"Error determining n_channels from first video file ({test_video_files[0]}): {e}")
        print("Cannot proceed without knowing the number of input channels.")
        return

    # Construct model path based on convention from training script
    model_path = f"{args.model_path}"

    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}")
        print("Ensure the model directory and fold index are correct and training produced the file.")
        return

    try:
        model = Unet(n_channels=n_channels, n_classes=args.n_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # --- 4. Prepare Output Directory ---
    fold_output_dir = os.path.join(args.output_dir, f'fold_{args.fold_index}')
    os.makedirs(fold_output_dir, exist_ok=True)
    print(f"Saving predictions to: {fold_output_dir}")

    # --- 5. Run Inference Video by Video ---
    total_videos_processed = 0
    total_videos_failed = 0

    print(f"\nRunning inference on {len(test_video_files)} videos...")
    # Outer loop iterates through each test video file
    for video_path in tqdm.tqdm(test_video_files, desc="Processing Videos"):
        video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_filename_base}_pred.npy"
        output_filepath = os.path.join(fold_output_dir, output_filename)

        try:
            # Load the data for the current video
            video_data = np.load(video_path, allow_pickle=True).item()
            if 'X' not in video_data or not isinstance(video_data['X'], np.ndarray) or video_data['X'].ndim != 4:
                 print(f"\nWarning: Skipping video {video_filename_base}. Invalid or missing 'X' data (expected C,F,H,W).")
                 total_videos_failed += 1
                 continue

            input_frames_np = video_data['X'] # Shape (C, F, H, W)
            if input_frames_np.shape[1] == 0: # Check if video has frames
                 print(f"\nWarning: Skipping video {video_filename_base}. Contains 0 frames.")
                 total_videos_failed += 1
                 continue


            # Create a temporary dataset and dataloader for the frames of THIS video
            single_video_dataset = SingleVideoFrameDataset(input_frames_np)
            single_video_loader = DataLoader(
                single_video_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )

            video_predictions = [] # List to store predicted masks for this video's frames

            # Inner loop iterates through batches of frames for the current video
            with torch.no_grad():
                for img_frame_batch in single_video_loader:
                    img_frame_batch = img_frame_batch.to(device, non_blocking=True) # (B, C, H, W)

                    # Get model prediction (logits)
                    logits = model(img_frame_batch) # Shape: (B, n_classes, H, W)

                    # Convert logits to prediction class indices
                    pred_masks_batch = torch.argmax(logits, dim=1) # Shape: (B, H, W)

                    # Move predictions to CPU and convert to NumPy (uint8 suitable for masks)
                    pred_masks_np = pred_masks_batch.cpu().numpy().astype(np.uint8)

                    # Append each predicted mask in the batch to the list
                    for i in range(pred_masks_np.shape[0]):
                        video_predictions.append(pred_masks_np[i]) # Append (H, W) array

            # Check if any predictions were generated for the video
            if not video_predictions:
                 print(f"\nWarning: No predictions generated for video {video_filename_base}. Possible issue during frame processing.")
                 total_videos_failed += 1
                 continue

            # Stack all predicted frames for this video into a single array
            # Expected final shape: (F, H, W)
            final_video_pred_mask = np.stack(video_predictions, axis=0)

            # Save the combined prediction mask for the video
            np.save(output_filepath, final_video_pred_mask)
            total_videos_processed += 1

        except Exception as e:
            print(f"\nError processing video {video_filename_base}: {e}")
            total_videos_failed += 1
            # Optionally: continue to next video or break/re-raise

    print(f"\n--- Inference Complete ---")
    print(f"Successfully processed and saved predictions for {total_videos_processed} videos.")
    if total_videos_failed > 0:
        print(f"Failed to process {total_videos_failed} videos. Check warnings/errors above.")
    print(f"Predictions saved in: {fold_output_dir}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using a trained UNet model on K-Fold test data (video .npy files)."
    )

    # --- Input/Output Arguments ---
    parser.add_argument('--data_dir', type=str, default="complete_HMC_QU/A4C/folds",
                        help='Root directory containing the fold folders (e.g., fold_0/test/, fold_1/test/, ...)')
    parser.add_argument('--model_path', type=str, default="model_weights/segmentation_A4C/unet_best_loss_fold0.pth",
                        help='Root directory containing the fold-specific model weight folders (e.g., model_weights_kfold_video/fold_0/unet.pth)')
    parser.add_argument('--output_dir', type=str, default='inference_results/A4C',
                        help='Base directory where fold-specific prediction output folders will be created.')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='Index of the fold (model and test data) to use (e.g., 0, 1, ...)')

    # --- Model Arguments ---
    parser.add_argument('--model_base_name', type=str, default='unet',
                        help='Base name of the saved model files (e.g., "unet" for "unet_best_loss_foldX.pth")')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of output classes the model was trained with.')
    # n_channels is inferred from data

    # --- Inference Parameters ---
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing frames within each video.')
    parser.add_argument('--num_workers', type=int, default=2, # Lower default for inference might be safer
                        help='Number of worker processes for DataLoader (loading frames within a video).')
    # save_format is implicitly npy now

    args = parser.parse_args()

    run_inference(args)