import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import tqdm
import torch.nn.functional as F

from model.unet import Unet

import hmc_load
from sklearn.model_selection import train_test_split, KFold
from metrics import batch_metric, accuracy, precision, recall, f_score, specificity
import argparse 


def train(dataset, n_channels=1, n_classes=2, out_pth=None, model_name='unet', folds=5, epochs=25, lr=10e-3):

    if out_pth is None:
        out_pth = os.path.join(os.getcwd(), 'model_weights')

    try:
        os.mkdir(out_pth)
    except:
        pass

    metrics = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_start = 0

    for i in range(fold_start, folds):

        fold_name = 'fold'+str(i)

        best_loss = float('inf')

        if model_name.lower() == 'unet':
            model = Unet(n_channels, n_classes)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        best_model = model_name + '_weights.pth'

        metrics[i] = []

        hmc_loader = {x: DataLoader(dataset[i][x], batch_size=1, shuffle=True, pin_memory=True)
                      for x in ['train', 'valid']
                      }

        for epoch in range(epochs):

            print(f'Fold {i} Epoch {epoch+1}/{epochs}')
            print('-'*60)

            for phase in ['train', 'valid']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                n = 0
                sets = 0

                total_acc = 0
                total_pre = 0
                total_rec = 0
                total_f1s = 0
                total_spe = 0

                with tqdm.tqdm(total=len(hmc_loader[phase])) as pbar:

                    for img, gt_msk in hmc_loader[phase]:

                        img = img.to(device)
                        gt_msk = gt_msk.to(device)

                        with torch.set_grad_enabled(phase == 'train'):

                            pr_msk = model(img)

                            loss = criterion(pr_msk, gt_msk)

                            msk_pr = F.one_hot(F.softmax(pr_msk, dim=1).argmax(
                                dim=1), n_classes).permute(0, 3, 1, 2).float()
                            msk_gt = F.one_hot(gt_msk, n_classes).permute(
                                0, 3, 1, 2).float()

                            if phase == 'train':

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                            total_acc += batch_metric(msk_pr, msk_gt, accuracy)
                            total_pre += batch_metric(msk_pr,
                                                      msk_gt, precision)
                            total_rec += batch_metric(msk_pr, msk_gt, recall)
                            total_f1s += batch_metric(msk_pr, msk_gt, f_score)
                            total_spe += batch_metric(msk_pr,
                                                      msk_gt, specificity)

                            running_loss += loss.item() * img.size(0)
                            n += 1
                            sets += img.size(0)

                            pbar.set_postfix_str("{:.2f} ({:.2f})".format(
                                running_loss / sets, loss.item()))
                            pbar.update()

                epoch_loss = running_loss / sets
                epoch_acc = total_acc / n
                epoch_rec = total_rec / n
                epoch_spe = total_spe / n
                epoch_pre = total_pre / n
                epoch_f1s = total_f1s / n

                if phase == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model, os.path.join(out_pth, 'best_loss.pth'))
                    print("Model Saved!")

                print(f'{phase.title()} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f} Sensitivity: {epoch_rec:.4f} Specificity: {epoch_spe:.4f} Precision: {epoch_pre:.4f} F1: {epoch_f1s:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model using pre-split K-Fold data from NumPy arrays.")
    # --- Input Arguments ---
    parser.add_argument('--npy_dir', type=str, default='hmc_kfold_numpy',
                        help='Directory containing the pre-split K-Fold NumPy files (X_train_kfold.npy, etc.)')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model architecture (e.g., unet)')
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs per fold')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # Adjusted default LR, 10e-3 might be high
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes (including background)')
    # --- Output ---
    parser.add_argument('--output_dir', type=str, default='model_weights_kfold',
                        help='Directory to save trained model weights for each fold')
    # --- Validation Split ---
    parser.add_argument('--valid_size', type=float, default=0.1,
                        help='Proportion of the training data per fold to use for validation')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for splitting train data into train/validation sets within each fold')

    args = parser.parse_args()

    # --- Load the pre-processed NumPy data ---
    print(f"Loading K-Fold data from: {args.npy_dir}")
    try:
        x_train_folds = np.load(os.path.join(args.npy_dir, 'X_train_kfold.npy'), allow_pickle=True)
        y_train_folds = np.load(os.path.join(args.npy_dir, 'y_train_kfold.npy'), allow_pickle=True)
        # x_test_folds = np.load(os.path.join(args.npy_dir, 'X_test_kfold.npy'), allow_pickle=True) # Not used by train function
        # y_test_folds = np.load(os.path.join(args.npy_dir, 'y_test_kfold.npy'), allow_pickle=True) # Not used by train function
        print(f"Loaded data for {len(x_train_folds)} folds.")
    except FileNotFoundError as e:
        print(f"Error loading .npy files: {e}")
        print("Please ensure the .npy files (X_train_kfold.npy, y_train_kfold.npy) exist in the specified directory.")
        exit() # Exit if files are not found
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        exit()


    num_folds = len(x_train_folds)
    if num_folds == 0:
        print("Error: No folds found in the loaded NumPy arrays.")
        exit()

    # --- Prepare the 'dataset' dictionary for the train function ---
    dataset = {}
    print(f"Preparing dataset structure for {num_folds} folds...")

    for i in range(num_folds):
        print(f"  Processing Fold {i}...")
        fold_X_train_np = x_train_folds[i] # NumPy array for fold i: (N_train_frames, C, H, W) or similar
        fold_y_train_np = y_train_folds[i] # NumPy array for fold i: (N_train_frames, H, W)

        if fold_X_train_np.size == 0 or fold_y_train_np.size == 0:
             print(f"  Warning: Fold {i} contains empty training data. Skipping fold preparation.")
             # Add empty entries to maintain structure if needed, or handle in train function
             dataset[i] = {'train': [], 'valid': []}
             continue

        # --- Split this fold's training data into train/validation ---
        num_train_samples = fold_X_train_np.shape[0]
        indices = np.arange(num_train_samples)

        if num_train_samples <= 1: # Cannot split if only 1 sample or less
            print(f"  Warning: Not enough samples ({num_train_samples}) in Fold {i} training data to create a validation split. Using all data for training.")
            train_indices = indices
            valid_indices = [] # No validation set
        elif args.valid_size <= 0 or args.valid_size >= 1:
            print(f"  Warning: Invalid validation size ({args.valid_size}). Using all data for training.")
            train_indices = indices
            valid_indices = []
        else:
            try:
                train_indices, valid_indices = train_test_split(
                    indices,
                    test_size=args.valid_size,
                    random_state=args.split_seed # Use a seed for reproducibility
                    # Add stratify=fold_y_train_np if labels are suitable for stratification (might need reshaping/summarizing labels)
                )
                print(f"    Split Fold {i} into {len(train_indices)} train / {len(valid_indices)} valid samples.")
            except Exception as e:
                print(f"    Error during train/valid split for Fold {i}: {e}. Using all data for training.")
                train_indices = indices
                valid_indices = []


        # --- Create lists of (image_tensor, mask_tensor) tuples ---
        dataset[i] = {}
        dataset[i]['train'] = []
        dataset[i]['valid'] = []

        # Process training subset
        for idx in train_indices:
            img_np = fold_X_train_np[idx] # Shape (C, H, W) or (H, W, C) depending on saving
            mask_np = fold_y_train_np[idx] # Shape (H, W)

            # Convert to PyTorch tensors
            # Assuming image is (C, H, W), convert directly
            # If image is (H, W, C), you might need to permute dims: .permute(2, 0, 1)
            img_tensor = torch.from_numpy(img_np).float()
            mask_tensor = torch.from_numpy(mask_np).long() # Masks typically need torch.long for CrossEntropyLoss

            dataset[i]['train'].append((img_tensor, mask_tensor))

        # Process validation subset
        for idx in valid_indices:
            img_np = fold_X_train_np[idx]
            mask_np = fold_y_train_np[idx]

            img_tensor = torch.from_numpy(img_np).float()
            mask_tensor = torch.from_numpy(mask_np).long()

            dataset[i]['valid'].append((img_tensor, mask_tensor))

    # --- Call the training function ---
    print("\nStarting training process...")
    # Determine n_channels from the first sample of the first fold if possible
    n_channels = 1 # Default
    if 0 in dataset and dataset[0]['train']:
        n_channels = dataset[0]['train'][0][0].shape[0]
        print(f"Determined n_channels={n_channels} from loaded data.")
    elif 0 in dataset and dataset[0]['valid']:
         n_channels = dataset[0]['valid'][0][0].shape[0]
         print(f"Determined n_channels={n_channels} from loaded data (validation set).")
    else:
        print(f"Warning: Could not determine n_channels from data. Using default {n_channels}.")


    train(
        dataset=dataset,
        n_channels=n_channels,
        n_classes=args.n_classes,
        out_pth=args.output_dir,
        model_name=args.model_name,
        folds=num_folds, # Pass the actual number of folds loaded
        epochs=args.epochs,
        lr=args.lr
    )

    print("\n--- Script Finished ---")