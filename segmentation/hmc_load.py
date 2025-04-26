import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io
import torchvision
import os
import torch

class HMCDataset(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, axis='A4C', frame_size=224, use_cache=True):
        super().__init__(root)
        self.root = root if root is not None else os.getcwd()
        self.axis = axis.upper()
        self.frame_size = frame_size
        self.use_cache = use_cache # Store the flag

        # --- Cache Directory Setup ---
        self.cache_dir = os.path.join(self.root, 'processed_cache', self.axis) # Cache specific to axis
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True) # Create cache dir if it doesn't exist
        # --- End Cache Directory Setup ---

        self.output_path = os.path.join(self.root, 'model_weights') # This seems unrelated to dataset loading itself

        self.seg_path = os.path.join(self.root, 'LV Ground-truth Segmentation Masks')
        self.vid_path = os.path.join(self.root, 'HMC-QU', self.axis)
        excel_path = os.path.join(self.root, f'{axis}.xlsx')

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Error: {axis}.xlsx not found at {excel_path}.")

        try:
            self.df = pd.read_excel(excel_path)
            print(self.df)
        except Exception as e:
            raise IOError(f"Error reading {axis}.xlsx: {e}")

        # --- Robust Column Handling ---
        # Assuming standard structure: Check for 'ECHO' and last 3 cols usually for start/end/mask
        if 'ECHO' not in self.df.columns:
             raise ValueError(f"Error: 'ECHO' column not found in {axis}.xlsx.")
        if self.df.shape[1] < 4:
             raise IndexError(f"Error: {axis}.xlsx does not have enough columns (expected at least 4).")

        # Use column names if available, otherwise fall back to indices cautiously
        echo_col_name = 'ECHO'
        mask_availability_col_name = self.df.columns[-1]
        end_frame_col_name = self.df.columns[-2]
        start_frame_col_name = self.df.columns[-3]
        # --- End Robust Column Handling ---


        mask_availability_col = self.df[mask_availability_col_name]
        valid_mask_indices = mask_availability_col == 'ü'

        if not valid_mask_indices.any():
            print(f"Warning: No entries with 'ü' found in the last column ('{mask_availability_col_name}') of {axis}.xlsx. Dataset will be empty.")
            self.sub_df_info = [] # Store relevant info as a list of tuples/dicts
            self.a4c_fn = []
        else:
            # Store filename, start index, end index directly
            self.sub_df_info = []
            valid_df = self.df.loc[valid_mask_indices]
            for _, row in valid_df.iterrows():
                 fn = str(row[echo_col_name]) # Ensure filename is string
                 # Use .get() for robustness if column names might vary slightly or have NaNs
                 s_idx = row.get(start_frame_col_name)
                 e_idx = row.get(end_frame_col_name)

                 # --- Handle potential NaN or non-numeric values ---
                 try:
                      s_idx = int(s_idx) if pd.notna(s_idx) else None
                      e_idx = int(e_idx) if pd.notna(e_idx) else None
                 except (ValueError, TypeError):
                      print(f"Warning: Could not convert start/end indices for {fn} to int. Skipping row.")
                      s_idx, e_idx = None, None

                 if s_idx is not None and e_idx is not None:
                      self.sub_df_info.append({'filename': fn, 'start': s_idx, 'end': e_idx})
                 else:
                      print(f"Warning: Invalid start/end frame indices ({s_idx}, {e_idx}) for {fn}. Excluding this item.")


            # Update a4c_fn based on successfully processed rows
            self.a4c_fn = [item['filename'] for item in self.sub_df_info]


    def __getitem__(self, index):
        if index >= len(self.sub_df_info):
            raise IndexError("Index out of bounds")

        item_info = self.sub_df_info[index]
        fn = item_info['filename']
        # Adjust start index (often 1-based in excel) and ensure they are int
        s_idx = int(item_info['start']) - 1 # Make 0-based
        e_idx = int(item_info['end'])      # End index is exclusive in slicing

        # --- Cache File Paths ---
        # Use a consistent naming scheme, e.g., based on the original filename
        base_cache_fn = f"{fn}_fs{self.frame_size}" # Include frame size in name
        cached_video_path = os.path.join(self.cache_dir, f"{base_cache_fn}_video.pt")
        cached_mask_path = os.path.join(self.cache_dir, f"{base_cache_fn}_mask.pt")
        # --- End Cache File Paths ---

        video_tensor = None
        mask_tensor = None

        # --- Try Loading from Cache ---
        if self.use_cache:
            try:
                if os.path.exists(cached_video_path) and os.path.exists(cached_mask_path):
                    video_tensor = torch.load(cached_video_path)
                    mask_tensor = torch.load(cached_mask_path)
                    # print(f"Cache hit for index {index} ({fn})") # Optional: for debugging
                    return video_tensor, mask_tensor
            except Exception as e:
                print(f"Warning: Error loading cache for index {index} ({fn}): {e}. Will re-process.")
        # --- End Try Loading from Cache ---

        # --- Process if not loaded from cache ---
        # print(f"Cache miss for index {index} ({fn}). Processing...") # Optional: for debugging
        v_fn_pth = os.path.join(self.vid_path, fn + '.avi')
        s_fn_pth = os.path.join(self.seg_path, 'Mask_' + fn + '.mat')

        video_np = self.readVid(v_fn_pth, s_idx, e_idx, self.frame_size)
        mask_np = self.readMat(s_fn_pth, self.frame_size) # Returns float32 numpy array

        # Convert to tensors BEFORE saving (if conversion is needed)
        if video_np is not None:
            # Ensure correct dtype, float32 is common for input images
            video_tensor = torch.from_numpy(video_np).float()
        else:
             print(f"Warning: readVid returned None for {fn}. Skipping this item.")
             # Return None or placeholder tensors depending on downstream collation needs
             # Returning None, None might cause issues in standard DataLoader collation
             # Consider returning zero tensors of expected shape or raising an error
             return None, None # Or handle appropriately

        if mask_np is not None:
             # Mask dtype depends on loss function.
             # For CrossEntropyLoss typically LongTensor (int64)
             # For BCE/DiceLoss typically FloatTensor (float32)
             # Assuming FloatTensor based on readMat returning float32
             mask_tensor = torch.from_numpy(mask_np).float()
             # If you use CrossEntropyLoss, you might need:
             # mask_tensor = torch.from_numpy(mask_np).long()
        else:
             print(f"Warning: readMat returned None for {fn}. Skipping this item.")
             return None, None # Or handle appropriately

        # --- Save to Cache ---
        if self.use_cache and video_tensor is not None and mask_tensor is not None:
            try:
                torch.save(video_tensor, cached_video_path)
                torch.save(mask_tensor, cached_mask_path)
            except Exception as e:
                print(f"Warning: Failed to save cache for index {index} ({fn}): {e}")
        # --- End Save to Cache ---

        return video_tensor, mask_tensor


    def __len__(self):
        return len(self.sub_df_info) # Length is based on valid items found


    def readVid(self, filename, start_idx, end_idx, frame_size):
        # (Copied from previous context, added error handling and release)
        # Ensure it returns a numpy array (C, F, H, W) or None
        frame_arr = []
        vid = cv.VideoCapture(filename)
        if not vid.isOpened():
            print(f"Error opening video file: {filename}")
            return None

        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
        frame_w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Validate indices against actual frame count
        if start_idx >= frame_count or start_idx < 0 or end_idx <= start_idx :
             print(f"Warning: Invalid frame indices [{start_idx}, {end_idx}) for video {filename} with {frame_count} frames.")
             # Decide how to handle: return None, return empty, or try to read all? Let's return None for invalid range.
             vid.release()
             return None
        # Ensure end_idx doesn't exceed frame count (though the loop handles this)
        effective_end_idx = min(end_idx, frame_count)


        do_resize = False
        if frame_w != frame_size or frame_h != frame_size:
            do_resize = True

        valid_frames_read = 0
        # Set frame position (more efficient than reading all frames)
        # Note: CAP_PROP_POS_FRAMES is not always precise, but often better
        vid.set(cv.CAP_PROP_POS_FRAMES, start_idx)

        for idx in range(start_idx, effective_end_idx):
            ret, frame = vid.read()
            if not ret:
                # This might happen if end_idx is beyond actual frames or read error
                print(f"Warning: Failed to read frame {idx} from {filename}. Stopping read for this video.")
                break

            # Convert to grayscale BEFORE potentially resizing
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if do_resize:
                 # Use INTER_AREA for shrinking, INTER_LINEAR/CUBIC for enlarging generally
                 interpolation = cv.INTER_AREA if frame.shape[0] > frame_size else cv.INTER_LINEAR
                 frame = cv.resize(frame, dsize=(frame_size, frame_size), interpolation=interpolation)

            frame_arr.append(frame)
            valid_frames_read += 1

        vid.release()

        if valid_frames_read == 0:
            print(f"Warning: No frames read for {filename} in range [{start_idx}, {effective_end_idx}).")
            return None

        # Stack frames: Resulting shape (F, H, W)
        video_np = np.stack(frame_arr, axis=0)

        # Add channel dim: (F, H, W, C) with C=1
        video_np = np.expand_dims(video_np, axis=-1)

        # Transpose to (C, F, H, W) and ensure float32
        return video_np.transpose(3, 0, 1, 2).astype(np.float32)


    def readMat(self, filename, frame_size):
        # (Copied from previous context, added error handling)
        # Returns numpy array (F, H, W) with float32 dtype or None
        try:
            mat_data = scipy.io.loadmat(filename)
            if 'predicted' not in mat_data:
                # Try common alternative keys if 'predicted' isn't found
                potential_keys = [k for k in mat_data if not k.startswith('__')]
                if len(potential_keys) == 1:
                    key_to_use = potential_keys[0]
                    print(f"Info: Using key '{key_to_use}' instead of 'predicted' in {filename}")
                    mat = mat_data[key_to_use]
                else:
                    raise KeyError(f"'predicted' key not found and ambiguous other keys in {filename}: {potential_keys}")
            else:
                mat = mat_data['predicted']

            # Check if mat is empty or has unexpected dimensions before proceeding
            if not isinstance(mat, np.ndarray) or mat.ndim != 3:
                 raise ValueError(f"Unexpected data type or dimensions ({mat.shape if isinstance(mat, np.ndarray) else type(mat)}) in MAT file {filename}")

        except FileNotFoundError:
            print(f"Error: Mask file not found: {filename}")
            return None
        except KeyError as e:
            print(f"Error: Key not found in MAT file {filename}: {e}")
            return None
        except ValueError as e:
             print(f"Error: Problem with MAT file content {filename}: {e}")
             return None
        except Exception as e:
            print(f"Error loading MAT file {filename}: {e}")
            return None

        f, h, w = mat.shape

        if h == frame_size and w == frame_size:
            # No resize needed, just ensure dtype
            return mat.astype(np.float32)
        else:
            # Resize each frame
            frame_arr = []
            for i in range(f):
                # Ensure frame is 2D before resizing
                frame_slice = mat[i]
                if frame_slice.ndim != 2:
                     print(f"Warning: Unexpected dimensions for frame {i} in {filename}: {frame_slice.shape}. Skipping frame.")
                     continue

                # Use INTER_NEAREST for masks to avoid introducing new values
                resized_frame = cv.resize(frame_slice.astype(np.uint8), # Resize uint8
                                          dsize=(frame_size, frame_size),
                                          interpolation=cv.INTER_NEAREST)
                frame_arr.append(resized_frame)

            if not frame_arr:
                 print(f"Warning: No frames could be resized for {filename}")
                 return None

            # Stack and ensure float32 dtype
            mat_resized = np.stack(frame_arr, axis=0).astype(np.float32)
            return mat_resized