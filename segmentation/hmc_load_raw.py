# hmc_load.py
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

        # self.output_path = os.path.join(self.root, 'model_weights') # This seems unrelated to dataset loading itself

        self.seg_path = os.path.join(self.root, 'LV Ground-truth Segmentation Masks')
        self.vid_path = os.path.join(self.root, 'HMC-QU', self.axis)
        excel_path = os.path.join(self.root, f'{axis}.xlsx')

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Error: {axis}.xlsx not found at {excel_path}.")

        try:
            self.df = pd.read_excel(excel_path)
            # print(self.df) # Keep for debugging if needed
        except Exception as e:
            raise IOError(f"Error reading {axis}.xlsx: {e}")

        # --- Robust Column Handling ---
        if 'ECHO' not in self.df.columns:
             raise ValueError(f"Error: 'ECHO' column not found in {axis}.xlsx.")
        if self.df.shape[1] < 4:
             raise IndexError(f"Error: {axis}.xlsx does not have enough columns (expected at least 4: ECHO, Start, End, MaskAvailability).")

        echo_col_name = 'ECHO'
        mask_availability_col_name = self.df.columns[-1]
        end_frame_col_name = self.df.columns[-2]
        start_frame_col_name = self.df.columns[-3]
        # --- End Robust Column Handling ---

        self.sub_df_info = [] # Store relevant info as a list of dicts
        self.a4c_fn = [] # Keep track of successfully parsed filenames

        # --- Process ALL rows, tracking mask availability ---
        print(f"Processing all entries in {axis}.xlsx...")
        processed_count = 0
        skipped_count = 0
        mask_available_count = 0

        for index, row in self.df.iterrows():
             fn = str(row[echo_col_name]) # Ensure filename is string
             s_idx = row.get(start_frame_col_name)
             e_idx = row.get(end_frame_col_name)
             mask_available_marker = str(row.get(mask_availability_col_name, '')).strip() # Get mask status

             # Determine if mask is expected
             has_mask = (mask_available_marker == 'ü')

             # --- Handle potential NaN or non-numeric values for indices ---
             try:
                  s_idx = int(s_idx) if pd.notna(s_idx) else None
                  e_idx = int(e_idx) if pd.notna(e_idx) else None
             except (ValueError, TypeError):
                  print(f"Warning: Could not convert start/end indices for {fn} (Row {index}) to int. Skipping row.")
                  s_idx, e_idx = None, None
                  skipped_count += 1

             if s_idx is not None and e_idx is not None:
                  self.sub_df_info.append({
                      'filename': fn,
                      'start': s_idx,
                      'end': e_idx,
                      'has_mask': has_mask # Store mask availability
                  })
                  self.a4c_fn.append(fn)
                  processed_count += 1
                  if has_mask:
                      mask_available_count += 1
             else:
                  print(f"Warning: Invalid start/end frame indices ({s_idx}, {e_idx}) for {fn} (Row {index}). Excluding this item.")
                  if skipped_count == 0: # Only increment if not already skipped for conversion error
                      skipped_count += 1

        print(f"Finished processing Excel. Total rows: {len(self.df)}. Usable entries: {processed_count} ({mask_available_count} with masks). Skipped: {skipped_count}.")
        if processed_count == 0:
            print("Warning: No usable entries found after processing the Excel file. Dataset will be empty.")
        # --- End Row Processing ---


    def __getitem__(self, index):
        if index >= len(self.sub_df_info):
            raise IndexError("Index out of bounds")

        item_info = self.sub_df_info[index]
        fn = item_info['filename']
        has_mask = item_info['has_mask']
        # Adjust start index (often 1-based in excel) and ensure they are int
        s_idx = int(item_info['start']) - 1 # Make 0-based
        e_idx = int(item_info['end'])      # End index is exclusive in slicing

        # --- Cache File Paths ---
        base_cache_fn = f"{fn}_fs{self.frame_size}" # Include frame size in name
        cached_video_path = os.path.join(self.cache_dir, f"{base_cache_fn}_video.pt")
        cached_mask_path = os.path.join(self.cache_dir, f"{base_cache_fn}_mask.pt") # Path always defined, but only used if has_mask
        # --- End Cache File Paths ---

        video_tensor = None
        mask_tensor = None # Default to None

        # --- Try Loading from Cache ---
        cache_hit = False
        if self.use_cache:
            try:
                video_cache_exists = os.path.exists(cached_video_path)
                mask_cache_exists = os.path.exists(cached_mask_path)

                # Load video if cache exists
                if video_cache_exists:
                    video_tensor = torch.load(cached_video_path)

                    # Load mask ONLY if it's expected AND its cache exists
                    if has_mask and mask_cache_exists:
                        mask_tensor = torch.load(cached_mask_path)
                        cache_hit = True # Full hit (video + expected mask)
                    elif not has_mask:
                        cache_hit = True # Partial hit (video only, mask not expected)
                    # If mask is expected but cache doesn't exist, it's not a full hit

                if cache_hit:
                    # print(f"Cache hit for index {index} ({fn}). Mask expected: {has_mask}") # Debug
                    return video_tensor, mask_tensor

            except Exception as e:
                print(f"Warning: Error loading cache for index {index} ({fn}): {e}. Will re-process.")
                video_tensor = None # Reset state in case of partial load error
                mask_tensor = None
                cache_hit = False # Ensure we re-process
        # --- End Try Loading from Cache ---

        # --- Process if not fully loaded from cache ---
        if not cache_hit:
            # print(f"Processing index {index} ({fn}). Mask expected: {has_mask}") # Debug

            # --- Load Video ---
            v_fn_pth = os.path.join(self.vid_path, fn + '.avi')
            video_np = self.readVid(v_fn_pth, s_idx, e_idx, self.frame_size)

            if video_np is not None:
                # Ensure correct dtype, float32 is common for input images
                video_tensor = torch.from_numpy(video_np).float()
            else:
                 print(f"Warning: readVid returned None for {fn}. Skipping this item.")
                 # Return None, None directly as we cannot proceed
                 return None, None

            # --- Load Mask (only if expected) ---
            if has_mask:
                s_fn_pth = os.path.join(self.seg_path, 'Mask_' + fn + '.mat')
                mask_np = self.readMat(s_fn_pth, self.frame_size) # Returns float32 numpy array or None

                if mask_np is not None:
                    # Convert to tensor (adjust dtype based on need, default FloatTensor)
                    mask_tensor = torch.from_numpy(mask_np).float()
                    # If you use CrossEntropyLoss, you might need:
                    # mask_tensor = torch.from_numpy(mask_np).long()
                else:
                    # Mask was expected but failed to load. video_tensor is still valid.
                    print(f"Warning: readMat returned None for expected mask {fn}. Proceeding without mask for this item.")
                    # mask_tensor remains None
            # else: mask_tensor remains None (as initialized)

            # --- Save to Cache ---
            if self.use_cache and video_tensor is not None: # Always save video if successful
                try:
                    torch.save(video_tensor, cached_video_path)
                    # Only save mask if it was expected AND successfully loaded
                    if has_mask and mask_tensor is not None:
                         torch.save(mask_tensor, cached_mask_path)
                except Exception as e:
                    print(f"Warning: Failed to save cache for index {index} ({fn}): {e}")
            # --- End Save to Cache ---

        return video_tensor, mask_tensor


    def __len__(self):
        return len(self.sub_df_info) # Length is based on valid items found


    def readVid(self, filename, start_idx, end_idx, frame_size):
        # (No changes needed here, keeping previous robust version)
        frame_arr = []
        vid = cv.VideoCapture(filename)
        if not vid.isOpened():
            print(f"Error opening video file: {filename}")
            return None

        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
        frame_w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Validate indices against actual frame count
        # Allow end_idx == start_idx for single frame videos, but end_idx must be > start_idx for slicing
        if start_idx >= frame_count or start_idx < 0 or end_idx <= start_idx:
             # Check edge case: maybe indices are 1-based and mean single frame?
             # If end_idx == start_idx+1 (after 0-basing start), it's one frame.
             if start_idx < frame_count and end_idx == start_idx + 1:
                 # This is okay, represents a single frame request
                 effective_end_idx = end_idx
             else:
                 print(f"Warning: Invalid frame indices [{start_idx}, {end_idx}) for video {filename} with {frame_count} frames.")
                 vid.release()
                 return None
        else:
            # Ensure end_idx doesn't exceed frame count
            effective_end_idx = min(end_idx, frame_count)


        do_resize = False
        if frame_w != frame_size or frame_h != frame_size:
            do_resize = True

        valid_frames_read = 0
        vid.set(cv.CAP_PROP_POS_FRAMES, start_idx) # Attempt to seek

        current_pos = int(vid.get(cv.CAP_PROP_POS_FRAMES))
        # If seek didn't work well, read frame by frame until start_idx
        while current_pos < start_idx:
            ret = vid.grab() # More efficient than read() if just skipping
            if not ret:
                print(f"Warning: Failed to reach start frame {start_idx} in {filename}.")
                vid.release()
                return None
            current_pos +=1


        for idx in range(start_idx, effective_end_idx):
            # Check if we are at the right frame position after potential seeking issues
            # Note: This check might be slightly off due to CAP_PROP_POS_FRAMES precision
            # current_frame_idx = int(vid.get(cv.CAP_PROP_POS_FRAMES))
            # if current_frame_idx != idx:
            #     print(f"Warning: Frame position mismatch in {filename}. Expected {idx}, got {current_frame_idx}. Attempting read anyway.")

            ret, frame = vid.read()
            if not ret:
                print(f"Warning: Failed to read frame {idx} (intended) from {filename}. Stopping read for this video.")
                break

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if do_resize:
                 interpolation = cv.INTER_AREA if frame.shape[0] > frame_size else cv.INTER_LINEAR
                 frame = cv.resize(frame, dsize=(frame_size, frame_size), interpolation=interpolation)

            frame_arr.append(frame)
            valid_frames_read += 1

        vid.release()

        if valid_frames_read == 0:
            print(f"Warning: No frames read for {filename} in range [{start_idx}, {effective_end_idx}).")
            return None

        video_np = np.stack(frame_arr, axis=0)
        video_np = np.expand_dims(video_np, axis=-1)
        return video_np.transpose(3, 0, 1, 2).astype(np.float32) # (C=1, F, H, W)


    def readMat(self, filename, frame_size):
        # (No changes needed here, keeping previous robust version)
        try:
            # Check if file exists before attempting to load
            if not os.path.exists(filename):
                 raise FileNotFoundError(f"Mask file not found: {filename}")

            mat_data = scipy.io.loadmat(filename)
            key_to_use = None
            if 'predicted' in mat_data:
                key_to_use = 'predicted'
            else:
                potential_keys = [k for k in mat_data if not k.startswith('__') and isinstance(mat_data[k], np.ndarray)]
                if len(potential_keys) == 1:
                    key_to_use = potential_keys[0]
                    # print(f"Info: Using key '{key_to_use}' instead of 'predicted' in {filename}") # Less verbose
                elif len(potential_keys) > 1:
                     # Heuristic: find key whose value is 3D array (F, H, W)
                     found_3d = False
                     for pk in potential_keys:
                         if mat_data[pk].ndim == 3:
                             key_to_use = pk
                             # print(f"Info: Using key '{key_to_use}' (found 3D array) instead of 'predicted' in {filename}")
                             found_3d = True
                             break
                     if not found_3d:
                         raise KeyError(f"'predicted' key not found and no clear 3D array key in {filename}: {potential_keys}")
                else:
                    raise KeyError(f"'predicted' key not found and no other array data keys in {filename}")

            mat = mat_data[key_to_use]

            if not isinstance(mat, np.ndarray) or mat.ndim != 3:
                 raise ValueError(f"Unexpected data type or dimensions ({mat.shape if isinstance(mat, np.ndarray) else type(mat)}) in MAT file {filename} using key '{key_to_use}'")

        except FileNotFoundError as e:
            print(f"Error: {e}") # More concise error
            return None
        except (KeyError, ValueError) as e:
            print(f"Error processing MAT file content {filename}: {e}")
            return None
        except Exception as e:
            print(f"Error loading MAT file {filename}: {e}")
            return None

        f, h, w = mat.shape

        if h == frame_size and w == frame_size:
            return mat.astype(np.float32) # (F, H, W)
        else:
            # print(f"Info: Resizing mask {filename} from ({h},{w}) to ({frame_size},{frame_size})") # Debug
            frame_arr = []
            for i in range(f):
                frame_slice = mat[i]
                if frame_slice.ndim != 2:
                     print(f"Warning: Unexpected dimensions for mask frame {i} in {filename}: {frame_slice.shape}. Skipping frame.")
                     continue

                # Use INTER_NEAREST for masks
                # Ensure input to resize is uint8 if needed by cv2, but store result based on original type interpretation
                # Convert to float for resize maybe better? No, stick to uint8 for nearest neighbor.
                # Output should match expected format (float32)
                temp_slice = frame_slice
                if not np.issubdtype(frame_slice.dtype, np.integer):
                    # If mask is float (e.g., 0.0, 1.0), convert to uint8 for resize
                    temp_slice = frame_slice.astype(np.uint8)

                resized_frame = cv.resize(temp_slice,
                                          dsize=(frame_size, frame_size),
                                          interpolation=cv.INTER_NEAREST)
                frame_arr.append(resized_frame) # Keep as resized type for now

            if not frame_arr:
                 print(f"Warning: No mask frames could be resized for {filename}")
                 return None

            mat_resized = np.stack(frame_arr, axis=0).astype(np.float32) # Convert final stack to float32
            return mat_resized # (F, H, W)