import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io
import torchvision
import os


import torch
from torch.utils.data import DataLoader, random_split

# --- Assume HMCDataset class definition is available here ---
# (Using the definition from the previous example)
import os
import pandas as pd
import numpy as np
import cv2 as cv
import scipy.io
import torchvision

# Dummy class definition provided by the user (copied from previous context)
class HMCDataset(torchvision.datasets.VisionDataset):
    # ... (rest of the HMCDataset class definition from the previous answer) ...
    # Make sure the full class definition is included here or imported
    def __init__(self, root=None, axis='A4C', frame_size=224):
        super().__init__(root)
        self.root = root
        self.axis = axis.upper()
        self.frame_size = frame_size
        if self.root is None: self.root = os.getcwd()

        self.output_path = os.path.join(self.root, 'model_weights')

        self.seg_path = os.path.join(self.root, 'LV Ground-truth Segmentation Masks')
        self.vid_path = os.path.join(self.root, 'HMC-QU', self.axis)
        excel_path = os.path.join(self.root, 'A4C.xlsx')
        if not os.path.exists(excel_path): raise FileNotFoundError(f"Error: A4C.xlsx not found at {excel_path}.")
        self.df = pd.read_excel(excel_path)
        if self.df.shape[1] < 4: raise IndexError("Error: A4C.xlsx does not have enough columns.")
        mask_availability_col = self.df.iloc[:, -1]
        valid_mask_indices = mask_availability_col == 'ü'
        if not valid_mask_indices.any():
            print(f"Warning: No entries with 'ü' found in the last column ('{mask_availability_col.name}') of A4C.xlsx. Dataset will be empty.")
            self.sub_df = np.empty((0, 4)); self.a4c_fn = []
        else:
             # Selecting columns: 0 (ECHO), -3 (Start?), -2 (End?), -1 (Mask indicator)
             # Ensure these column indices make sense for your actual A4C.xlsx structure
             self.sub_df = self.df.loc[valid_mask_indices].iloc[:, [0, -3, -2, -1]].to_numpy()
             self.a4c_fn = list(self.df.loc[valid_mask_indices]['ECHO']) # Ensure 'ECHO' is the correct column name (index 0)

    def __getitem__(self, index):
        if index >= len(self.a4c_fn): raise IndexError("Index out of bounds")
        # Cast indices to int before using them
        s_idx, e_idx = self.sub_df[index, 1:3].astype(int)
        s_idx = s_idx - 1
        fn = self.a4c_fn[index]
        v_fn_pth = os.path.join(self.vid_path, fn + '.avi')
        s_fn_pth = os.path.join(self.seg_path, 'Mask_' + fn + '.mat')

        video = self.readVid(v_fn_pth, s_idx, e_idx, self.frame_size)
        seg_frames = self.readMat(s_fn_pth, self.frame_size)
        # Ensure return types are suitable for collation (e.g., torch tensors)
        # Convert numpy arrays to torch tensors
        if video is not None: video = torch.from_numpy(video)
        if seg_frames is not None: seg_frames = torch.from_numpy(seg_frames)

        return video, seg_frames # Should return torch tensors

    def __len__(self): return len(self.a4c_fn)

    def readVid(self, filename, start_idx, end_idx, frame_size):
        # ... (readVid implementation - ensure it returns numpy array or None) ...
        # (Copied from previous context, added error handling and release)
        frame_arr = []
        vid = cv.VideoCapture(filename)
        if not vid.isOpened(): print(f"Error opening video file: {filename}"); return None
        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT)); frame_w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH)); frame_h = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        do_resize = False
        if frame_w != frame_size or frame_h != frame_size: do_resize = True
        valid_frames_read = 0
        for idx in range(frame_count):
            ret, frame = vid.read()
            if not ret: break
            if start_idx <= idx < end_idx:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if do_resize: frame = cv.resize(frame, dsize=(frame_size, frame_size))
                frame_arr.append(frame)
                valid_frames_read += 1
        vid.release()
        if valid_frames_read == 0: print(f"Warning: No frames read for {filename} in range [{start_idx}, {end_idx})"); return None
        video = np.array(frame_arr, dtype=np.float32); video = np.expand_dims(video, -1) # (F,H,W,C)
        return video.transpose(3, 0, 1, 2) # (C,F,H,W)


    def readMat(self, filename, frame_size):
        # ... (readMat implementation - ensure it returns numpy array or None) ...
        # (Copied from previous context, added error handling)
        try:
             mat_data = scipy.io.loadmat(filename)
             if 'predicted' not in mat_data: raise KeyError(f"'predicted' key not found in {filename}")
             mat = mat_data['predicted']
        except FileNotFoundError: print(f"Error: Mask file not found: {filename}"); return None
        except Exception as e: print(f"Error loading MAT file {filename}: {e}"); return None
        f, h, w = mat.shape
        if h != frame_size or w != frame_size:
            frame_arr = []
            for i in range(f):
                resized_frame = cv.resize(mat[i].astype(np.uint8), dsize=(frame_size, frame_size), interpolation=cv.INTER_NEAREST)
                frame_arr.append(resized_frame)
            mat = np.array(frame_arr, dtype=np.float32)
        # Ensure mask is returned with appropriate dtype (often long for segmentation tasks if using CrossEntropyLoss)
        # Or keep as float if using BCE/Dice loss. Let's keep float32 for now.
        return mat.astype(np.float32)
