# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class SegPatchDataset(Dataset):
    """
    Dataset for slide testing. Each would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, patch_arr, mask_arr=None):
        self.patches = patch_arr
        self.masks = mask_arr
        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx,...]
        if self.transform:
            patch = self.transform(patch)

        if isinstance(self.masks, np.ndarray):
            mask = np.expand_dims(self.masks[idx,...], axis=0)
            return patch, mask
        else:
            return patch


class ClsPatchDataset(Dataset):
    """
    Dataset for slide testing. Each would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, patch_arr, model_input_size=224):
        self.patches = patch_arr
        self.rgb_mean = (0.700, 0.520, 0.720)
        self.rgb_std = (0.170, 0.200, 0.128)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(model_input_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(self.rgb_mean, self.rgb_std)])
    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx,...]
        if self.transform:
            patch = self.transform(patch)

        return patch
