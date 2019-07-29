# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import itertools
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class PatchDataset(Dataset):
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


def wsi_stride_splitting(wsi_h, wsi_w, patch_len, stride_len):
    """ Spltting whole slide image to patches by stride.
    Parameters
    -------
    wsi_h: int
        height of whole slide image
    wsi_w: int
        width of whole slide image
    patch_len: int
        length of the patch image
    stride_len: int
        length of the stride
    Returns
    -------
    coors_arr: list
        list of starting coordinates of patches ([0]-h, [1]-w)
    """

    coors_arr = []
    def stride_split(ttl_len, patch_len, stride_len):
        p_sets = []
        if patch_len > ttl_len:
            raise AssertionError("patch length larger than total length")
        elif patch_len == ttl_len:
            p_sets.append(0)
        else:
            stride_num = int(np.ceil((ttl_len - patch_len) * 1.0 / stride_len))
            for ind in range(stride_num+1):
                cur_pos = int(((ttl_len - patch_len) * 1.0 / stride_num) * ind)
                p_sets.append(cur_pos)

        return p_sets

    h_sets = stride_split(wsi_h, patch_len, stride_len)
    w_sets = stride_split(wsi_w, patch_len, stride_len)

    # combine points in both w and h direction
    if len(w_sets) > 0 and len(h_sets) > 0:
        coors_arr = list(itertools.product(h_sets, w_sets))

    return coors_arr
