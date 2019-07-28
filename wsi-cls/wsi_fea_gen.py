# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from pyslide import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from patch_loader import PatchDataset, wsi_stride_splitting

def gen_wsi_feas(img_dir, fea_dir, patch_len=448, stride_len=224):
    img_list = [ele for ele in os.listdir(img_dir) if "jpg" in ele]

    min_num, max_num = 100000, 0
    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(img_list)))
        img_path = os.path.join(img_dir, ele)
        cur_img = io.imread(img_path)
        # split coors and save patches
        coors_arr = wsi_stride_splitting(cur_img.shape[0], cur_img.shape[1], patch_len, stride_len)
        patch_list, coor_list = [], []
        for coor in coors_arr:
            start_h, start_w = coor[0], coor[1]
            patch_img = cur_img[start_h:start_h+patch_len, start_w:start_w+patch_len]
            # image background control
            if patch.patch_bk_ratio(patch_img, bk_thresh=0.864) > 0.88:
                continue
            patch_list.append(patch_img)
            coor_list.append([start_h, start_w, start_h+patch_len, start_w+patch_len])
        # For patch feature generation
        patch_arr = np.asarray(patch_list)
        patch_dset = PatchDataset(patch_arr)
        patch_loader = DataLoader(patch_dset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)



if __name__ == "__main__":
    img_dir = "../data/SlideCLS/SlideImgs/tissue-train-pos/train"
    fea_dir = "../data/SlideCLS/SlideFea/train/1Pos"
    gen_wsi_feas(img_dir, fea_dir, patch_len=448, stride_len=224)
