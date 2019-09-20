# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import torch

import pydaily
from pyslide import patch


def save_mil_input(slide_dir, save_path, patch_size):
    slide_dict = {}
    slide_dict["mult"] = 2
    slide_dict["level"] = 0
    slide_dict["patch_size"] = patch_size

    grid_list, target_list = [], []

    slide_list = pydaily.filesystem.find_ext_files(slide_dir, "jpg")
    for ind, ele in enumerate(slide_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(slide_list)))
        cur_slide = io.imread(ele)
        coors_arr = patch.wsi_coor_splitting(cur_slide.shape[0], cur_slide.shape[1], patch_size, overlap_flag=True)
        grids = []
        for coor in coors_arr:
            start_h, start_w = coor[0], coor[1]
            patch_img = cur_slide[start_h:start_h+patch_size, start_w:start_w+patch_size]
            if patch.patch_bk_ratio(patch_img, bk_thresh=0.80) < 0.84:
                grids.append([start_h, start_w])
        if len(grids) == 0:
            raise AssertionError("{} has no grid.".format(ele))
        grid_list.append(grids)
        label_str = os.path.basename(os.path.dirname(ele))
        if label_str == "Pos":
            target_list.append(1)
        elif label_str == "Neg":
            target_list.append(0)
        else:
            raise AssertionError("{} not recognizable.".format(ele))

    slide_dict["grid"] = grid_list
    slide_dict["targets"] = target_list
    slide_dict["slides"] = slide_list
    torch.save(slide_dict, save_path)



if __name__ == "__main__":
    slide_dir = "../data/MIL/Split1234/Train"
    save_path = "../data/MIL/Split1234/train_inputs.pkl"
    patch_size = 448
    save_mil_input(slide_dir, save_path, patch_size)
