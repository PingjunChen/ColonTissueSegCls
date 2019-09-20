# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import pydaily
from pyslide import patch


def locate_patch(slide_dir, patch_size=448):
    # overwrite directory
    mask_dir = os.path.join(os.path.dirname(slide_dir), os.path.basename(slide_dir)+"_mask")
    pydaily.filesystem.overwrite_dir(mask_dir)
    # traversing files
    slide_list = pydaily.filesystem.find_ext_files(slide_dir, "jpg")
    for ind, ele in enumerate(slide_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(slide_list)))
        mask_path = os.path.join(mask_dir, os.path.basename(ele))
        cur_slide = io.imread(ele)
        cur_mask = np.zeros((cur_slide.shape[0], cur_slide.shape[1]), dtype=np.uint8)
        coors_arr = patch.wsi_coor_splitting(cur_slide.shape[0], cur_slide.shape[1], patch_size, overlap_flag=True)
        for coor in coors_arr:
            start_h, start_w = coor[0], coor[1]
            patch_img = cur_slide[start_h:start_h+patch_size, start_w:start_w+patch_size]
            if patch.patch_bk_ratio(patch_img, bk_thresh=0.80) < 0.84:
                cur_mask[start_h:start_h+patch_size, start_w:start_w+patch_size] = 255
        io.imsave(mask_path, cur_mask)


if __name__ == "__main__":
    slide_dir = "../data/Split1235/Pos/train"
    locate_patch(slide_dir)
