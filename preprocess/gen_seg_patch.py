# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import shutil, uuid
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import pydaily
from pyslide import patch


def gen_patches(img_dir, patch_dir, patch_size=448):
    img_list = pydaily.filesystem.find_ext_files(img_dir, "jpg")
    img_list = [os.path.basename(ele) for ele in img_list]
    patch_img_dir = os.path.join(patch_dir, "imgs")
    if not os.path.exists(patch_img_dir):
        os.makedirs(patch_img_dir)
    patch_mask_dir = os.path.join(patch_dir, "masks")
    if not os.path.exists(patch_mask_dir):
        os.makedirs(patch_mask_dir)

    pos_num, neg_num = 0, 0
    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(img_list)))
        img_path = os.path.join(img_dir, ele)
        mask_path = os.path.join(img_dir, os.path.splitext(ele)[0]+".png")
        cur_img = io.imread(img_path)
        cur_mask = io.imread(mask_path)
        # split coors and save patches
        coors_arr = patch.wsi_coor_splitting(cur_img.shape[0], cur_img.shape[1], patch_size, overlap_flag=True)
        for coor in coors_arr:
            start_h, start_w = coor[0], coor[1]
            patch_img = cur_img[start_h:start_h+patch_size, start_w:start_w+patch_size]
            # image background control
            if patch.patch_bk_ratio(patch_img, bk_thresh=0.864) > 0.88:
                continue
            # mask control
            patch_mask = cur_mask[start_h:start_h+patch_size, start_w:start_w+patch_size]
            pixel_ratio = np.sum(patch_mask > 0) * 1.0 / patch_mask.size
            save_flag = True
            if pixel_ratio >= 0.05:
                pos_num += 1
            else:
                if "Neg" in img_dir:
                    if np.random.random_sample() > 0.92:
                        neg_num += 1
                    else:
                        save_flag = False
                else:
                    neg_num += 1
            if save_flag == True:
                patch_name = str(uuid.uuid4())[:8]
                io.imsave(os.path.join(patch_img_dir, patch_name+".jpg"), patch_img)
                io.imsave(os.path.join(patch_mask_dir, patch_name+".png"), patch_mask)
    print("There are {} pos samples and {} neg samples".format(pos_num, neg_num))


if __name__ == "__main__":
    img_dir = "/media/pingjun/CPATHOLOGY/Segmentation/ColonData/Split1235/Pos/val"
    patch_dir = "/media/pingjun/CPATHOLOGY/Segmentation/ColonData/Split1235/SegPatches/val"
    gen_patches(img_dir, patch_dir, patch_size=448)
