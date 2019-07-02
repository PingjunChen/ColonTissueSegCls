# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import shutil, uuid
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import pydaily


def adjust_pos(ori_pos_dir, save_pos_dir):
    filelist = pydaily.filesystem.find_ext_files(ori_pos_dir, "jpg")
    img_list = [os.path.basename(ele) for ele in filelist if "mask" not in ele]
    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("Processing {}/{}".format(ind, len(img_list)))
        cur_img_path = os.path.join(ori_pos_dir, ele)
        save_name = str(uuid.uuid4())[:8]
        shutil.copy(cur_img_path, os.path.join(save_pos_dir, save_name+".jpg"))
        mask_path = os.path.join(ori_pos_dir, os.path.splitext(ele)[0] + "_mask.jpg")
        mask_arr = ((io.imread(mask_path) > 128) * 255).astype(np.uint8)
        save_mask_path = os.path.join(save_pos_dir, save_name+".png")
        io.imsave(save_mask_path, mask_arr)

if __name__ == "__main__":
    np.random.seed(1234)
    ori_pos_dir = "../data/tissue-train-pos"
    save_pos_dir = "../data/PosImgs"
    adjust_pos(ori_pos_dir, save_pos_dir)
