# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import shutil, uuid
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import pydaily


def adjust_pos(pos_dir):
    filelist = pydaily.filesystem.find_ext_files(pos_dir, "jpg")
    img_list = [os.path.basename(ele) for ele in filelist if "mask" not in ele]
    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("Processing {}/{}".format(ind, len(img_list)))
        cur_img_path = os.path.join(pos_dir, ele)
        save_name = str(uuid.uuid4())[:8]
        os.rename(cur_img_path, os.path.join(pos_dir, save_name+".jpg"))
        mask_path = os.path.join(pos_dir, os.path.splitext(ele)[0] + "_mask.jpg")
        mask_arr = ((io.imread(mask_path) > 128) * 255).astype(np.uint8)
        mask_path = os.path.join(pos_dir, save_name+".png")
        io.imsave(mask_path, mask_arr)

if __name__ == "__main__":
    np.random.seed(1234)
    pos_dir = "../data/tissue-train-pos"
    adjust_pos(pos_dir)
