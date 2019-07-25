# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import shutil, uuid
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import pydaily


def gen_neg_mask(neg_dir):
    img_list = pydaily.filesystem.find_ext_files(neg_dir, "jpg")

    for ind, ele in enumerate(img_list):
        ele = os.path.basename(ele)
        if ind > 0 and ind % 10 == 0:
            print("Processing {}/{}".format(ind, len(img_list)))
        cur_img_path = os.path.join(neg_dir, ele)
        save_name = str(uuid.uuid4())[:8]
        cur_img = io.imread(cur_img_path)
        mask_arr = np.zeros(cur_img.shape[:2], dtype=np.uint8)
        os.rename(cur_img_path, os.path.join(neg_dir, save_name+".jpg"))
        mask_path = os.path.join(neg_dir, save_name+".png")
        io.imsave(mask_path, mask_arr)

if __name__ == "__main__":
    np.random.seed(1234)
    neg_dir = "../data/tissue-train-neg"
    gen_neg_mask(neg_dir)
