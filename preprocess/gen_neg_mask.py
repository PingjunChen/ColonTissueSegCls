# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import shutil, uuid
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import pydaily


def gen_neg_mask(ori_neg_dir, save_neg_dir):
    img_list = pydaily.filesystem.find_ext_files(ori_neg_dir, "jpg")

    for ind, ele in enumerate(img_list):
        ele = os.path.basename(ele)
        if ind > 0 and ind % 10 == 0:
            print("Processing {}/{}".format(ind, len(img_list)))
        cur_img_path = os.path.join(ori_neg_dir, ele)
        save_name = str(uuid.uuid4())[:8]
        shutil.copy(cur_img_path, os.path.join(save_neg_dir, save_name+".jpg"))
        cur_img = io.imread(cur_img_path)
        mask_arr = np.zeros(cur_img.shape[:2], dtype=np.uint8)
        save_mask_path = os.path.join(save_neg_dir, save_name+".png")
        io.imsave(save_mask_path, mask_arr)

if __name__ == "__main__":
    np.random.seed(1234)
    ori_neg_dir = "../data/tissue-train-neg"
    save_neg_dir = "../data/NegImgs"
    gen_neg_mask(ori_neg_dir, save_neg_dir)
