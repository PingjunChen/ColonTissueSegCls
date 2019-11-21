# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import pydaily


def gen_mal_overlay(pos_dir):
    filelist = pydaily.filesystem.find_ext_files(pos_dir, "jpg")
    img_list = [os.path.basename(ele) for ele in filelist]
    alp = 0.6
    for ind, ele in enumerate(img_list):
        img_path = os.path.join(pos_dir, ele)
        mask_path = os.path.join(pos_dir, os.path.splitext(ele)[0]+".png")
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        cmap = plt.get_cmap('jet')
        heat_map = cmap(mask)[:, :, :-1]
        alpha_img = (img * alp + heat_map * 255.0 * (1 - alp)).astype(np.uint8)
        overlay_path = os.path.join(pos_dir, os.path.splitext(ele)[0]+"_mask.png")
        io.imsave(overlay_path, alpha_img)


if __name__ == "__main__":
    np.random.seed(1234)
    pos_dir = "../data/tissue-train-pos"

    gen_mal_overlay(pos_dir)
