# -*- coding: utf-8 -*-

import os, sys
import shutil, random, uuid
import numpy as np
import pydaily

def split_train_val(img_dir, ratio):
    img_list = pydaily.filesystem.find_ext_files(img_dir, "jpg")
    img_list = [os.path.basename(ele) for ele in img_list]
    random.shuffle(img_list)
    # copy to train and val
    train_num = int(len(img_list) * (1 - ratio))
    train_dir = os.path.join(img_dir, "train")
    pydaily.filesystem.overwrite_dir(train_dir)
    train_list = img_list[:train_num]
    for ele in train_list:
        img_path = os.path.join(img_dir, ele)
        mask_path = os.path.join(img_dir, os.path.splitext(ele)[0]+".png")
        shutil.move(img_path, train_dir)
        shutil.move(mask_path, train_dir)

    val_dir = os.path.join(img_dir, "val")
    pydaily.filesystem.overwrite_dir(val_dir)
    val_list = img_list[train_num:]
    for ele in val_list:
        img_path = os.path.join(img_dir, ele)
        mask_path = os.path.join(img_dir, os.path.splitext(ele)[0]+".png")
        shutil.move(img_path, val_dir)
        shutil.move(mask_path, val_dir)


if __name__ == "__main__":
    np.random.seed(1238)
    # split pos to train/val, remove *mask.jpg first
    img_dir = "../data/tissue-train-pos"
    split_train_val(img_dir, ratio=0.2)

    # split neg to train/val
    img_dir = "../data/tissue-train-neg"
    split_train_val(img_dir, ratio=0.2)
