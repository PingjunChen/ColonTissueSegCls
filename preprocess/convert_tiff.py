# -*- coding: utf-8 -*-

import os, sys
import uuid
import pydaily
import pyslide

def convert_jpg2tiff(input_dir, output_dir):
    pydaily.filesystem.overwrite_dir(output_dir)
    img_list = pydaily.filesystem.find_ext_files(input_dir, "jpg")
    for ind, cur_img in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {:3d}/{:3d}".format(ind+1, len(img_list)))
        convert_cmd = "convert " + cur_img
        convert_option = " -compress jpeg -quality 90 -define tiff:tile-geometry=256x256 ptif:"
        convert_dst = os.path.join(output_dir, os.path.basename(cur_img)[:-4] + ".tiff")
        status = os.system(convert_cmd + convert_option + convert_dst)
        assert status == 0, "conversion error..."

if __name__ == "__main__":
    input_dir = "../data/TIFF/Pos"
    output_dir = "../data/TIFF/PosTIFF"
    # pydaily.filesystem.batch_uuid_rename(input_dir, output_dir, ext=".jpg")
    convert_jpg2tiff(input_dir, output_dir)
