# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import itertools
import torch
import torch.nn.functional as F

def gen_patch_mask_wmap(slide_img, mask_img, coors_arr, plen):
    patch_list, mask_list = [], []
    wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        patch_list.append(slide_img[ph:ph+plen, pw:pw+plen] / 255.0)
        mask_list.append(mask_img[ph:ph+plen, pw:pw+plen])
        wmap[ph:ph+plen, pw:pw+plen] += 1
    patch_arr = np.asarray(patch_list).astype(np.float32)
    mask_arr = np.asarray(mask_list).astype(np.float32)

    return patch_arr, mask_arr, wmap


def gen_patch_wmap(slide_img, coors_arr, plen):
    patch_list = []
    wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        patch_list.append(slide_img[ph:ph+plen, pw:pw+plen] / 255.0)
        wmap[ph:ph+plen, pw:pw+plen] += 1
    patch_arr = np.asarray(patch_list).astype(np.float32)

    return patch_arr, wmap


def wsi_stride_splitting(wsi_h, wsi_w, patch_len, stride_len):
    """ Spltting whole slide image to patches by stride.
    Parameters
    -------
    wsi_h: int
        height of whole slide image
    wsi_w: int
        width of whole slide image
    patch_len: int
        length of the patch image
    stride_len: int
        length of the stride
    Returns
    -------
    coors_arr: list
        list of starting coordinates of patches ([0]-h, [1]-w)
    """

    coors_arr = []
    def stride_split(ttl_len, patch_len, stride_len):
        p_sets = []
        if patch_len > ttl_len:
            raise AssertionError("patch length larger than total length")
        elif patch_len == ttl_len:
            p_sets.append(0)
        else:
            stride_num = int(np.ceil((ttl_len - patch_len) * 1.0 / stride_len))
            for ind in range(stride_num+1):
                cur_pos = int(((ttl_len - patch_len) * 1.0 / stride_num) * ind)
                p_sets.append(cur_pos)

        return p_sets

    h_sets = stride_split(wsi_h, patch_len, stride_len)
    w_sets = stride_split(wsi_w, patch_len, stride_len)

    # combine points in both w and h direction
    if len(w_sets) > 0 and len(h_sets) > 0:
        coors_arr = list(itertools.product(h_sets, w_sets))

    return coors_arr
