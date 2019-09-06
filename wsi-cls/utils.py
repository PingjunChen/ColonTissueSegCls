# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import filters, io
import matplotlib.pyplot as plt


def getfilelist(Imagefolder, inputext, with_ext=False):
    '''inputext: ['.json'] '''

    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    allfiles = sorted(os.listdir(Imagefolder))

    for f in allfiles:
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
            filelist.append(os.path.join(Imagefolder,f))
            if with_ext is True:
                filenames.append(os.path.basename(f))
            else:
                filenames.append(os.path.splitext(os.path.basename(f))[0])

    return filelist, filenames



def getfolderlist(Imagefolder):
    '''inputext: ['.json'] '''

    folder_list = []
    folder_names = []
    allfiles = sorted(os.listdir(Imagefolder))

    for f in allfiles:
        this_path = os.path.join(Imagefolder, f)
        if os.path.isdir(this_path):
            folder_list.append(this_path)
            folder_names.append(f)

    return folder_list, folder_names


def overlayWSI(wsi_path, coors, weights, alp=0.65):
    wsi_img = io.imread(wsi_path)

    alpha = np.zeros((wsi_img.shape[0], wsi_img.shape[1]), np.uint8)
    max_weight = max(weights)
    norm_weights = [ele / max_weight for ele in weights]
    for coor, weight in zip(coors, norm_weights):
        w_val = int(weight * 255)
        h_s, w_s, h_len, w_len = coor
        alpha[h_s:h_s+h_len, w_s:w_s+w_len] = w_val

    alpha = filters.gaussian(alpha, sigma=30)
    cmap = plt.get_cmap('jet')
    heat_img = cmap(alpha)[:, :, :-1] * 255

    overlay_img = (wsi_img * alp + heat_img * (1.0 - alp)).astype(np.uint8)

    return overlay_img
