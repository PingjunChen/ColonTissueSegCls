# -*- coding: utf-8 -*-

import os, sys


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
