# -*- coding: utf-8 -*-

import os, sys
import math, random, time, copy
import numpy as np
import deepdish as dd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from multiprocessing import Pool


from wsi_config import bin_class_map_dict, folder_map_dict, folder_reverse_map
from utils import getfolderlist, getfilelist


class wsiDataSet(Dataset):
    def __init__(self, data_dir, pre_load=True, testing=False):
        self.data_dir = data_dir
        self.pre_load = pre_load
        self.testing  = testing

        # category configuration
        self.folder_map_dict = folder_map_dict
        self.folder_reverse_map = folder_reverse_map
        self.class_map_dict = bin_class_map_dict

        # Load files
        file_list, label_list, file_path_list = get_all_files(data_dir, inputext=['.h5',],
                                                class_map_dict=self.folder_map_dict, pre_load=self.pre_load)
        self.file_list = file_list
        self.label_list = label_list
        self.file_path_list = file_path_list
        self.img_num = len(self.file_list)
        self.indices = list(range(self.img_num))


        self.fixed_num = 8
        self.additoinal_num = 4
        self.cell_low_num = 10
        self.cell_high_num = 18
        # self.testing_num = 12
        self.testing_num = self.cell_low_num

        self.chosen_num_list = list(range(self.cell_low_num, self.cell_high_num))
        self.max_num = self.cell_high_num if not self.testing else self.testing_num

    def get_true_label(self, label):
        new_label =  self.class_map_dict[self.folder_reverse_map[label]]
        return new_label

    def __len__(self):
        return self.img_num


    def __getitem__(self, index):
        if self.pre_load == True:
            data = self.file_list[index]
        else:
            this_data_path = self.file_list[index]
            data = dd.io.load(this_data_path)

        feas = np.asarray(data['feas'], dtype=np.float32)
        logits = np.asarray(data['probs'], dtype=np.float32)

        ttl_chosen_num = random.choice(self.chosen_num_list)
        total_ind = np.array(range(0, len(logits)))
        feas_placeholder = np.zeros((self.max_num, feas.shape[1]), dtype=np.float32)

        if self.testing:
            if len(feas) < self.testing_num:
                test_patch_num = len(feas)
            else:
                test_patch_num = self.testing_num
            chosen_total_ind_ = total_ind[:test_patch_num]
        else:
            if len(feas) > self.cell_high_num:
                # front fixed chosen part
                fixed_chosen_num = self.fixed_num+self.additoinal_num
                fixed_chosen_ind = np.random.choice(total_ind[:fixed_chosen_num], self.fixed_num)
                # later random chosen part
                random_chosen_num = ttl_chosen_num-self.fixed_num
                random_chosen_probs = logits[fixed_chosen_num:, 1] / np.sum(logits[fixed_chosen_num:, 1])
                random_chosen_ind = np.random.choice(total_ind[fixed_chosen_num:], random_chosen_num,
                                                     replace=False, p=random_chosen_probs)
                chosen_total_ind_ = np.concatenate([fixed_chosen_ind, random_chosen_ind], 0)
            else:
                chosen_total_ind_ = total_ind


        chosen_feas = feas[chosen_total_ind_]
        true_num = chosen_feas.shape[0]
        feas_placeholder[:true_num] = chosen_feas
        this_true_label = self.get_true_label(self.label_list[index])

        return feas_placeholder, this_true_label, true_num

    def __iter__(self):
        return self


def _load_h5(inputs):
    this_file_path, pre_load, this_cls = inputs
    if pre_load:
        data = dd.io.load(this_file_path)
    else:
        data = this_file_path

    return data, this_file_path, this_cls


def get_all_files(rootFolder, inputext, class_map_dict=None, pre_load=True):
    '''
    Given a root folder, this function needs to return 2 lists. imglist and clslist:
        (img_data, label)
    '''
    #if pre_load is True:
    working_pool = Pool(6)

    sub_folder_list, sub_folder_name = getfolderlist(rootFolder)
    file_list, label_list, file_path_list = [], [], []

    for idx, (this_folder, this_folder_name) in enumerate(zip(sub_folder_list, sub_folder_name)):
        filelist, filenames = getfilelist(this_folder, inputext, with_ext=False)
        this_cls = class_map_dict[this_folder_name]
        inputs_list = []
        for this_file_path, this_file_name in zip(filelist, filenames):
            inputs_list.append((this_file_path, pre_load, this_cls))
        targets = working_pool.imap(_load_h5, inputs_list)

        for _data, _this_path, _this_cls in targets:
            if _data is not None:
                file_list.append(_data)
                file_path_list.append(_this_path)
                label_list.append(_this_cls)

    return file_list, label_list, file_path_list
