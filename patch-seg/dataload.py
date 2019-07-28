# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LiverPatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.patch_dir = os.path.join(self.data_dir, "imgs")
        self.mask_dir = os.path.join(self.data_dir, "masks")
        self.patch_list = [ele for ele in os.listdir(self.patch_dir) if "jpg" in ele]
        self.transform = transform
        self.cur_img_name = None

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        self.cur_img_name = os.path.splitext(self.patch_list[idx])[0]
        patch_path = os.path.join(self.patch_dir, self.patch_list[idx])
        mask_path = os.path.join(self.mask_dir, self.cur_img_name + ".png")

        image = (io.imread(patch_path) / 255.0).astype(np.float32)
        mask = (io.imread(mask_path) / 255.0).astype(np.float32)
        mask = torch.from_numpy(np.expand_dims(mask, axis=0))

        if self.transform:
            image = self.transform(image)
        return [image, mask]



def gen_dloader(data_dir, batch_size, mode="train"):
    trans = transforms.Compose([transforms.ToTensor(), ])
    dset = LiverPatchDataset(data_dir, transform=trans)
    if mode == "train":
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=True)
    elif mode == "val":
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=False)
    else:
        raise Exception("Unknow mode: {}".format(mode))

    return dloader
