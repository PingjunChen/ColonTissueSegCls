# -*- coding: utf-8 -*-

import os, sys
import argparse
import pydaily
import torch
from torch.utils.data import DataLoader

from wsinet import WsiNet
from wsi_dataset import wsiDataSet
from wsi_train_engine import train_cls


def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description="Colon image classification")
    parser.add_argument('--data_dir',        type=str,          default="../data/SlideCLS/Split1235")
    parser.add_argument('--patch_model',     type=str,          default="vgg16bn")
    parser.add_argument('--fea_len',         type=int,          default=4096)
    parser.add_argument('--batch_size',      type=int,          default=32,      help='batch size.')
    parser.add_argument('--device_id',       type=str,          default="7",     help='which device')
    parser.add_argument('--pre_load',        type=bool,         default=True,    help='load setting')
    parser.add_argument('--lr',              type=float,        default=1.0e-3,  help='learning rate (default: 0.01)')
    parser.add_argument('--maxepoch',        type=int,          default=100,     help='number of epochs to train (default: 10)')
    parser.add_argument('--fusion_mode',     type=str,          default="pooling")

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # prepare model
    net = WsiNet(class_num=2, in_channels=args.fea_len, mode=args.fusion_mode)
    net.cuda()

    # prepare dataset
    train_data_dir = os.path.join(args.data_dir, "SlideFeas", args.patch_model, "train")
    train_dataset = wsiDataSet(train_data_dir, pre_load=args.pre_load, testing=False)
    val_data_dir = os.path.join(args.data_dir, "SlideFeas", args.patch_model, "val")
    val_dataset = wsiDataSet(val_data_dir, pre_load=args.pre_load, testing=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    print(">> START training")
    model_root = os.path.join(args.data_dir, "WsiModels", args.patch_model, args.fusion_mode)
    pydaily.filesystem.overwrite_dir(model_root)
    train_cls(net, train_dataloader, val_dataloader, model_root, args)
