# -*- coding: utf-8 -*-

import os, sys
import argparse
import pydaily
import torch
from torch.utils.data import DataLoader

from wsinet import WsiNet
from wsi_dataset import wsiDataSet
from train_wsinet import train_cls


def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description="Colon image classification")
    parser.add_argument('--data_dir',        type=str,          default="../data/SlideCLS/SlideFeas")
    parser.add_argument('--batch_size',      type=int,          default=32,      help='batch size.')
    parser.add_argument('--device_id',       type=str,          default="1",     help='which device')
    parser.add_argument('--pre_load',        type=bool,         default=False,   help='load setting')
    parser.add_argument('--lr',              type=float,        default=1.0e-3,  help='learning rate (default: 0.01)')
    parser.add_argument('--maxepoch',        type=int,          default=100,     help='number of epochs to train (default: 10)')
    parser.add_argument('--fusion_mode',     type=str,          default="selfatt")
    parser.add_argument('--session',         type=str,          default="01")

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    # prepare model
    net = WsiNet(class_num=2, in_channels=2048, mode=args.fusion_mode)
    net.cuda()

    # prepare dataset
    train_data_dir = os.path.join(args.data_dir, "train")
    train_dataset = wsiDataSet(train_data_dir, pre_load=args.pre_load, testing=False)
    test_data_dir = os.path.join(args.data_dir, "test")
    test_dataset = wsiDataSet(test_data_dir, pre_load=args.pre_load, testing=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    print(">> START training")
    model_root = os.path.join("../data/SlideCLS/WsiModel", args.session, args.fusion_mode)
    pydaily.filesystem.overwrite_dir(model_root)
    train_cls(net, train_dataloader, test_dataloader, model_root, args)
