# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
from torch.utils.data import DataLoader

from wsinet import WsiNet
from wsi_dataset import wsiDataSet
from wsi_test_engine import load_wsinet, test_cls


def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description="TCT slides classification")
    parser.add_argument('--feas_dir',        type=str,          default="../data/SlideCLS/Split1239/SlideFeas")
    parser.add_argument('--model_dir',       type=str,          default="../data/SlideCLS/Split1239/WsiModels")
    parser.add_argument('--batch_size',      type=int,          default=16,         help='batch size.')
    parser.add_argument('--device_id',       type=str,          default="2",        help='which device')
    parser.add_argument('--cnn_model',       type=str,          default="vgg16bn",  help='cnn model')
    parser.add_argument('--fea_len',         type=int,          default=4096)
    parser.add_argument('--fusion_mode',     type=str,          default="pooling")
    parser.add_argument('--wsi_cls_name',    type=str,          default="epoch_099_acc_0.985_tn_080_fp_002_fn_000_tp_050.pth")

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # prepare model
    wsinet = load_wsinet(args)
    # prepare dataset
    test_data_dir = os.path.join(args.feas_dir, args.cnn_model, "val")
    test_dataset = wsiDataSet(test_data_dir, pre_load=True, testing=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # start testing
    test_cls(wsinet, test_dataloader)
