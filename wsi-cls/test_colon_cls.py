# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
from torch.utils.data import DataLoader

from wsinet import WsiNet
from wsi_dataset import wsiDataSet
from test_wsinet import load_wsinet, test_cls


def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description="TCT slides classification")
    parser.add_argument('--test_data_dir',   type=str,          default="../data/SlideCLS/SlideFeas")
    parser.add_argument('--model_dir',       type=str,          default="../data/SlideCLS/WsiModel")
    parser.add_argument('--batch_size',      type=int,          default=16,      help='batch size.')
    parser.add_argument('--device_id',       type=str,          default="0",     help='which device')
    parser.add_argument('--fusion_mode',     type=str,          default="selfatt")
    parser.add_argument('--wsi_cls_name',    type=str,          default="epoch_099_acc_0.985_tn_081_fp_001_fn_001_tp_049.pth")

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
    test_data_dir = os.path.join(args.test_data_dir, "test")
    test_dataset = wsiDataSet(test_data_dir, pre_load=True, testing=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)
    # start testing
    test_cls(wsinet, test_dataloader)
