# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch

from train_eng import train_patch_model

def set_args():
    parser = argparse.ArgumentParser(description='Patch Classification')
    parser.add_argument('--epochs',          type=int,   default=10)
    parser.add_argument('--batch_size',      type=int,   default=64)
    # Optimization parameters
    parser.add_argument('--lr',              type=float, default=1.0e-3)
    parser.add_argument('--lr_decay_epochs', type=int,   default=2)
    parser.add_argument('--lr_decay_ratio',  type=float, default=0.3)
    parser.add_argument('--weight_decay',    type=float, default=5.0e-4)
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--log_interval',    type=int,   default=100)
    # model directory and name
    parser.add_argument('--model_dir',       type=str,   default="../data/PatchCLS/Models")
    parser.add_argument('--class_num',       type=int,   default=2)
    parser.add_argument('--model_name',      type=str,   default="resnet50")
    parser.add_argument('--session',         type=str,   default="02")
    parser.add_argument("--gpu",             type=str,   default="2",     help="training gpu")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)

    print("Fine-tuning {} model".format(args.model_name))
    train_patch_model(args)
