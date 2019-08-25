
# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch


from train_eng import validate_model
from patchloader import val_loader


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--batch_size',      type=int,   default=256)
    parser.add_argument('--model_dir',       type=str,   default="../data/PatchCLS/Split1235/Models")
    parser.add_argument('--model_name',      type=str,   default="resnet50")
    parser.add_argument('--model_path',      type=str,   default="04-0.897.pth")
    parser.add_argument("--gpu",             type=str,   default="1")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)

    model_full_path = os.path.join(args.model_dir, args.model_name, args.model_path)
    ft_model = torch.load(model_full_path)
    ft_model.cuda()
    test_data_loader = val_loader(args.batch_size)

    print("Start testing...")
    test_acc = validate_model(test_data_loader, ft_model)
    print("Testing accuracy is: {:.3f}".format(test_acc))
