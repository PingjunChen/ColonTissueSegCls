# -*- coding: utf-8 -*-

import os, sys
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import numpy as np
import argparse, uuid, time
from skimage import io
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import defaultdict
from pydaily import filesystem


from segnet import UNet, pspnet
# from encoding import models
from dataload import gen_dloader
from loss import calc_loss, print_metrics
from utils import gen_patch_pred


def set_args():
    parser = argparse.ArgumentParser(description="Colon Patch Segmentation")

    parser.add_argument("--class_num",       type=int,   default=1)
    parser.add_argument("--in_channels",     type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=24)
    parser.add_argument("--gpu",             type=str,   default="1, 2, 3")
    parser.add_argument("--model_name",      type=str,   default="UNet")
    parser.add_argument("--best_model",      type=str,   default="UNet-048-0.623.pth")
    parser.add_argument("--model_dir",       type=str,   default="../data/PatchSeg/Model1235")
    parser.add_argument("--data_dir",        type=str,   default="../data/PatchSeg/SegPatches1235")
    parser.add_argument("--seed",            type=int,   default=1234)

    args = parser.parse_args()
    return args


def test_seg_model(args):
    if args.model_name == "UNet":
        model = UNet(n_channels=args.in_channels, n_classes=args.class_num)
    elif args.model_name == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(448, 448))
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise NotImplemented("Unknown model {}".format(args.model_name))
    model_path = os.path.join(args.model_dir, args.best_model)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    print('--------Start testing--------')
    since = time.time()
    dloader = gen_dloader(os.path.join(args.data_dir, "val"), args.batch_size, mode="val")

    metrics = defaultdict(float)
    ttl_samples = 0

    # preds_dir = os.path.join(args.data_dir, "val/preds", args.model_name)
    # filesystem.overwrite_dir(preds_dir)
    for batch_ind, (imgs, masks) in enumerate(dloader):
        if batch_ind != 0 and batch_ind % 100 == 0:
            print("Processing {}/{}".format(batch_ind, len(dloader)))
        inputs = Variable(imgs.cuda())
        masks = Variable(masks.cuda())

        with torch.no_grad():
            outputs = model(inputs)
            loss = calc_loss(outputs, masks, metrics)
            # result_img = gen_patch_pred(inputs, masks, outputs)
            # result_path = os.path.join(preds_dir, str(uuid.uuid1())[:8] + ".png")
            # io.imsave(result_path, result_img)

        ttl_samples += inputs.size(0)
    avg_dice = metrics['dice'] / ttl_samples
    time_elapsed = time.time() - since
    print('Testing takes {:.0f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("----Dice coefficient is: {:.3f}".format(avg_dice))

if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # train model
    print("Test using: {}".format(args.model_name))
    test_seg_model(args)
