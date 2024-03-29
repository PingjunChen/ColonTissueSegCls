# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse, uuid, time
from timeit import default_timer as timer
from skimage import io, transform, morphology
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter("ignore", UserWarning)
import pydaily


from segnet import pspnet, UNet
from utils import wsi_stride_splitting
from patch_loader import PatchDataset


def set_args():
    parser = argparse.ArgumentParser(description = 'Colon Tumor Slide Segmentation')
    parser.add_argument("--class_num",       type=int,   default=1)
    parser.add_argument("--in_channels",     type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=24)
    parser.add_argument("--stride_len",      type=int,   default=448)
    parser.add_argument("--patch_len",       type=int,   default=448)
    parser.add_argument("--model_name",      type=str,   default="UNet")
    parser.add_argument("--gpu",             type=str,   default="1, 2, 3")
    parser.add_argument("--best_model",      type=str,   default="UNet-048-0.623.pth")
    parser.add_argument("--model_dir",       type=str,   default="../data/PatchSeg/Model1235")
    parser.add_argument("--slides_dir",      type=str,   default="../data/SlideSeg/TestPosSlides")
    parser.add_argument("--result_dir",      type=str,   default="../data/SlideSeg/TestPosResultsPred")
    parser.add_argument("--seed",            type=int,   default=1234)

    args = parser.parse_args()
    return args


def test_slide_seg(args):
    if args.model_name == "UNet":
        model = UNet(n_channels=args.in_channels, n_classes=args.class_num)
    elif args.model_name == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(args.patch_len, args.patch_len))
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise NotImplemented("Unknown model {}".format(args.model_name))

    model_path = os.path.join(args.model_dir, args.best_model)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    since = time.time()
    pydaily.filesystem.overwrite_dir(args.result_dir)
    slide_names = [ele for ele in os.listdir(args.slides_dir) if "jpg" in ele]

    ttl_pred_dice = 0.0
    for num, cur_slide in enumerate(slide_names):
        print("--{:2d}/{:2d} Slide:{}".format(num+1, len(slide_names), cur_slide))
        start_time = timer()
        # load slide image and mask
        slide_path = os.path.join(args.slides_dir, cur_slide)
        slide_img = io.imread(slide_path)
        # split and predict
        coors_arr = wsi_stride_splitting(slide_img.shape[0], slide_img.shape[1], patch_len=args.patch_len, stride_len=args.stride_len)
        wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
        pred_map = np.zeros_like(wmap).astype(np.float32)

        patch_list, coor_list = [], []
        for ic, coor in enumerate(coors_arr):
            ph, pw = coor[0], coor[1]
            patch_list.append(slide_img[ph:ph+args.patch_len, pw:pw+args.patch_len] / 255.0)
            coor_list.append([ph, pw])
            wmap[ph:ph+args.patch_len, pw:pw+args.patch_len] += 1
            if len(patch_list) == args.batch_size or ic+1 == len(coors_arr):
                patch_arr = np.asarray(patch_list).astype(np.float32)
                patch_dset = PatchDataset(patch_arr)
                patch_loader = DataLoader(patch_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
                with torch.no_grad():
                    pred_list = []
                    for patches in patch_loader:
                        inputs = Variable(patches.cuda())
                        outputs = model(inputs)
                        preds = F.sigmoid(outputs)
                        preds = torch.squeeze(preds, dim=1).data.cpu().numpy()
                        pred_list.append(preds)
                    batch_preds = np.concatenate(pred_list, axis=0)
                    for ind, coor in enumerate(coor_list):
                        ph, pw = coor[0], coor[1]
                        pred_map[ph:ph+args.patch_len, pw:pw+args.patch_len] += batch_preds[ind]
                patch_list, coor_list = [], []

        prob_pred = np.divide(pred_map, wmap)
        slide_pred = morphology.remove_small_objects(prob_pred>0.5, min_size=20480).astype(np.uint8)
        pred_save_path = os.path.join(args.result_dir, os.path.splitext(cur_slide)[0]+".png")
        io.imsave(pred_save_path, slide_pred*255)
        end_time = timer()
        print("Takes {}".format(pydaily.tic.time_to_str(end_time-start_time, 'sec')))

    time_elapsed = time.time() - since
    print("stride-len: {} with batch-size: {}".format(args.stride_len, args.batch_size))
    print("Testing takes {:.0f}m {:.2f}s".format(time_elapsed // 60, time_elapsed % 60))


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # train model
    print("Prediction using model: {}".format(args.best_model))
    test_slide_seg(args)
