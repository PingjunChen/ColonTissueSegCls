# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse, uuid, time
from skimage import io, transform
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pydaily import filesystem

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter("ignore", UserWarning)

from segnet import pspnet
from utils import wsi_stride_splitting,  gen_patch_mask_wmap
from patch_loader import PatchDataset


def set_args():
    parser = argparse.ArgumentParser(description="Colon Tumor Slide Segmentation")
    parser.add_argument("--class_num",       type=int,   default=1)
    parser.add_argument("--in_channels",     type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--stride_len",      type=int,   default=64)
    parser.add_argument("--patch_len",       type=int,   default=448)
    parser.add_argument("--gpu",             type=str,   default="2, 3")
    parser.add_argument("--best_model",      type=str,   default="PSP-023-0.667.pth")
    parser.add_argument("--model_dir",       type=str,   default="../data/PatchSeg/BestModels")
    parser.add_argument("--slides_dir",      type=str,   default="../data/SlideSeg/TestPosSlides")
    parser.add_argument("--result_dir",      type=str,   default="../data/SlideSeg/TestPosResults")
    parser.add_argument("--seed",            type=int,   default=1234)

    args = parser.parse_args()
    return args


def test_slide_seg(args):
    model = pspnet.PSPNet(n_classes=19, input_size=(args.patch_len, args.patch_len))
    model.load_pretrained_model(model_path="./segnet/pspnet/pspnet101_cityscapes.caffemodel")
    model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)

    model_path = os.path.join(args.model_dir, args.best_model)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    since = time.time()
    filesystem.overwrite_dir(args.result_dir)
    slide_names = [ele for ele in os.listdir(args.slides_dir) if "jpg" in ele]

    ttl_pred_dice = 0.0
    for num, cur_slide in enumerate(slide_names):
        metrics = defaultdict(float)
        # load slide image and mask
        slide_path = os.path.join(args.slides_dir, cur_slide)
        slide_img = io.imread(slide_path) / 255.0
        mask_path = os.path.join(args.slides_dir, os.path.splitext(cur_slide)[0]+".png")
        mask_img = io.imread(mask_path) / 255.0
        # split and predict
        coors_arr = wsi_stride_splitting(slide_img.shape[0], slide_img.shape[1], patch_len=args.patch_len, stride_len=args.stride_len)
        wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
        pred_map = np.zeros_like(wmap).astype(np.float32)

        patch_list, coor_list = [], []
        for ic, coor in enumerate(coors_arr):
            ph, pw = coor[0], coor[1]
            patch_list.append(slide_img[ph:ph+args.patch_len, pw:pw+args.patch_len])
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
        slide_pred = (prob_pred > 0.5).astype(np.uint8)
        pred_save_path = os.path.join(args.result_dir, os.path.splitext(cur_slide)[0]+".png")
        io.imsave(pred_save_path, slide_pred*255)
        intersection = np.multiply(mask_img, slide_pred)
        pred_dice = np.sum(intersection) / (np.sum(mask_img)+np.sum(slide_pred)-np.sum(intersection) + 1.0e-8)
        ttl_pred_dice += pred_dice
        print("--{:2d}/{:2d} Slide:{} JI:{:.3f}".format(num+1, len(slide_names), cur_slide, pred_dice))

    time_elapsed = time.time() - since
    print('Testing takes {:.0f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Slide-level average Dice coefficient is {:.3f}'.format(ttl_pred_dice/len(slide_names)))



if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # train model
    print("Prediction using model: {}".format(args.best_model))
    test_slide_seg(args)
