# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import argparse
import deepdish as dd
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from pyslide import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from patch_loader import PatchDataset, wsi_stride_splitting


def extract_model_feas(patch_model, input_tensor, args):
    if args.model_name == "resnet50":
        x = patch_model.conv1(input_tensor)
        x = patch_model.bn1(x)
        x = patch_model.relu(x)
        x = patch_model.maxpool(x)

        x = patch_model.layer1(x)
        x = patch_model.layer2(x)
        x = patch_model.layer3(x)
        x = patch_model.layer4(x)

        x = patch_model.avgpool(x)
        feas = torch.flatten(x, 1)
        logits = patch_model.fc(feas)
        probs = F.softmax(logits, dim=1)
    elif args.model_name == "vgg16bn":
        x = patch_model.features(input_tensor)
        x = patch_model.avgpool(x)
        x = torch.flatten(x, 1)
        feas = patch_model.classifier[:4](x)
        logits = patch_model.classifier[4:](feas)
        probs = F.softmax(logits, dim=-1)
    else:
        raise AssertionError("Unknown model name {}".format(args.model_name))

    return feas, probs

def gen_wsi_feas(patch_model, img_dir, fea_dir, args):
    img_list = [ele for ele in os.listdir(img_dir) if "jpg" in ele]
    for ind, ele in enumerate(img_list):
        img_name = os.path.splitext(ele)[0]
        if ind > 0 and ind % 10 == 0:
            print("processing {:03d}/{:03d}, {}".format(ind, len(img_list), img_name))
        feas_list, probs_list, coor_list = [], [], []

        img_path = os.path.join(img_dir, ele)
        cur_img = io.imread(img_path)
        # split coors and save patches
        coors_arr = wsi_stride_splitting(cur_img.shape[0], cur_img.shape[1], args.patch_len, args.stride_len)
        patch_list = []
        for ind, coor in enumerate(coors_arr):
            start_h, start_w = coor[0], coor[1]
            patch_img = cur_img[start_h:start_h+args.patch_len, start_w:start_w+args.patch_len]
            # image background control
            if patch.patch_bk_ratio(patch_img, bk_thresh=0.864) <= 0.88:
                patch_list.append(patch_img)
                coor_list.append([start_h, start_w, start_h+args.patch_len, start_w+args.patch_len])
            # Processing the feature extraction in batch-wise manner to avoid huge memory consumption
            if len(patch_list) == 16 or ind+1 == len(coors_arr):
                patch_arr = np.asarray(patch_list)
                patch_dset = PatchDataset(patch_arr)
                patch_loader = DataLoader(patch_dset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
                with torch.no_grad():
                    for inputs in patch_loader:
                        batch_tensor = Variable(inputs.cuda())
                        feas, probs = extract_model_feas(patch_model, batch_tensor, args)
                        batch_feas = feas.cpu().data.numpy().tolist()
                        batch_probs = probs.cpu().data.numpy().tolist()
                        feas_list.extend(batch_feas)
                        probs_list.extend(batch_probs)
                patch_list = []

        all_feas = np.asarray(feas_list).astype(np.float32)
        all_probs = np.asarray(probs_list).astype(np.float32)
        all_coors = np.asarray(coor_list).astype(np.float32)
        sorted_ind = np.argsort(all_probs[:, 0])
        sorted_feas = all_feas[sorted_ind]
        sorted_probs = all_probs[sorted_ind]
        sorted_coors = all_coors[sorted_ind]

        if len(feas_list) != len(probs_list) or len(feas_list) != len(coor_list):
            print("{} feas/probs/coors not consistent.".format(img_name))
        else:
            patch_fea_dict = {
                "feas": sorted_feas,
                "probs": sorted_probs,
                "coors": sorted_coors,
            }
            if not os.path.exists(args.fea_dir):
                os.makedirs(args.fea_dir)
            dd.io.save(os.path.join(args.fea_dir, img_name+".h5"), patch_fea_dict)

def set_args():
    parser = argparse.ArgumentParser(description="WSI patch-based feature extraction")
    parser.add_argument('--img_dir',       type=str,  default="../data/SlideCLS/Split1238/SlideImgs/Pos/train")
    parser.add_argument('--fea_dir',       type=str,  default="../data/SlideCLS/Split1238/SlideFeas/vgg16bn/train/1Pos")
    parser.add_argument('--model_dir',     type=str,  default="../data/PatchCLS/Split1238/Models")
    parser.add_argument('--model_name',    type=str,  default="vgg16bn")
    parser.add_argument('--patch_cls_name',type=str,  default="07-0.873.pth")
    parser.add_argument('--device_id',     type=str,  default="2",  help='which device')
    parser.add_argument('--class_num',     type=int,  default=2)
    parser.add_argument('--patch_len',     type=int,  default=448)
    parser.add_argument('--stride_len',    type=int,  default=256)
    parser.add_argument('--batch_size',    type=int,  default=16)
    parser.add_argument('--seed',          type=int,  default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # load patch cls model
    cls_weightspath = os.path.join(args.model_dir, args.model_name, args.patch_cls_name)
    patch_model = torch.load(cls_weightspath)
    patch_model.cuda()
    patch_model.eval()
    print('Load patch classification model from {}'.format(args.patch_cls_name))

    gen_wsi_feas(patch_model, args.img_dir, args.fea_dir, args)
