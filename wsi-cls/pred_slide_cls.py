# -*- coding: utf-8 -*-

import os, sys
from skimage import io
import numpy as np
import argparse
import deepdish as dd
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from pyslide import patch
import pydaily
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


from patch_loader import PatchDataset, wsi_stride_splitting
from wsinet import WsiNet
from utils import overlayWSI


def extract_model_feas(patch_model, input_tensor):
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

    return feas, probs


def gen_wsi_feas(patch_model, img_path, args):
    img_name = os.path.splitext(img_path)[0]
    feas_list, probs_list, coor_list = [], [], []

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
        if len(patch_list) == args.batch_size or ind+1 == len(coors_arr):
            patch_arr = np.asarray(patch_list)
            patch_dset = PatchDataset(patch_arr)
            patch_loader = DataLoader(patch_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
            with torch.no_grad():
                for inputs in patch_loader:
                    batch_tensor = Variable(inputs.cuda())
                    feas, probs = extract_model_feas(patch_model, batch_tensor)
                    batch_feas = feas.cpu().data.numpy().tolist()
                    batch_probs = probs.cpu().data.numpy().tolist()
                    feas_list.extend(batch_feas)
                    probs_list.extend(batch_probs)
            patch_list = []

    all_feas = np.asarray(feas_list).astype(np.float32)
    all_probs = np.asarray(probs_list).astype(np.float32)
    sorted_ind = np.argsort(all_probs[:, 0])

    feas_placeholder = np.zeros((args.test_patch_num, all_feas.shape[1]), dtype=np.float32)
    test_patch_num = min(len(all_feas), args.test_patch_num)
    chosen_total_ind = sorted_ind[:test_patch_num]
    feas_placeholder[:test_patch_num] = all_feas[chosen_total_ind]
    chosen_coors = np.asarray(coor_list)[chosen_total_ind].tolist()
    return feas_placeholder, test_patch_num, chosen_coors



def set_args():
    parser = argparse.ArgumentParser(description="Colon slides classification")
    parser.add_argument('--seed',          type=int,  default=1234)
    parser.add_argument('--device_id',     type=str,  default="4",  help='which device')
    parser.add_argument('--img_dir',       type=str,  default="../data/SlideCLS/Split1234/SlideImgs/tissue-train-pos/val")
    parser.add_argument('--patch_model',   type=str,  default="../data/PatchCLS/Split1234/Models/resnet50/05-0.833.pth")
    parser.add_argument('--wsi_model_dir', type=str,  default="../data/SlideCLS/Split1234/WsiModels/resnet50")
    parser.add_argument('--fusion_mode',   type=str,  default="selfatt")
    parser.add_argument('--wsi_model_name',type=str,  default="epoch_099_acc_0.977_tn_080_fp_002_fn_001_tp_049.pth")
    parser.add_argument('--class_num',     type=int,  default=2)
    parser.add_argument('--stride_len',    type=int,  default=448)
    parser.add_argument('--patch_len',     type=int,  default=448)
    parser.add_argument('--batch_size',    type=int,  default=128)
    parser.add_argument('--test_patch_num',type=int,  default=12)
    parser.add_argument('--save_overlay',  action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # load patch cls model
    patch_model = torch.load(args.patch_model)
    patch_model.cuda()
    patch_model.eval()

    # load wsi cls model
    wsinet = WsiNet(class_num=2, in_channels=2048, mode=args.fusion_mode)
    wsi_weights_path = os.path.join(args.wsi_model_dir, args.fusion_mode, args.wsi_model_name)
    wsi_weights_dict = torch.load(wsi_weights_path, map_location=lambda storage, loc: storage)
    wsinet.load_state_dict(wsi_weights_dict)
    wsinet.cuda()
    wsinet.eval()

    # test slide
    test_slide_list = [ele for ele in os.listdir(args.img_dir) if "jpg" in ele]
    total_num = len(test_slide_list)

    correct_num = 0
    for ind, test_slide in enumerate(test_slide_list):
        print("--{:2d}/{:2d} Slide:{}".format(ind+1, total_num, test_slide))
        start_time = timer()
        test_slide_path = os.path.join(args.img_dir, test_slide)
        chosen_feas, chosen_num, chosen_coors = gen_wsi_feas(patch_model, test_slide_path, args)
        slide_fea_data = torch.from_numpy(chosen_feas).unsqueeze(0)
        im_data = Variable(slide_fea_data.cuda())
        true_num = torch.from_numpy(np.array([chosen_num]))
        cls_probs, assignments = wsinet(im_data, None, true_num=true_num)
        test_prob = cls_probs.cpu().data.numpy()[0][1]
        if test_prob >  0.5:
            correct_num += 1

        if args.save_overlay:
            weights = assignments[0].data.cpu().tolist()
            overlay_wsi = overlayWSI(test_slide_path, chosen_coors, weights)
            overlay_save_dir = os.path.join(os.path.dirname(args.img_dir), "overlay")
            pydaily.filesystem.overwrite_dir(overlay_save_dir)
            io.imsave(os.path.join(overlay_save_dir, test_slide), overlay_wsi)

        end_time = timer()
        print("Takes {}".format(pydaily.tic.time_to_str(end_time-start_time, 'sec')))
    print("stride-len: {} with batch-size: {}".format(args.stride_len, args.batch_size))
    print("Testing accuracy is {}/{}".format(correct_num, total_num))
