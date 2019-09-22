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
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter("ignore", UserWarning)
import pydaily
from pyslide import patch

from segnet import pspnet, UNet
from utils import wsi_stride_splitting
from patch_loader import SegPatchDataset, ClsPatchDataset
from wsinet import WsiNet


def load_seg_model(args):
    if args.seg_model_name == "UNet":
        seg_model = UNet(n_channels=args.in_channels, n_classes=args.seg_class_num)
    elif args.seg_model_name == "PSP":
        seg_model = pspnet.PSPNet(n_classes=19, input_size=(args.patch_len, args.patch_len))
        seg_model.classification = nn.Conv2d(512, args.seg_class_num, kernel_size=1)
    else:
        raise NotImplemented("Unknown model {}".format(args.seg_model_name))

    seg_model_path = os.path.join(args.model_dir, "SegBestModel", args.best_seg_model)
    seg_model = nn.DataParallel(seg_model)
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.cuda()
    seg_model.eval()

    return seg_model


def load_patch_model(args):
    patch_model_path = os.path.join(args.model_dir, "PatchBestModel", args.cnn_model, args.best_patch_model)
    patch_model = torch.load(patch_model_path)
    patch_model.cuda()
    patch_model.eval()

    return patch_model


def load_wsi_model(args):
    wsi_model = WsiNet(class_num=args.wsi_class_num, in_channels=args.fea_len, mode=args.fusion_mode)
    wsi_weights_path = os.path.join(args.model_dir, "wsiBestModel", args.cnn_model,
                                    args.fusion_mode, args.wsi_model_name)
    wsi_weights_dict = torch.load(wsi_weights_path, map_location=lambda storage, loc: storage)
    wsi_model.load_state_dict(wsi_weights_dict)
    wsi_model.cuda()
    wsi_model.eval()

    return wsi_model


def seg_slide_img(seg_model, slide_path, args):
    slide_img = io.imread(slide_path)
    coors_arr = wsi_stride_splitting(slide_img.shape[0], slide_img.shape[1], patch_len=args.patch_len, stride_len=args.stride_len)
    print("h: {} w: {}".format(slide_img.shape[0], slide_img.shape[1]))
    wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
    pred_map = np.zeros_like(wmap).astype(np.float32)

    patch_list, coor_list = [], []
    for ic, coor in enumerate(coors_arr):
        ph, pw = coor[0], coor[1]
        patch_list.append(slide_img[ph:ph+args.patch_len, pw:pw+args.patch_len] / 255.0)
        coor_list.append([ph, pw])
        wmap[ph:ph+args.patch_len, pw:pw+args.patch_len] += 1
        if len(patch_list) == args.seg_batch_size or ic+1 == len(coors_arr):
            patch_arr = np.asarray(patch_list).astype(np.float32)
            patch_dset = SegPatchDataset(patch_arr)
            patch_loader = DataLoader(patch_dset, batch_size=args.seg_batch_size, shuffle=False, num_workers=0, drop_last=False)
            with torch.no_grad():
                pred_list = []
                for patches in patch_loader:
                    inputs = Variable(patches.cuda())
                    outputs = seg_model(inputs)
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
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]

    if args.gt_exist == True:
        mask_path = os.path.join(os.path.dirname(slide_path), slide_name+".png")
        if os.path.exists(mask_path):
            mask_img = io.imread(mask_path) / 255.0
            intersection = np.multiply(mask_img, slide_pred)
            pred_dice = np.sum(intersection) / (np.sum(mask_img)+np.sum(slide_pred)-np.sum(intersection) + 1.0e-8)
            print("Dice: {:.3f}".format(pred_dice))

    pred_save_path = os.path.join(args.output_dir, "predictions", os.path.basename(slide_path))
    io.imsave(pred_save_path, slide_pred*255)


def extract_model_feas(patch_model, input_tensor, args):
    if args.cnn_model == "resnet50":
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
    elif args.cnn_model == "vgg16bn":
        x = patch_model.features(input_tensor)
        x = patch_model.avgpool(x)
        x = torch.flatten(x, 1)
        feas = patch_model.classifier[:4](x)
        logits = patch_model.classifier[4:](feas)
        probs = F.softmax(logits, dim=-1)
    else:
        raise AssertionError("Unknown model name {}".format(args.cnn_model))

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
        if len(patch_list) == args.cls_batch_size or ind+1 == len(coors_arr):
            patch_arr = np.asarray(patch_list)
            patch_dset = ClsPatchDataset(patch_arr)
            patch_loader = DataLoader(patch_dset, batch_size=args.cls_batch_size, shuffle=False, num_workers=0, drop_last=False)
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
    sorted_ind = np.argsort(all_probs[:, 0])

    feas_placeholder = np.zeros((args.wsi_patch_num, all_feas.shape[1]), dtype=np.float32)
    test_patch_num = min(len(all_feas), args.wsi_patch_num)
    chosen_total_ind = sorted_ind[:test_patch_num]
    feas_placeholder[:test_patch_num] = all_feas[chosen_total_ind]
    chosen_coors = np.asarray(coor_list)[chosen_total_ind].tolist()
    return feas_placeholder, test_patch_num, chosen_coors


def cls_slide_img(patch_model, wsi_model, slide_path, args):
    chosen_feas, chosen_num, chosen_coors = gen_wsi_feas(patch_model, slide_path, args)
    slide_fea_data = torch.from_numpy(chosen_feas).unsqueeze(0)
    im_data = Variable(slide_fea_data.cuda())
    true_num = torch.from_numpy(np.array([chosen_num]))
    cls_probs, assignments = wsi_model(im_data, None, true_num=true_num)
    pos_prob = cls_probs.cpu().data.numpy()[0][1]

    return pos_prob

def set_args():
    parser = argparse.ArgumentParser(description = 'Colon Tumor Slide Segmentation')
    parser.add_argument("--seed",            type=int,  default=1234)
    parser.add_argument('--device_id',       type=str,  default="0",  help='which device')

    parser.add_argument("--in_channels",     type=int,  default=3)
    parser.add_argument("--seg_class_num",   type=int,  default=1)
    parser.add_argument('--wsi_class_num',   type=int,  default=2)
    parser.add_argument("--seg_batch_size",  type=int,  default=16)
    parser.add_argument('--cls_batch_size',  type=int,  default=96)
    parser.add_argument('--stride_len',      type=int,  default=448)
    parser.add_argument('--patch_len',       type=int,  default=448)

    parser.add_argument('--input_dir',       type=str,  default="/input")
    parser.add_argument('--output_dir',      type=str,  default="/output")
    parser.add_argument('--model_dir',       type=str,  default="./Models")
    parser.add_argument("--seg_model_name",  type=str,  default="PSP")
    parser.add_argument("--best_seg_model",  type=str,  default="PSP-050-0.665.pth")
    parser.add_argument('--cnn_model',       type=str,  default="vgg16bn")
    parser.add_argument('--fea_len',         type=int,  default=4096)
    parser.add_argument('--best_patch_model',type=str,  default="1235-05-0.909.pth")
    parser.add_argument('--fusion_mode',     type=str,  default="pooling")
    parser.add_argument('--wsi_model_name',  type=str,  default="1235-80-0.981.pth")
    parser.add_argument('--wsi_patch_num',   type=int,  default=12)
    parser.add_argument('--gt_exist',        action='store_true', default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Start testing...")
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    if torch.cuda.is_available() == False:
        raise Exception("CUDA settings error")

    # load models
    seg_model = load_seg_model(args)
    patch_model = load_patch_model(args)
    wsi_model = load_wsi_model(args)

    # start analysis
    since = time.time()
    pydaily.filesystem.overwrite_dir(os.path.join(args.output_dir, "predictions"))
    slide_names = [ele for ele in os.listdir(args.input_dir) if "jpg" in ele]
    score_list = []

    for num, cur_slide in enumerate(slide_names):
        print("--{:2d}/{:2d} Slide:{}".format(num+1, len(slide_names), cur_slide))
        start_time = timer()
        test_slide_path = os.path.join(args.input_dir, cur_slide)
        # segmentation
        seg_slide_img(seg_model, test_slide_path, args)
        pos_prob = cls_slide_img(patch_model, wsi_model, test_slide_path, args)
        score_list.append(pos_prob)
        end_time = timer()
        print("Takes {}".format(pydaily.tic.time_to_str(end_time-start_time, 'sec')))

    pred_dict = {}
    pred_dict["image_name"] = slide_names
    pred_dict["score"] = score_list
    pred_csv_path = os.path.join(args.output_dir, "predict.csv")
    pydaily.format.dict_to_csv(pred_dict, pred_csv_path)

    time_elapsed = time.time() - since
    print("Testing takes {:.0f}m {:.2f}s".format(time_elapsed // 60, time_elapsed % 60))
