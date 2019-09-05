# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import time
from sklearn.metrics import confusion_matrix

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from wsinet import WsiNet


def load_wsinet(args):
    wsinet = WsiNet(class_num=2, in_channels=2048, mode=args.fusion_mode)
    weightspath = os.path.join(args.model_dir, args.cnn_model, args.fusion_mode, args.wsi_cls_name)
    wsi_weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    wsinet.load_state_dict(wsi_weights_dict)
    wsinet.cuda()
    wsinet.eval()

    return wsinet


def test_cls(net, dataloader):
    start_timer = time.time()
    total_pred, total_gt = [], []
    for ind, (batch_feas, gt_classes, true_num) in enumerate(dataloader):
        # print("Pred {:03d}/{:03d}".format(ind+1, len(dataloader)))
        im_data = Variable(batch_feas.cuda())
        # true_num = Variable(true_num.cuda())
        cls_probs, assignments = net(im_data, None, true_num=true_num)
        _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
        cls_labels = cls_labels.numpy()[:, 0]
        total_gt.extend(gt_classes.tolist())
        total_pred.extend(cls_labels.tolist())

    con_mat = confusion_matrix(total_gt, total_pred)
    cur_eval_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)

    total_time = time.time()-start_timer
    print("Testing Acc: {:.3f}".format(cur_eval_acc))
    print("Confusion matrix:")
    print(con_mat)
