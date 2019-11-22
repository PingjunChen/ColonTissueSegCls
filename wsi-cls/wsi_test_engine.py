# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import time, json
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from wsinet import WsiNet


def load_wsinet(args):
    wsinet = WsiNet(class_num=2, in_channels=args.fea_len, mode=args.fusion_mode)
    weightspath = os.path.join(args.model_dir, args.cnn_model, args.fusion_mode, args.wsi_cls_name)
    wsi_weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    wsinet.load_state_dict(wsi_weights_dict)
    wsinet.cuda()
    wsinet.eval()

    return wsinet


def test_cls(net, dataloader):
    start_timer = time.time()
    total_pred, total_pred_probs, total_gt = [], [], []
    for ind, (batch_feas, gt_classes, true_num) in enumerate(dataloader):
        # print("Pred {:03d}/{:03d}".format(ind+1, len(dataloader)))
        im_data = Variable(batch_feas.cuda())
        # true_num = Variable(true_num.cuda())
        cls_probs, assignments = net(im_data, None, true_num=true_num)
        _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
        cls_labels = cls_labels.numpy()[:, 0]

        total_pred.extend(cls_labels.tolist())
        total_pred_probs.extend(cls_probs.tolist())
        total_gt.extend(gt_classes.tolist())

    con_mat = confusion_matrix(total_gt, total_pred)
    cur_eval_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)

    total_time = time.time()-start_timer
    print("Testing Acc: {:.3f}".format(cur_eval_acc))
    print("Confusion matrix:")
    print(con_mat)

    test_gt_pred = {}
    save_json = False
    if save_json == True:
        test_gt_pred['preds'] = total_pred_probs
        test_gt_pred['gts'] = to_categorical(total_gt).tolist()
        json_path = os.path.join("../Vis", "pred_gt.json")
        with open(json_path, 'w') as fp:
            json.dump(test_gt_pred, fp)
