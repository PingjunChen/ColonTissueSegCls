# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import time
from sklearn.metrics import confusion_matrix

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def train_cls(net, train_dataloader, test_dataloader, model_root, args):
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5.0e-40, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(args.maxepoch, 0, 0).step)

    best_eval_acc = 0.90
    for epoc_num in range(args.maxepoch):
        start_timer = time.time()
        train_epoch_loss, sample_count = 0.0, 0
        net.train()
        for batch_idx, (batch_feas, gt_classes, true_num) in enumerate(train_dataloader):
            im_data = Variable(batch_feas.cuda())
            im_label = Variable(gt_classes.cuda())
            im_label = im_label.view(-1, 1)
            out, assignments = net(im_data, im_label, true_num=true_num)
            loss = net.loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            train_epoch_loss += loss.data.cpu().item()
            # sample_count += im_data.size()[0]
            sample_count += 1

        cur_lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, loss: {:.5f}, learning rate: {:.5f}".format(
            str(epoc_num).zfill(3), train_epoch_loss/sample_count, cur_lr))
        lr_scheduler.step()

        net.eval()
        total_pred, total_gt = [], []
        for batch_feas, gt_classes, true_num in test_dataloader:
            im_data = Variable(batch_feas.cuda())
            # true_num = Variable(true_num.cuda())
            cls_probs, assignments = net(im_data, None, true_num=true_num)
            _, cls_labels = torch.topk(cls_probs.cpu(), 1, dim=1)
            cls_labels = cls_labels.numpy()[:, 0]
            total_gt.extend(gt_classes.tolist())
            total_pred.extend(cls_labels.tolist())

        con_mat = confusion_matrix(total_gt, total_pred)
        cur_eval_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
        print("Testing Acc: {:.3f}".format(cur_eval_acc))
        print("Confusion matrix:")
        print(con_mat)
        if cur_eval_acc >= best_eval_acc or (epoc_num+1 == args.maxepoch):
            best_eval_acc = cur_eval_acc
            tn, fp, fn, tp = con_mat[0, 0], con_mat[0, 1], con_mat[1, 0], con_mat[1, 1]
            save_model_name = 'epoch_{:03d}_acc_{:.3f}_tn_{:03d}_fp_{:03d}_fn_{:03d}_tp_{:03d}.pth'.format(
                epoc_num, best_eval_acc, tn, fp, fn, tp)
            save_model_path = os.path.join(model_root, save_model_name)
            torch.save(net.state_dict(), save_model_path)
            print("Save model in {}".format(save_model_path))
        total_time = time.time()-start_timer
        print("Epoch {} takes {}.".format(str(epoc_num).zfill(3), total_time))
