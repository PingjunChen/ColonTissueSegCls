# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from numba import jit
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(B, N, true_num=None):
    '''
    Parameters:
    ------------
        B@int: batch size
        N@int: length of N
        true_num: np array of int, of shape (B,)
    Returns:
    ------------
        mask: of type np.bool, of shape (B, N). 1 indicates valid, 0 invalid.
    '''
    dis_ = np.ones((B, N), dtype=np.int32)

    if true_num is not None:
        for idx in range(B):
            this_num = true_num[idx]
            if this_num < N:
                dis_[idx, this_num::] = 0
    return dis_


class MILAtten(nn.Module):
    """MILAtten layer implementation"""
    def __init__(self, dim=2048, dl=128):
        """
        Args:
            dim : int
                Dimension of descriptors
        """

        super(MILAtten, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.dl = dl
        self.atten_dim = self.dim

        self.V = nn.Parameter(torch.Tensor(self.atten_dim, self.dl), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.dl, 1), requires_grad=True)
        self.reset_params()


    def reset_params(self):
        std1 = 1./((self.dl*self.dim)**(1/2))
        self.V.data.uniform_(-std1, std1)

        std2 = 1./((self.dl)**(1/2))
        self.W.data.uniform_(-std2, std2)


    def forward(self, x, true_num=None):
        '''
        Parameters:
        -----------
            x: B x N x D
            true_num: B
        Return
            feat_fusion:
                Bxfeat_dim
            soft_assign
                BxN
        '''

        B, num_dis, D = x.size()
        if true_num is not None:
            _mask  = get_mask(B, num_dis, true_num)
        else:
            _mask  = np.ones((B, num_dis), dtype=np.int32)
        device_mask = x.new_tensor(_mask)

        feat_ = x
        x_   = torch.tanh(torch.matmul(x,  self.V)) # BxNxL used to be torch.tanh
        dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
        dis_ = dis_/np.sqrt(self.dl)

        # set unwanted value to 0, so it won't affect.
        dis_.masked_fill_(device_mask==0, -1e20)
        soft_assign_ = F.softmax(dis_, dim=1) # BxN

        soft_assign = soft_assign_.unsqueeze(-1).expand_as(feat_)  # BxNxD
        feat_fusion = torch.sum(soft_assign*feat_, 1, keepdim=False) # BxD

        return feat_fusion, soft_assign_


def batch_fea_pooling(feas, fea_num):
    batch_size = len(fea_num)
    assignments = torch.cuda.FloatTensor(batch_size, feas.shape[1]).fill_(0)
    vlad = torch.cuda.FloatTensor(batch_size, feas.shape[2]).fill_(0)
    for ip in range(batch_size):
        patch_num = fea_num[ip]
        assignments[ip][:patch_num].fill_(1.0/patch_num)
        vlad[ip] = torch.mean(feas[ip][:patch_num], dim=0)

    return vlad, assignments


class WsiNet(nn.Module):
    def __init__(self, class_num, in_channels, mode="selfatt"):
        super(WsiNet, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        if self.mode == "selfatt":
            self.atten = MILAtten(dim=in_channels, dl=64)
            self.fc = nn.Linear(in_features=self.atten.out_dim, out_features=class_num)
        elif self.mode == "pooling":
            self.fc = nn.Linear(in_features=self.in_channels, out_features=class_num)
        else:
            raise NotImplemented()

        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.5]))
        self._loss = 0.0

    def forward(self, x, label=None, true_num=None):
        B, N, C = x.size()
        if self.mode == "selfatt":
            vlad, assignments = self.atten(x, true_num)
        elif self.mode == "pooling":
            vlad, assignments = batch_fea_pooling(x, true_num)
        else:
            raise NotImplemented()

        fusionFea = F.dropout(vlad, training=self.training)
        logits = self.fc(fusionFea)

        if self.training:
            self._loss = self.criterion(logits, label.view(-1))
            return logits, assignments
        else:
            cls_probs = F.softmax(logits, dim=1)
            return cls_probs, assignments

    @property
    def loss(self):
        return self._loss
