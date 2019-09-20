# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io, transform
import argparse, random
import PIL
import PIL.Image as Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


def set_args():
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument('--train_lib',   type=str,   default='../data/MIL/Split1235/train_inputs.pkl')
    parser.add_argument('--val_lib',     type=str,   default='../data/MIL/Split1235/val_inputs.pkl')
    parser.add_argument('--output',      type=str,   default='../data/MIL/Split1235/', help='directory of output files')
    parser.add_argument('--batch_size',  type=int,   default=512,  help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs',     type=int,   default=100,  help='number of epochs')
    parser.add_argument('--weights',     type=float, default=0.5,  help='unbalanced positive class weight')
    parser.add_argument('--k',           type=int,   default=1,    help='top k tiles are assumed to be of the same class as the slide')
    parser.add_argument('--gpu',         type=str,   default="7",  help="training gpu")
    args = parser.parse_args()
    return args

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        self.slide_list = lib["slides"]
        grid, slideIDX = [], []
        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
        self.grid = grid
        self.slideIDX = slideIDX
        self.targets = lib['targets']
        self.mult = lib['mult']
        self.size = 224*self.mult
        self.level = lib['level']
        self.transform = transform
        self.mode = None

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        import pdb; pdb.set_trace()
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = io.imread(self.slide_list[slideIDX])
            img = img[coord[0]:coord[0]+self.size, coord[1]:coord[1]+self.size]
            if self.mult != 1:
                img = transform.resize(img, (224,224)).astype(np.float64)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = io.imread(self.slide_list[slideIDX])
            img = img[coord[0]:coord[0]+self.size, coord[1]:coord[1]+self.size]
            if self.mult != 1:
                img = transform.resize(img, (224,224)).astype(np.float64)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        else:
            raise Exception("Unknown mode {}".format(self.mode))

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


def train_mil(args):
    best_acc = 0.0
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()
    # loss && optimizer
    criterion = nn.CrossEntropyLoss(torch.Tensor([1-args.weights, args.weights])).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_dset = MILdataset(args.val_lib, trans)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation

        val_dset.setmode(1)
        probs = inference(epoch, val_loader, model)
        maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err,fpr,fnr = calc_err(pred, val_dset.targets)
        print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},error,{}\n'.format(epoch+1, err))
        fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
        fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
        fconv.close()

        #Save best model
        err = (fpr+fnr)/2.
        if 1-err >= best_acc:
            best_acc = 1-err
            obj = { 'epoch': epoch+1, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer' : optimizer.state_dict()}
            torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_mil(args)
