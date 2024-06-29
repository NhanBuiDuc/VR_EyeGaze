#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:04:18 2019

@author: Aayush Chaudhary

References:
    https://evalai-forum.cloudcv.org/t/fyi-on-semantic-segmentation/180
    https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py
    https://discuss.pytorch.org/t/using-cross-entropy-loss-with-semantic-segmentation-model/31988
    https://github.com/LIVIAETS/surface-loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import os
from spiking_unet.model.ann_model.U_Net import *
from spiking_unet.snn import simulation
import os
from spiking_unet.snn.conversion import Parser
import torch
from torch import optim
from sklearn.metrics import precision_score , recall_score,f1_score
from scipy.ndimage import distance_transform_edt as distance
#%%
class FocalLoss2d(nn.Module):
    def __init__(self, weight=None,gamma=2):
        super(FocalLoss2d,self).__init__()
        self.gamma = gamma 
        self.loss = nn.NLLLoss(weight)
    def forward(self, outputs, targets):
        return self.loss((1 - nn.Softmax2d()(outputs)).pow(self.gamma) * torch.log(nn.Softmax2d()(outputs)), targets)

###https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py
# https://discuss.pytorch.org/t/using-cross-entropy-loss-with-semantic-segmentation-model/31988
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)
    
class SurfaceLoss(nn.Module):
    # Author: Rakshit Kothari
    def __init__(self, epsilon=1e-5, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []
    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2)*distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2) # Mean between pixels per channel
        score = torch.mean(score, dim=1) # Mean between channels
        return score
    
    
class GeneralizedDiceLoss(nn.Module):
    # Author: Rakshit Kothari
    # Input: (B, C, ...)
    # Target: (B, C, ...)
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True, useGPU=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        self.useGPU = useGPU
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, ip, target):

        # Rapid way to convert to one-hot. For future version, use functional
        Label = (np.arange(4) == target.cpu().numpy()[..., None]).astype(np.uint8)
        if self.useGPU:
            target = torch.from_numpy(np.rollaxis(Label, 3,start=1))
        else:
            target = torch.from_numpy(np.rollaxis(Label, 3,start=1))

        assert ip.shape == target.shape
        ip = self.norm(ip)

        # Flatten for multidimensional data
        if self.useGPU:
            ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
            target = torch.flatten(target, start_dim=2, end_dim=-1).to(torch.float32).to("cuda")
        else:
            ip = torch.flatten(ip, start_dim=2, end_dim=-1).to(torch.float32)
            target = torch.flatten(target, start_dim=2, end_dim=-1).to(torch.float32)


        numerator = ip*target
        denominator = ip + target

        class_weights = 1./(torch.sum(target, dim=2)**2).clamp(min=self.epsilon)

        A = class_weights*torch.sum(numerator, dim=2)
        B = class_weights*torch.sum(denominator, dim=2)

        dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
        if self.reduction:
            return torch.mean(1. - dice_metric.clamp(min=self.epsilon))
        else:
            return 1. - dice_metric.clamp(min=self.epsilon)

#https://github.com/LIVIAETS/surface-loss
def one_hot2dist(posmask):
    # Input: Mask. Will be converted to Bool.
    # Author: Rakshit Kothari
    assert len(posmask.shape) == 2
    h, w = posmask.shape
    res = np.zeros_like(posmask)
    # posmask = posmask.astype(np.bool)
    posmask = posmask.astype(bool)
    mxDist = np.sqrt((h-1)**2 + (w-1)**2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res/mxDist

def mIoU(predictions, targets,info=False):  ###Mean per class accuracy
    unique_labels = np.unique(targets)
    num_unique_labels = len(unique_labels)
    ious = []
    for index in range(num_unique_labels):
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection.numpy())/np.sum(union.numpy())
        ious.append(iou_score)
    if info:
        print ("per-class mIOU: ", ious)
    return np.mean(ious)
    
#https://evalai-forum.cloudcv.org/t/fyi-on-semantic-segmentation/180
#GA: Global Pixel Accuracy
#CA: Mean Class Accuracy for different classes
#
#Back: Background (non-eye part of peri-ocular region)
#Sclera: Sclera
#Iris: Iris
#Pupil: Pupil
#Precision: Computed using sklearn.metrics.precision_score(pred, gt, ‘weighted’)
#Recall: Computed using sklearn.metrics.recall_score(pred, gt, ‘weighted’)
#F1: Computed using sklearn.metrics.f1_score(pred, gt, ‘weighted’)
#IoU: Computed using the function below
def compute_mean_iou(flat_pred, flat_label,info=False):
    '''
    compute mean intersection over union (IOU) over all classes
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return: mean IOU
    '''
    unique_labels = np.unique(flat_label)
    num_unique_labels = len(unique_labels)

    Intersect = np.zeros(num_unique_labels)
    Union = np.zeros(num_unique_labels)
    precision = np.zeros(num_unique_labels)
    recall = np.zeros(num_unique_labels)
    f1 = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = flat_pred == val
        label_i = flat_label == val
        
        if info:
            precision[index] = precision_score(pred_i, label_i, 'weighted')
            recall[index] = recall_score(pred_i, label_i, 'weighted')
            f1[index] = f1_score(pred_i, label_i, 'weighted')
        
        Intersect[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        Union[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    if info:
        print ("per-class mIOU: ", Intersect / Union)
        print ("per-class precision: ", precision)
        print ("per-class recall: ", recall)
        print ("per-class f1: ", f1)
    mean_iou = np.mean(Intersect / Union)
    return mean_iou

def total_metric(nparams,miou):
    S = nparams * 4.0 /  (1024 * 1024)
    total = min(1,1.0/S) + miou
    return total * 0.5
    
    
def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_predictions(output):
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices


class Logger():
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)
     
    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print (msg)
    def write_summary(self,msg):
        self.log_file.write(msg)
        self.log_file.write('\n')
        self.log_file.flush()
        print (msg)        

def init_spike_unet(input_channel = 1, class_num = 4):

    timesteps = 15
    dataset_name = "ISBI_2012"
    base_path = './spiking_unet/test/seg_train'
    method = "connection_wise"
    scale_method = "robust"
    neuron_class = "multi"
    reset_method = 'reset_by_subtraction'
    vth = 1.0
    opts = "adam"
    batch_size = 8
    learning_rate = 1e-6
    epochs = 100
    seed1 = "42"
    seed2 = "52"
    seed3 = "21"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = { }
    str_n_join = ['optim', opts, 'batch_size', batch_size, 'lr', learning_rate, 'ep', epochs, 'm', method, 't', timesteps, 'neuron', neuron_class]
    str_s_join = ['s1', seed1, 's2', seed2, 's3', seed3] + str_n_join
    str_n = '_'.join(str(s) for s in str_n_join)
    str_s = '_'.join(str(s) for s in str_s_join)

    post_path = os.path.join(dataset_name, scale_method, reset_method, str_n)
    path = os.path.join(base_path, post_path)
    post_log_path = os.path.join('logs', dataset_name, scale_method, reset_method, str_s)
    logs_path  = os.path.join(base_path, post_log_path)

    parser = Parser(path = './spiking_unet/lambda_factor/ISBI_2012')
    pytorch_model = Segmentation_UNet(input_channel=input_channel, class_num=class_num, fnum=16) 
    random_tensor = torch.randn((batch_size, input_channel, 640, 480), dtype= torch.float32)

    temp = pytorch_model(x = random_tensor, input_type = "original")
    print("segment mask shape ", temp.shape)
    parser_model = parser.parse(pytorch_model,random_tensor, method=method, scale_method=scale_method)

    snn_model = parser.convert_to_snn(parser_model, neuron_class=neuron_class, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)
    # snn_model.to(device)

    # model_path = os.path.join(path, 'snn_model', 'snn_model.pth')


    simulator = simulation.simulator(timesteps=timesteps, dataset_name=dataset_name, 
                                        path=path, logs_path=logs_path, device=device, **kwargs)

    # train_data = random_tensor.to(device=device, dtype=torch.float32)
    # train_label = random_tensor.to(device=device, dtype=torch.long)
    # train_label = torch.squeeze(train_label, dim=1)

    # output = simulator.simulate_for_sample(snn_model, train_data)
    # print("segment mask shape ", output.shape)
    return snn_model, simulator