# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.0, l_threshold=15.0, class_num=1.0):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))

def masked_adv_loss(advOutput, imgCode, threshold = 0):
    mask = torch.abs(advOutput - imgCode) < (1+threshold)
    m = torch.sum(mask == 1).cpu().data.numpy().astype(float)[0]
    if m == 0:
        return Variable(torch.Tensor([0]))
    advOutput_masked = torch.masked_select(advOutput, mask)
    imgCode_masked = torch.masked_select(imgCode, mask)
    return torch.pow(torch.dot(advOutput_masked, imgCode_masked) / m + 1, 2)

def cal_class_code_center(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    hashbit = code.shape[1]
    imgCenterCodeListByClass = np.zeros([num_class, hashbit])
    
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_i_code = code[class_i_index, :]
        imgCenterCodeListByClass[i] = class_i_code.mean(axis=0)
    return imgCenterCodeListByClass

def cal_class_code_variance(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    hashbit = code.shape[1]
    imgVarianceCodeListByClass = np.zeros([num_class, hashbit])
    
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_i_code = code[class_i_index, :]
        imgVarianceCodeListByClass[i] = class_i_code.var(axis=0)
    return imgVarianceCodeListByClass


def multiTargetL1Loss(output, multiTargetCode, targetWeights):
    targetSize = targetWeights.shape[0]
    loss = nn.L1Loss()
    l1lossList = [loss(output, multiTargetCode[i]) for i in range(targetSize)]
    l1lossTensor = torch.stack(l1lossList)
    targetWeightsTensor = Variable(torch.Tensor(targetWeights).cuda())
    dis = torch.dot(l1lossTensor, targetWeightsTensor)
    return dis

def multiTargetL1LossEnsRes(output_list, model_weights, targetCodes_list, codesWeights_list):
    model_size = len(output_list)
    dis_list = []
    for j in range(model_size):
        targetWeights = codesWeights_list[j]
        output = output_list[j]
        multiTargetCode = targetCodes_list[j]
        
        targetSize = targetWeights.shape[0]
        loss = nn.L1Loss()
        l1lossList = [loss(output, multiTargetCode[i]) for i in range(targetSize)]
        l1lossTensor = torch.stack(l1lossList)
        targetWeightsTensor = Variable(torch.Tensor(targetWeights).cuda())
        dis = torch.dot(l1lossTensor, targetWeightsTensor) * model_weights[j]
        dis_list.append(dis)
    return torch.sum(torch.stack(dis_list))