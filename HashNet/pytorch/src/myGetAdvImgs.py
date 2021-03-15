# -*- coding: utf-8 -*-

from myGetAdv import get_dsets, get_adv_untarget_Tao, get_round_adv, get_adv_untarget, get_adv_cluster_circle_loss, get_adv_untarget_find_space_weighted
from myGetAdv import get_adv_cluster_circle_loss_lambda, make_one_hot
from myRetrieval import get_retrieval_result_by_query_code
from myLoss import masked_adv_loss, clusterCircleLoss, emptySpaceLoss_away, clusterCircleLoss_lambda
from myLoss import cal_class_code_center
from myExtractCodeLabel import get_dsets_loader
from myVerifyClass import adv_class_verify_cifar10, adv_class_verify_mnist

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset

import random
import sys, os
import argparse
from myExperiments import get_dsets_resize
from myRetrieval import get_query_result_num_by_class, get_img_num_by_class_from_img, get_query_result_num_by_class

Hashbit = 48

def code_to_result(query_code, database_code, multi_label, job_dataset):
    if job_dataset == 'mnist':
        th_h, th_l = 16, 10
    if job_dataset == 'cifar10':
        th_h, th_l = 15, 8
    if 'imagenet' in job_dataset:
        th_h, th_l =  14, 9
    if job_dataset == 'fashion_mnist':
        th_h, th_l = 16, 7
    query_result_count_high = get_query_result_num_by_class(query_code, database_code, multi_label, threshold=th_h)
    query_result_count_low = get_query_result_num_by_class(query_code, database_code, multi_label, threshold=th_l)
    return query_result_count_high, query_result_count_low


def img_hamming_distance_max_adv(model, dset_test, random_query_index):

    global Hashbit
    adv_img = np.zeros([query_size, 3, 224, 224])
    adv_code = np.zeros([query_size, Hashbit])
    for i in range(query_size):
        index = random_query_index[i]
        img = dset_test[index][0]
        img_t = img
        targetCode = code_test[index]
        adv__ = get_adv_untarget(model, img_t, targetCode, eps=1.0/255, threshold = 42)
        if adv__ is not False:
            adv_img[i] = adv__.cpu().data.numpy()
            adv_code[i] = np.sign(model(adv__).cpu().data.numpy())
    return adv_img, adv_code
        
def img_cluster_based_weighted_max_adv(model, dset_test, random_query_index, job_dataset, query_size):
    if 'imagenet' in job_dataset:
        lr = 0.03
    if job_dataset == 'mnist':
        lr = 0.003
    if job_dataset == 'cifar10':
        lr = 0.003
    if job_dataset == 'fashion_mnist':
        lr = 0.003

    global Hashbit
    adv_img = np.zeros([query_size, 3, 224, 224])
    adv_code = np.zeros([query_size, Hashbit])
    for i in range(query_size):
        index = random_query_index[i]
        img = dset_test[index][0]
        img_t = img
        if 'imagenet' in job_dataset:
            #adv__ = get_adv_cluster_circle_loss(model, img_t, lr=lr/255, code_test = code, loss_fun = clusterCircleLoss, job_dataset=job_dataset)
            adv__ = get_adv_cluster_circle_loss_lambda(model, img_t, lr=0.03/255, code_test = code_test, loss_fun = clusterCircleLoss_lambda, var_lambda=0.99, job_dataset=job_dataset)
        else: 
            adv__ = get_adv_cluster_circle_loss(model, img_t, lr=lr/255, code_test = code_test, loss_fun = clusterCircleLoss, job_dataset=job_dataset)
        if adv__ is not False:
            adv_img[i] = adv__.cpu().data.numpy()
            adv_code[i] = np.sign(model(adv__).cpu().data.numpy())
    return adv_img, adv_code

def img_code_raw(dset_test, random_query_index, code_test):
    adv_img = np.zeros([query_size, 3, 224, 224])
    adv_code = np.zeros([query_size, Hashbit])
    for i in range(query_size):
        index = random_query_index[i]
        img = dset_test[index][0]
        code = code_test[index]
        adv_img[i] = img.numpy()
        adv_code[i] = code
    return adv_img, adv_code

if __name__ == "__main__": 
    job_dataset = 'mnist'
    adv_method = 'raw'
    print adv_method, ' on ', job_dataset
    # global Hashbit
    if job_dataset == 'mnist':
        snapshot_path = '../snapshot/mnist_48bit_resnet50_hashnet_mnist/'
        model_path = snapshot_path + 'iter_46000_model.pth.tar'
        database_path = './save_for_load/mnist_database_output_code_label.npz'
        query_path = './save_for_load/mnist_test_output_code_label.npz'
        adv_class_verify = adv_class_verify_mnist
    if job_dataset == 'cifar10':
        snapshot_path = '../snapshot/cifar10_48bit_resnet50_hashnet/'
        model_path = snapshot_path + 'iter_10000_model.pth.tar'
        database_path = './save_for_load/cifar10_database_output_code_label.npz'
        query_path = './save_for_load/cifar10_test_output_code_label.npz'
        adv_class_verify = adv_class_verify_cifar10
    if job_dataset == 'imagenet':
        snapshot_path = '../snapshot/imagenet_48bit_resnet50_hashnet/'
        model_path = snapshot_path + 'iter_40000_model.pth.tar'
        query_path = './save_for_load/imagenet_test_output_code_label.npz'
        database_path = './save_for_load/imagenet_database_output_code_label.npz'  
    if job_dataset == 'imagenet64':
        snapshot_path = '../snapshot/imagenet_64bit_resnet50_hashnet/'
        model_path = snapshot_path + 'iter_26000_model.pth.tar'
        query_path = './save_for_load/imagenet64_test_output_code_label.npz'
        database_path = './save_for_load/imagenet64_database_output_code_label.npz' 
        Hashbit = 64
    if job_dataset == 'fashion_mnist':
        snapshot_path = '../snapshot/fashion_mnist_48bit_resnet50_hashnet/'
        model_path = snapshot_path + 'iter_08000_model.pth.tar'
        query_path = './save_for_load/fashion_mnist_test_output_code_label.npz'
        database_path = './save_for_load/fashion_mnist_database_output_code_label.npz'
        
    model = torch.load(model_path)
    model = model.eval()
    dsets = get_dsets_resize(job_dataset)
    #dsets, dset_loaders = get_dsets_loader()
    #from myGetAdv import get_dsets
    dsets = get_dsets(job_dataset)
    dset_test = dsets['test']
    dset_database = dsets['database']
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    tmp = np.load(query_path)
    output_test, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    query_code = code_test
    query_multi_label = multi_label_test
    
    th = {}
    th['mnist'] = [16, 10]
    th['cifar10'] = [15, 8]
    th['imagenet'] = [14, 9]
    th['imagenet64'] = [14, 9]
    th['fashion_mnist'] = [16, 7]
    

    query_size = 500
    random.seed(1)
    random_query_index = random.sample(range(query_multi_label.shape[0]), 1000)[:query_size]

    '''
    query_result_count_high, query_result_count_low = code_to_result(query_code[random_query_index], code, database_label, job_dataset)
    weak_count_high, strong_count_high = query_result_count_to_success_rate(query_result_count_high, query_label[random_query_index])
    weak_count_low, strong_count_low = query_result_count_to_success_rate(query_result_count_low, query_label[random_query_index])
    print weak_count_high, strong_count_high
    print weak_count_low, strong_count_low
    '''
    # True: Will do the whole process
    # False: Will skip the generate step, instead, loading them directly.
    if True:
        if adv_method == 'hdm':
            hdm_adv_img, hdm_adv_code = img_hamming_distance_max_adv(model, dset_test, random_query_index, query_size)
            adv_code = hdm_adv_code
        if adv_method == 'cbwm':
            cbwm_adv_img, cbwm_adv_code = img_cluster_based_weighted_max_adv(model, dset_test, random_query_index, job_dataset, query_size)
            adv_code = cbwm_adv_code
        if adv_method == 'raw':
            cbwm_adv_img, cbwm_adv_code = img_code_raw(dset_test, random_query_index, code_test)
            adv_code = cbwm_adv_code
        np.savez('save_for_load/usability/'+job_dataset+'_' + adv_method + '_size_'+str(query_size), cbwm_adv_img, adv_code, random_query_index)
        
        if adv_method == 'hdm':
            no_false_index = np.argwhere(adv_code[:,0]!=0).reshape([-1])
        else:
            label_one_hot = make_one_hot(multi_label, C=int(multi_label.max()+1) )
            center_by_class = cal_class_code_center(code, label_one_hot)  
            dis_by_class = Hashbit / 2 - 0.5 * np.matmul(adv_code, center_by_class.transpose())
            no_false_index = dis_by_class.min(axis=1)>th[job_dataset][0]
            
    else:
        if adv_method != 'raw':
            npz_path ='save_for_load/usability/'+job_dataset+'_' + adv_method + '_size_'+str(query_size)+'.npz'
            data_loaded = np.load(npz_path)
            adv_img = data_loaded['arr_0']
            adv_code = data_loaded['arr_1']
            random_query_index = data_loaded['arr_2']    
            if adv_method == 'hdm':
                no_false_index = np.argwhere(adv_code[:,0]!=0).reshape([-1])
            else:
                #label_test_one_hot = make_one_hot(multi_label_test, C=int(multi_label_test.max()+1) )
                #center_by_class = cal_class_code_center(code_test, label_test_one_hot)
                label_one_hot = make_one_hot(multi_label, C=int(multi_label.max()+1) )
                center_by_class = cal_class_code_center(code, label_one_hot)  
                dis_by_class = Hashbit / 2 - 0.5 * np.matmul(adv_code, center_by_class.transpose())
                no_false_index = dis_by_class.min(axis=1)>th[job_dataset][0]
        else:
            # adv_code is actually the query code here
            adv_code = query_code[random_query_index]
            no_false_index = np.arange(len(random_query_index))
        

