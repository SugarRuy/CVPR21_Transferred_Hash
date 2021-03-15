# -*- coding: utf-8 -*-

# This file is to get the adv images for AD_attack jobs..

import scipy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pre_process as prep
import torch.utils.data as util_data
from data_list import ImageList, default_loader
from torch.autograd import Variable
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.datasets as dset

from torch.nn.functional import linear
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from myExtractCodeLabel import trans_train_resize_mnist, trans_train_resize_cifar10, trans_train_resize_imagenet
from myRetrieval import get_query_result_num_by_class, get_img_num_by_class_from_img, get_query_result_num_by_class_topN, get_img_num_by_class_from_img_topN
from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class

from myGetAdvFeature import getConvLayerByNumber

from publicFunctions import load_net_inputs, load_net_params, load_dset_params
from myGetAdvFeature import iFGSMTargetLayerFeatures, iFGSMTargetFeatures, iFGSMTargetLayerFeaturesWeight, getWeightsByTargetFeat, iFGSMTargetLayerFeaturesRandomTarget

from myGetAdv import get_multi_code, iFGSMTargetAttackMultiTarget

def iFGSMTargetAttack(model, img_t, targetCode, eps=1.0/255, l_inf_max = 32):
    # modified from myGetAdv.iFGSMTargetAttack()
    # comment print block
    # add a early stop block
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape! 
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    targetCode = targetCode.detach()

    loss = nn.L1Loss()
    l1loss = loss(output, targetCode.detach())
    #l1loss.backward(retain_graph=True)
    l1loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    #print '...targeted iterative FGSM begin....'
    #print 'initial distance', np.sum(np.abs(tCodeValue-oCodeValue))/ 2
    i = 0
    while i < l_inf_max:
        #print 'epoch ', i, ' loss: ', l1loss.cpu().data.numpy()
        print 'Hamming distance  : ', np.sum(np.abs(tCodeValue-oCodeValue))/ 2
        # early stop
        if early_stop:
            if np.sum(np.abs(tCodeValue-oCodeValue))  / 2 <= 0:
                print '...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2
                return inputs
        
        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)

        
        output_adv = model(inputs)

        l1loss = loss(output_adv, targetCode.detach())
        l1loss.backward(retain_graph=True)

        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        
    print '...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2
    return inputs


def get_test_dis():
    test_dis_black = np.ones([500, 100])*49
    for i in range(500):
        for j in range(100):
            if int(multi_label_test[i]) != j:
                a = np.linalg.norm(code_test2[i:i+1]  - code2[multi_label2==j], ord=0, axis=-1)
                test_dis_black[i, j] = a.mean()
                
    test_dis_white = np.ones([500, 100])*49
    for i in range(500):
        for j in range(100):
            if int(multi_label_test[i]) != j:
                a = np.linalg.norm(code_test[i:i+1]  - code[multi_label==j], ord=0, axis=-1)
                test_dis_white[i, j] = a.mean()
    return test_dis_white, test_dis_black

def get_adv_by_method(adv_method, step, linf):
    # choose the attack method of adv_method variable
    # adv_method: name of the adversarial method
    # step: step size
    # linf: maximum l_infinite difference
    if adv_method == 'iFGSM':
        adv__ = iFGSMTargetAttack(model1, img_t, targetCode, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMF':
        adv__ = iFGSMTargetFeatures(model1, img_t, targetFeature, targetCode, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLF':
        adv__ = iFGSMTargetLayerFeatures(model1, img_t, targetFeature, targetCode, sub_model, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLFW':
        adv__ = iFGSMTargetLayerFeaturesWeight(model1, img_t, targetFeature, targetCode, sub_model, map_weights, eps=step/255, l_inf_max = linf)
    if adv_method == 'FGSM':
        adv__ = iFGSMTargetLayerFeaturesWeight(model1, img_t, targetFeature, targetCode, sub_model, map_weights, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMMT':
        adv__ = iFGSMTargetAttackMultiTarget(model1, img_t, targetCodes, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLFRT':
        adv__ = iFGSMTargetLayerFeaturesRandomTarget(model1, img_t, targetFeatures, targetCode, sub_model, eps=step/255, l_inf_max = linf)

    return adv__

if __name__ == "__main__": 
    job_dataset = 'imagenet'
    threshold = 5
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = ['ResNet18','ResNet34', 'AlexNet', 'ResNet152', 'ResNext101_32x4d']#'ResNext101_32x4d'
    adv_method_values = ['FGSM', 'iFGSM', 'iFGSMLF', 'iFGSMLFW', 'iFGSMLFRT']
    adv_method = 'iFGSMLFW'
    
    # flllowing segment sets the AD images as the inputs 
    ad_datapath = '../data/ad_dataset/ads/0/'
    inputs1 = load_net_inputs(ad_datapath, 0, batch_size = 100)
    #inputs2 = load_net_inputs(ad_datapath, 100)
    inputs2 = load_net_inputs(ad_datapath, 0)
    
    # following segment sets the ImageNet images as the inputs
    dset_test, dset_database = load_dset_params(job_dataset)
        # same net exp
    net1 = 'ResNet152'
    net2 = 'ResNext101_32x4d'
    #net2 = 'ResNet152'
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    tmp = np.load(query_path)
    _, code_test,  multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']    
    
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    tmp = np.load(query_path2)
    _, code_test2, _ = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    
    # adv 
    import os
    ad_imagelist = os.listdir(ad_datapath)
    ad_imagelist = ad_imagelist
    
    noiid = False
    
    i_max = 16
    i_step = 1
    i_offset = 0
    j_max = 32  
    j_step = 150    
    j_offset = 0
    
    i_index_set = np.arange(i_offset, i_offset+i_step*i_max, i_step)
    j_index_set = np.arange(j_offset, j_offset+j_step*j_max, j_step)
    '''
    i_max = 1
    i_step = 1
    i_offset = 79
    j_max = 32
    j_step = 1
    j_offset = 7876
    '''
    step = 1.0
    linf = 32
    early_stop = False
    early_stop_name = 'early' if early_stop else 'noearly'
    
    layer_number = 51
    
    adv_imgs = np.zeros([i_max, j_max, 3, 224, 224])
    sub_model = getConvLayerByNumber(model1, layer_number, net1)
    if not noiid:
        inputs_ori_tensor = torch.stack([dset_test[i_index_set[i]][0] for i in range(i_max)])
    
    bLoad = True 
    if noiid:
        if 'iFGSMLF' in adv_method:
            npy_name = '/%s_imgs_layerNum_%d_step%1.1f_linf%d_%dx%d_noiid.npy' %(adv_method, layer_number, step, linf, i_max, j_max)  
            npy_path = 'save_for_load/'+net1+npy_name
        else:
            npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_noiid.npy' %(adv_method, step, linf, i_max, j_max)  
            npy_path = 'save_for_load/'+net1+npy_name
    else:
        if 'iFGSMLF' in adv_method:
            npy_name = '/%s_imgs_layerNum_%d_step%1.1f_linf%d_%dx%d.npy' %(adv_method, layer_number, step, linf, i_max, j_max)  
            npy_path = 'save_for_load/'+net1+npy_name
        else:
            npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d.npy' %(adv_method, step, linf, i_max, j_max)  
            npy_path = 'save_for_load/'+net1+npy_name
        
    if os.path.exists(npy_path):
        bLoad = True
    else:
        bLoad = False
    
    u, indices = np.unique(multi_label, return_index = True)
    
    if 'iFGSMLFW' in adv_method:
        #map_weights = getWeightsByTargetFeat(targetFeature_final)
        sub_model_fore = getConvLayerByNumber(model1, layer_number, net = net1)
        from myMapSelection import getRestLayerByNumber
        sub_model_rest = getRestLayerByNumber(model1, layer_number, net = net1)
        from myMapSelection import create_grads, save_grad, grads
    
    if not bLoad:

        for j in range(j_max):  
            j_index = j_index_set[j]
            img_t = dset_database[j_index][0]
            X = np.array(img_t.unsqueeze(0))
            inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)

            
            if 'iFGSMLF' in adv_method:
                targetFeature = sub_model(inputs)
                targetFeature_final = model1.feature_layers(inputs)

                    
            elif 'iFGSMMT' in adv_method:
                # this is wrong
                targetCodes = get_multi_code(code, multi_label, indices[int(multi_label[j_index])])
            elif adv_method == 'iFGSMLFRT':
                # random inputs
                inputs_tar_tensor = torch.stack([dset_database[ii][0] for ii in np.arange(4)+j*j_step+j_offset])
                targetFeatures = sub_model(Variable(inputs_tar_tensor).cuda())
                

            for i in range(i_max):
                i_index = i_index_set[i]
                if 'iFGSMLFW' in adv_method:
                    #map_weights = getWeightsByTargetFeat(targetFeature_final)
                    index_sample = np.array([i_index])

                    target_codes = np.repeat(np.expand_dims(code[j_index], 0), 1, axis=0)
                    grads_data = create_grads(index_sample, target_codes, dset_test, model1, sub_model_fore, sub_model_rest, layer_number)
                    index_weight = grads_data.reshape([-1])
                    map_weights = index_weight * (index_weight>0)
                
                if noiid:
                    ad_image_num = ad_imagelist[i_index]
                    ad_imagepath = ad_datapath + ad_image_num
                    img = default_loader(ad_imagepath)     
                    t = trans_train_resize_imagenet()
                    img_t = t(img)  
                else:
                    img_t = dset_test[i_index][0]
                X = np.array(img_t.unsqueeze(0))
                inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
                targetCode = code[j_index]
                adv__ = get_adv_by_method(adv_method, step, linf)

                adv_imgs[i, j] = adv__.cpu().data.numpy()
                print i,j
                
        np.save(npy_path, adv_imgs)
            
    else:     

        adv_imgs = np.load(npy_path)



    inputs_ori = Variable(inputs_ori_tensor.cuda())
    
    
    inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())
    
    label_targeted = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
    label2_targeted = np.array([multi_label2[j_index_set[j]] for j in range(j_max)])
    X = np.stack([dset_database[j_index_set[j]][0] for j in range(j_max)])
    inputs_target = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    img_num_by_class_target = get_img_num_by_class_from_img_batch(inputs_target, model1, code, multi_label, threshold=threshold, batch_size=16)  
    img_num_by_class_target_black = get_img_num_by_class_from_img_batch(inputs_target, model2, code2, multi_label2, threshold=threshold, batch_size=8)  

    img_num_by_class_adv = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label, threshold=threshold, batch_size=16)
    img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2, threshold=threshold, batch_size=8)
  
    img_num_by_class_ori_black = get_img_num_by_class_from_img_batch(inputs_ori, model2, code2, multi_label2, threshold=threshold, batch_size=8)
    
    
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_target, label_targeted)
    img_num_by_class_target_black_targeted = get_targeted_from_all_class(img_num_by_class_target_black, label2_targeted)
    
    img_num_adv_targeted = get_targeted_from_all_class(img_num_by_class_adv, label_targeted)
    img_num_adv_black_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, label2_targeted)
    
    img_num_by_class_ori_black_targeted = get_targeted_from_all_class(img_num_by_class_ori_black, label2_targeted)
    
    # clean the 
    same_cate_index = np.ones([i_max, j_max])
    for i in range(i_max):
        for j in range(j_max):
            if multi_label_test[i_index_set[i]] ==  multi_label[j_index_set[j]]:
                same_cate_index[i, j] = 0
                
    img_num_adv_black_targeted = img_num_adv_black_targeted * same_cate_index
    
    # GUIDE:
    # Compare img_num_adv_black_targeted with img_num_target_targeted, 
    # if one item in img_num_target_targeted is high enough, ignore it.
    # if we found one item has a great difference, we succeed.
    
    print adv_method+":"

    print "WhiteBox(%d imgs overall):"%(i_max*j_max)
    print "", img_num_adv_targeted.sum(), (img_num_adv_targeted>0).sum()
    
    print "BlackBox(%d imgs overall):"%(i_max*j_max)
    print "", img_num_adv_black_targeted.sum(), (img_num_adv_black_targeted>0).sum()
    
    
    from publicFunctions import model_np_batch
    code_adv_black = np.sign(model_np_batch(model2, inputs_adv, batch_size=8))
    code_ori_black = code_test2[i_index_set]
    code_targeted = code2[j_index_set]


    code_diff_adv_target = np.transpose(np.linalg.norm(code_adv_black- code_targeted,ord=0,  axis = -1))
    code_diff_ori_adv = np.linalg.norm(np.swapaxes(code_adv_black, 0, 1)- code_ori_black,ord=0,  axis = -1)
    code_diff_ori_target = np.array([np.transpose(np.linalg.norm(code_ori_black- code_targeted[j],ord=0,  axis = -1))  for j in range(j_max)])

    print code_diff_adv_target.mean()
    print code_diff_ori_adv.mean()
    print code_diff_ori_target.mean()
    

    
    
    succeed_index = np.where(img_num_adv_black_targeted>0)
    print img_num_adv_black_targeted[img_num_adv_black_targeted>0].astype(int)
    print succeed_index[0],'\n',succeed_index[1]
    
def ana():
    from myRetrieval import get_retrieval_result_by_query_code
    source_code_model1 = np.sign(model1(inputs).cpu().data.numpy())
    #a = get_retrieval_result_by_query_code(source_code_model1 , code_test, 4)
    #print a 
    
    print
    get_retrieval_result_by_query_code(code_test2[10:10 + 1], code2[multi_label2 == multi_label[750]], 5)
    
    test_dis_black = np.ones([500, 100])*49
    for i in range(500):
        for j in range(100):
            if int(multi_label_test[i]) != j:
                a = np.linalg.norm(code_test2[i:i+1]  - code2[multi_label2==j], ord=0, axis=-1)
                test_dis_black[i, j] = a.mean()
                
    test_dis_white = np.ones([500, 100])*49
    for i in range(500):
        for j in range(100):
            if int(multi_label_test[i]) != j:
                a = np.linalg.norm(code_test[i:i+1]  - code[multi_label==j], ord=0, axis=-1)
                test_dis_white[i, j] = a.mean()
    
