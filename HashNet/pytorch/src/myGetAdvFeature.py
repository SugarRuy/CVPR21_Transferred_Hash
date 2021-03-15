# -*- coding: utf-8 -*-

from myGetAdv import get_dsets
from myGetAdv import get_trans_img
from myExtractCodeLabel import trans_train_resize_imagenet

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.autograd import Variable

from myRetrieval import get_query_result_num_by_class, get_img_num_by_class_from_img
from data_list import ImageList, default_loader

import matplotlib.pyplot as plt
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

def iFGSMTargetFeatures(model, img_t, targetFeature, targetCode, eps=1.0/255, l_inf_max = 32):
    # iFGSM Target attack using features difference's loss(L2loss/MSELoss)
    # model: whitebox mode model
    # img_t: the original image
    # targetFeature: the middle-layer feature of target image
    # targetCode: the hash code of target image
    # eps: step size
    # l_inf_max: maximum iterations
    
    #if not isinstance(targetFeature, torch.cuda.FloatTensor):
    #    targetFeature = Variable(torch.Tensor(targetFeature).cuda())
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape! 
    
    # model[0] for the silly ori author's sequetial model dump saving method
    #feature_layers = model[0].feature_layers
    # when it comes to the ResNext, use model instead of model[0]
    feature_layers = model.feature_layers
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    outFeature = feature_layers(inputs)
    targetFeature = targetFeature.detach()
    targetCode = targetCode.detach()

    loss = nn.MSELoss()
    l2loss = loss(outFeature, targetFeature.detach())
    #l1loss.backward(retain_graph=True)
    l2loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    print('...targeted iterative FGSM begin....')
    print('initial distance', np.sum(np.abs(tCodeValue-oCodeValue))/ 2)
    i = 0
    while i < l_inf_max:
        print('epoch ', i, ' loss: ', l2loss.cpu().data.numpy())
        hamm_dis = np.sum(np.abs(tCodeValue-oCodeValue))/ 2
        print('Hamming distance  : ', hamm_dis)
        
        # early stop
        if hamm_dis <= 2:
            print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
            return inputs

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_adv = model(inputs)
        outFeature_adv = feature_layers(inputs)

        l2loss = loss(outFeature_adv, targetFeature.detach())
        l2loss.backward(retain_graph=True)

        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
    return inputs

def iFGSMTargetLayerFeatures(model, img_t, targetFeature, targetCode, feature_layers, eps=1.0/255, l_inf_max = 32):
    # iFGSM Target attack using features difference's loss(L2loss/MSELoss)
    # model: whitebox mode model
    # img_t: the original image
    # targetFeature: the middle-layer feature of target image
    # targetCode: the hash code of target image
    # eps: step size
    # l_inf_max: maximum iterations
    
    #if not isinstance(targetFeature, torch.cuda.FloatTensor):
    #    targetFeature = Variable(torch.Tensor(targetFeature).cuda())
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape! 
    
    # model[0] for the silly ori author's sequetial model dump saving method
    #feature_layers = model[0].feature_layers
    # when it comes to the ResNext, use model instead of model[0]
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    outFeature = feature_layers(inputs)
    targetFeature = targetFeature.detach()
    targetCode = targetCode.detach()

    loss = nn.MSELoss()
    l2loss = loss(outFeature, targetFeature.detach())
    #l1loss.backward(retain_graph=True)
    l2loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    print('...targeted iterative FGSM begin....')
    print('initial distance', np.sum(np.abs(tCodeValue-oCodeValue))/ 2)
    i = 0
    while i < l_inf_max:
        print('epoch ', i, ' loss: ', l2loss.cpu().data.numpy())
        hamm_dis = np.sum(np.abs(tCodeValue-oCodeValue))/ 2
        print('Hamming distance  : ', hamm_dis)
        
        # early stop
        if hamm_dis <= 2:
            print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
            return inputs

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_adv = model(inputs)
        outFeature_adv = feature_layers(inputs)

        l2loss = loss(outFeature_adv, targetFeature.detach())
        l2loss.backward(retain_graph=True)

        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
    return inputs

def apply_map_weights(feature, map_weights):
    map_size = feature.shape[1]
    weight_size = map_weights.shape[0]
    block_size = map_size / weight_size
    feature_weights = feature.clone()
    # ori_weight is 1 divieded by the block_size
    ori_weight = 1.0/float(block_size)
    for i in range(weight_size):
        feature_weights[:,block_size*i:block_size*i+block_size] = feature[:,block_size*i:block_size*i+block_size]*map_weights[i]/ori_weight
    return feature_weights

def iFGSMTargetLayerFeaturesWeight(model, img_t, targetFeature, targetCode, feature_layers, map_weights, eps=1.0/255, l_inf_max = 32):
    # iFGSM Target attack using features difference's loss(L2loss/MSELoss)
    # model: whitebox mode model
    # img_t: the original image
    # targetFeature: the middle-layer feature of target image
    # targetCode: the hash code of target image
    # eps: step size
    # l_inf_max: maximum iterations
    
    #if not isinstance(targetFeature, torch.cuda.FloatTensor):
    #    targetFeature = Variable(torch.Tensor(targetFeature).cuda())
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape! 
    
    # model[0] for the silly ori author's sequetial model dump saving method
    #feature_layers = model[0].feature_layers
    # when it comes to the ResNext, use model instead of model[0]
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    outFeature = feature_layers(inputs)
    targetFeature = apply_map_weights(targetFeature, map_weights)
    targetFeature = targetFeature.detach()
    targetCode = targetCode.detach()

    loss = nn.MSELoss()
    l2loss = loss(outFeature, targetFeature.detach())
    #l1loss.backward(retain_graph=True)
    l2loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    print('...targeted iterative FGSM begin....')
    print('initial distance', np.sum(np.abs(tCodeValue-oCodeValue))/ 2)
    i = 0
    while i < l_inf_max:
        print('epoch ', i, ' loss: ', l2loss.cpu().data.numpy())
        hamm_dis = np.sum(np.abs(tCodeValue-oCodeValue))/ 2
        print('Hamming distance  : ', hamm_dis)
        
        # early stop
        if hamm_dis <= 2:
            print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
            return inputs

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_adv = model(inputs)
        outFeature_adv = feature_layers(inputs)
        
        # multiply weights here
        output_adv_weighted = apply_map_weights(outFeature_adv, map_weights)

        l2loss = loss(output_adv_weighted, targetFeature.detach())
        l2loss.backward(retain_graph=True)

        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
    return inputs

def iFGSMTargetLayerFeaturesRandomTarget(model, img_t, targetFeatures, targetCode, feature_layers, eps=1.0/255, l_inf_max = 32):
    # iFGSM Target attack using features difference's loss(L2loss/MSELoss)
    # model: whitebox mode model
    # img_t: the original image
    # targetFeature: the middle-layer feature of target image
    # targetCode: the hash code of target image
    # eps: step size
    # l_inf_max: maximum iterations
    
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    #if not isinstance(targetFeatures, torch.cuda.FloatTensor):
    #    targetFeatures = Variable(torch.Tensor(targetFeatures).cuda())
    # BE CAUTIOUS when you do the reshape! 
    
    # model[0] for the silly ori author's sequetial model dump saving method
    #feature_layers = model[0].feature_layers
    # when it comes to the ResNext, use model instead of model[0]
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    outFeature = feature_layers(inputs)
    targetFeatures = targetFeatures.detach()
    
    len_feature = targetFeatures.shape[0]
    rand_index = np.random.randint(len_feature, size=[l_inf_max])
    targetFeature = targetFeatures[rand_index[0]].unsqueeze(0)
    loss = nn.MSELoss()
    l2loss = loss(outFeature, targetFeature.detach())
    #l1loss.backward(retain_graph=True)
    l2loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    print('...targeted iterative FGSM begin....')
    print('initial distance', np.sum(np.abs(tCodeValue-oCodeValue))/ 2)
    i = 0
    while i < l_inf_max:
        targetFeature = targetFeatures[rand_index[i]].unsqueeze(0)
        print('epoch ', i, ' loss: ', l2loss.cpu().data.numpy())
        hamm_dis = np.sum(np.abs(tCodeValue-oCodeValue))/ 2
        print('Hamming distance  : ', hamm_dis)
        
        # early stop
        if hamm_dis <= 2:
            print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
            return inputs

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_adv = model(inputs)
        outFeature_adv = feature_layers(inputs)

        l2loss = loss(outFeature_adv, targetFeature.detach())
        l2loss.backward(retain_graph=True)

        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue))  / 2)
    return inputs


def get_adv_by_method(adv_method, step, linf):
    # choose the attack method of adv_method variable
    # adv_method: name of the adversarial method
    # step: step size
    # linf: maximum l_infinite difference
    
    if adv_method == 'iFGSMF':
        adv__ = iFGSMTargetFeatures(model, img_t, targetFeature, targetCode, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLF':
        adv__ = iFGSMTargetLayerFeatures(model, img_t, targetFeature, targetCode, sub_model, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLFW':
        adv__ = iFGSMTargetLayerFeaturesWeight(model, img_t, targetFeature, targetCode, sub_model, map_weights, eps=step/255, l_inf_max = linf)
    if adv_method == 'iFGSMLFRT':
        adv__ = iFGSMTargetLayerFeaturesRandomTarget(model, img_t, targetFeatures, targetCode, sub_model, eps=step/255, l_inf_max = linf)
    return adv__


def featureOutput(model, img_t):
    # fe
    #model is a Sequential, model[0] is the hashnet class 
    if len(img_t.shape) == 3:
        X = np.array(img_t.unsqueeze(0))
    else:
        X = np.array(img_t)
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    # model[0] for the silly ori author's sequetial model dump saving method
    # feature_layers = model[0].feature_layers
    
    # when it comes to the ResNext, use model instead of model[0]
    if adv_method == 'iFGSMF':
        feature_layers = model.feature_layers
    if adv_method == 'iFGSMLF':
        feature_layers = sub_model
    if adv_method == 'iFGSMLFW':
        feature_layers = sub_model    
    if adv_method == 'iFGSMLFRT':
        feature_layers = sub_model   
    feature = feature_layers(inputs)
    return feature

def getConvIndex(model):
    conv_layers_index = []
    i = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers_index.append(i)
        i+=1
    return conv_layers_index

def getConvLayerByIndex(model, layer_index, net = 'ResNext101_32x4d'):
    # Input a layerIndex, 
    # Output a Sequential object for extracting features
    if net == 'ResNext101_32x4d':
        feature_layers = model.feature_layers
        
        if layer_index == 464:
            sub_model = nn.Sequential(*list(feature_layers[0])[:7])
        if layer_index == 482:    
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:1])     
        if layer_index == 487:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:1]), *list(feature_layers[0][7][1][0][0][0])[:3]))
        if layer_index == 489:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), feature_layers[0][7][0], feature_layers[0][7][1][0][0][0])    
        if layer_index == 497:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:2])
        if layer_index == 502:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:2]), *list(feature_layers[0][7][2][0][0][0])[:3]))
        if layer_index == 505:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:2]), feature_layers[0][7][2][0][0][0])
        if layer_index == 512:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:3])
        if layer_index == 514:
            sub_model = feature_layers
        return sub_model.eval()
    
    if net == 'ResNet152':
        feature_layers = model.feature_layers
        if layer_index == 425:
            sub_model = feature_layers
        if layer_index == 413:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:7]), *list(feature_layers[7])[:2])
        if layer_index == 405:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:7]), *list(feature_layers[7])[:1])
        if layer_index == 393:
            sub_model = nn.Sequential(*list(feature_layers)[:7])
        if layer_index == 377:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:6]), *list(feature_layers[6])[:34])
        if layer_index == 369:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:6]), *list(feature_layers[6])[:33])
        if layer_index == 361:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:6]), *list(feature_layers[6])[:32])
                   
        return sub_model.eval()

def getConvLayerByNumber(model, layer_number, net = 'ResNext101_32x4d'):
    # Input a layerIndex, 
    # Output a Sequential object for extracting features
    if net == 'ResNext101_32x4d':
        feature_layers = model.feature_layers
        if layer_number<=3:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:4]), *list(feature_layers[0][4])[:layer_number])
        elif layer_number<=3+4:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:5]), *list(feature_layers[0][5])[:layer_number-3])
        elif layer_number<=3+4+23:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:6]), *list(feature_layers[0][6])[:layer_number-3-4])
        else:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:layer_number-3-4-23])

        return sub_model.eval()
    
    if net == 'ResNet152':
        feature_layers = model.feature_layers
        if layer_number<=3:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:4]), *list(feature_layers[4])[:layer_number])
        elif layer_number <= 3+8:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:5]), *list(feature_layers[5])[:layer_number - 3])
        elif layer_number <= 3+8+36:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:6]), *list(feature_layers[6])[:layer_number - 3-8])
        elif layer_number <= 3+8+36+3:
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:7]), *list(feature_layers[7])[:layer_number - 3-8-36])
        else:
            sub_model = feature_layers
        return sub_model.eval()

def getAnyLayerByIndex(model, layerIndex):
    # Input a layerIndex, 
    # Output a Sequential object for extracting features
    children_list = list(model.children())
    my_model = nn.Sequential(children_list[0], children_list[1][:0])

    return my_model

def getWeightsByTargetFeat(targetFeature):
    targetFeature_np = targetFeature.cpu().data.numpy()
    #th_feature = np.median(targetFeature_np)
    th_feature = targetFeature_np.mean()
    feature_mask = (targetFeature_np>th_feature).reshape([-1])
    index_weight = feature_mask.astype(float) / feature_mask.sum()
    #map_weights = np.array([0.1,0.2,0.3,0.4])
    map_weights = index_weight
    return map_weights


def grad_weigts(model, img_t, targetCode):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    output = model(inputs)
    
    loss = nn.MSELoss()
    
    l2loss = loss(output, targetCode.detach())
    l2loss.backward(retain_graph=True)
    
    grad = model.avgpool.grad
    return grad

if __name__ == "__main__": 
    job_dataset = 'imagenet'
    threshold = 5
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = ['ResNet18','ResNet34', 'AlexNet']#'ResNext101_32x4d'
    net = 'ResNet152'
    
    adv_method_values = ['iFGSMF', 'iFGSMLF', 'iFGSMLFW', 'iFGSMLFRT']
    adv_method = 'iFGSMLFW'
    step = 1.0
    linf = 32
    
    # convIndex is the index for submodel conv, starting from 1 
    layer_index = 405
    layer_number = 60
    
    R = 500
    from .publicVariables import iters_list
    iters = iters_list[net]
 
    from .publicFunctions import create_loading_path
    path_list = create_loading_path(job_dataset, net, iters)
    snapshot_path = path_list[0]
    model_path = path_list[1]
    model_dict_path = path_list[2]
    query_path = path_list[3]
    database_path = path_list[4]
           
    from .publicFunctions import load_model
    model = load_model(net, model_path, model_dict_path)
    model = model.cuda().eval()
    
    dsets = get_dsets(job_dataset)
    dset_test = dsets['test']
    dset_database = dsets['database']
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    tmp = np.load(query_path)
    output_test, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    # set index for the targeted image
    index = 0
    
    # Load the advertised img

    noiid = False
    if noiid:
        ad_datapath = '../data/ad_dataset/ads/0/'
        ad_imagepath = ad_datapath + '50.jpg'
        
        
        #ad_imagepath = '/home/yxiao/Workspace/data/imagenet/image/n02256656_13626.JPEG'
        
        #ad_imagepath = '/home/yxiao/Workspace/data/imagenet/image/n02834397_34931.JPEG'
        img = default_loader(ad_imagepath)
        
        t = trans_train_resize_imagenet()
        img_t = t(img)
    else:
        img_t = dset_test[0][0]
        
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    #targetCode = np.sign(model(inputs).cpu().data.numpy())
    targetCode = code[index]
    img_t_target = dset_database[index][0]
    #sub_model = getConvLayerByIndex(model, layer_index, net)
    sub_model = getConvLayerByNumber(model, layer_number, net)
    targetFeature = featureOutput(model, img_t_target)

    #model_list = get_model_list('ResNet18', 'ResNet50', 'ResNet152')
    #multiTargetCode_list = get_multiTargetCode_list(['ResNet18', 'ResNet50', 'ResNet152'], index)
    if adv_method == 'iFGSMLFW':
        model1 = model
        net1 = net
        sub_model_fore = getConvLayerByNumber(model1, layer_number, net = net1)
        from myMapSelection import getRestLayerByNumber
        sub_model_rest = getRestLayerByNumber(model1, layer_number, net = net1)
        from myMapSelection import create_grads, save_grad, grads
        index_sample = np.arange(1)
        sample_size = index_sample.shape[0]
        target_codes = np.repeat(np.expand_dims(code[180], 0), sample_size, axis=0)
        grads_data = create_grads(index_sample, target_codes, dset_database, model1, sub_model_fore, sub_model_rest, layer_number)
        index_weight = grads_data.reshape([-1])
        map_weights = index_weight
        #map_weights = getWeightsByTargetFeat(targetFeature)
        
    if adv_method == 'iFGSMLFRT':
        inputs_tar_tensor = torch.stack([dset_database[i][0] for i in np.arange(8)+0*150])
        targetFeatures = sub_model(Variable(inputs_tar_tensor).cuda())
    adv__ = get_adv_by_method(adv_method, step, linf)
   
    
    if np.array_equal(adv__.cpu().data.numpy(), X):
        import warnings
        warnings.warn( "The adv generating failed!", UserWarning)
        
    # Rounding the adv__ to int
    #adv__ = (adv__*255).round()/255
    img_num_by_class_ori = get_img_num_by_class_from_img(inputs, model, code, multi_label, threshold)
    print('ori info:', img_num_by_class_ori, np.sum(img_num_by_class_ori[:], axis=1))
    img_num_by_class = get_img_num_by_class_from_img(adv__, model, code, multi_label, threshold)
    print('adv info:', img_num_by_class, np.sum(img_num_by_class[:], axis=1))
    img_num_by_class_target = get_query_result_num_by_class(np.expand_dims(targetCode,0), code, multi_label, threshold)
    print('target info:', img_num_by_class_target, np.sum(img_num_by_class_target[:], axis=1))
    
    # tmp 
    X = np.array(img_t.unsqueeze(0))
    img_ori = get_trans_img(X[0], job_dataset)
    plt.figure(666)
    plt.subplot(2,2,1)
    plt.title('Original Image')
    plt.imshow(img_ori)
    plt.subplot(2,2,2)
    plt.title('Original Image')
    plt.imshow(img_ori.mean(axis=2), cmap='gray')
    #plt.imshow(np.moveaxis(img_ori, 0, -1))
    
    img_adv = adv__.cpu().data.numpy()
    img_adv = get_trans_img(img_adv[0], job_dataset)
    plt.subplot(2,2,3)
    plt.title('Adversarial Image')
    plt.imshow(img_adv)
    plt.subplot(2,2,4)
    plt.title('Adversarial Image')
    plt.imshow(img_adv.mean(axis=2), cmap='gray')  
    
    #just trying to do an anydesk test
    print('L-inf:', np.linalg.norm((img_adv-img_ori).reshape([-1]), np.inf)*255)

    