# -*- coding: utf-8 -*-
import os

import scipy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pre_process as prep
import torch.utils.data as util_data

from publicFunctions import load_net_params, load_dset_params
from data_list import ImageList, default_loader
from torch.autograd import Variable
import torchvision.transforms as transforms

import torchvision.datasets as dset


import matplotlib.pyplot as plt

from myExtractCodeLabel import trans_train_resize_mnist, trans_train_resize_cifar10, trans_train_resize_imagenet

from myRetrieval import get_query_result_num_by_class, get_img_num_by_class_from_img

import time
#global
st_var = False
saved_eps = 0
#st_var = True 

def simple_loss():
    return 

def generate_mask():
    return

def apply_mask():
    return

def hamming_dis():
    return 

def make_one_hot(labels, C=10):
    one_hot = np.zeros([labels.shape[0], C])
    for i in range(labels.shape[0]):
        one_hot[i, labels[i].astype("uint8")] = 1
    return one_hot

def trans_train(resize = 256, crop_size = 224):
    return  transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(crop_size),
        transforms.Grayscale(num_output_channels = 3), 
        transforms.ToTensor()
        ])
    
def ndarray2PILImage(img_np):
    t = transforms.ToPILImage()
    return t(np.moveaxis(np.uint8(img_np*255), 0, -1))

def randomly_resize_transform(crop_size=199, ori_size=224):
    import random 
    resize_size = random.randint(int(crop_size/2), int(ori_size/2))*2
    return  transforms.Compose([
        transforms.Resize(size=(resize_size, resize_size)),
        transforms.Pad(int(ori_size/2)-int(resize_size/2)),
        transforms.ToTensor()
        ])

def randomly_input_diversity(adv_np, p = 0.5):
    import random 
    t_prob = random.random()
    if t_prob > p:
        #print 'original adv'
        return Variable(torch.Tensor(adv_np).unsqueeze_(0).cuda(), requires_grad=True)
    else:
        t = randomly_resize_transform()
        
        adv_pil = ndarray2PILImage(adv_np)
        adv_tensor = t(adv_pil).unsqueeze_(0)
        # print adv_tensor.size()
        return Variable(adv_tensor.cuda(), requires_grad=True)

def get_multi_code(code_test, multi_label_test, index):
    sameCateIndex = (multi_label_test==multi_label_test[index])
    multi_code = code_test[sameCateIndex]
    return multi_code
    
def get_adv_untarget(model, img_t, targetCode, eps=1.0/255, threshold = 40, loss = nn.L1Loss()):
    start = time.time()
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    # BE CAUTIOUS when you do the reshape! 
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    targetCode = targetCode.detach()

    l1loss = loss(output, targetCode.detach())
    #l1loss.backward(retain_graph=True)
    l1loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    # Everything with 'code' in it is the one signed before
    # xxxCodeValue was signed 
    # 
    print('...non-targeted iterative FGSM begin....')
    print('initial distance', np.sum(np.abs(tCodeValue-oCodeValue)) / 2)
    i = 0
    while np.sum(np.abs(tCodeValue-oCodeValue)) / 2  <= threshold:
        print('epoch ', i, ' loss: ', l1loss.cpu().data.numpy())
        print('Hamming distance: ', np.sum(np.abs(tCodeValue-oCodeValue)) / 2) 

        adv = inputs + eps * torch.sign(inputs.grad)
        #inputs.grad.data.zero_()
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_adv = model(inputs)

        l1loss = loss(output_adv, targetCode.detach())
        l1loss.backward(retain_graph=True)
        
        #oCodeValue = torch.sign(outputCode).cpu().data.numpy()
        oCodeValue = torch.sign(output_adv).cpu().data.numpy()

        i = i + 1 
        if i>=20:
            print('Adv generation failed')
            end = time.time()
            print(end-start)
            return False
        
    print('Final Hamming distance : ', np.sum(np.abs(tCodeValue-oCodeValue)) / 2) 
    return inputs

def iFGSMTargetAttack(model, img_t, targetCode, eps=1.0/255, l_inf_max = 32):
    return targetedAttack(model, img_t, targetCode, eps=eps, l_inf_max=l_inf_max, adv_method='iFGSM')

def miFGSMTargetAttack(model, img_t, targetCode, eps=1.0/255, threshold = 2, loss = nn.L1Loss(), decay_factor = 1.0):
    return targetedAttack(model, img_t, targetCode, eps=eps, threshold_stop=threshold, loss=loss, decay_factor=decay_factor, bEarlyStop=True, adv_method='miFGSM')



def iFGSMTargetAttackMultiTarget(model, img_t, multiTargetCode, eps=1.0/255, l_inf_max = 32):
    
    multiTargetNP = multiTargetCode
    if not isinstance(multiTargetCode, torch.cuda.FloatTensor):
        multiTargetCode = Variable(torch.Tensor(multiTargetCode).cuda())
        
    # Calculate the weights of each target code.
    uniqueTargetNP, targetIndex, targetCounts = np.unique(multiTargetNP, return_index=True, return_counts=True, axis=0)
    targetWeights = targetCounts.astype(float) / targetCounts.sum()
    multiTargetCode = multiTargetCode[targetIndex]
    
    from myLoss import multiTargetL1Loss as loss
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(inputs)
    multiTargetCode = multiTargetCode.detach()

    multiL1loss = loss(output, multiTargetCode, targetWeights)
    multiL1loss.backward(retain_graph=True)
    tmp = inputs.grad
    
    print('...targeted iterative FGSM begin....')

    i = 0
    while i < l_inf_max:
        print('epoch ', i, ' loss: ', multiL1loss.cpu().data.numpy())

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)

        output_adv = model(inputs)

        multiL1loss = loss(output_adv, multiTargetCode, targetWeights)
        multiL1loss.backward(retain_graph=True)
        i = i + 1 

    return inputs

def iFGSMTargetAttackMultiTargetDI(model, img_t,multiTargetCode, eps=1.0/255, l_inf_max = 32, t_prob = 0.5):
    
    multiTargetNP = multiTargetCode
    if not isinstance(multiTargetCode, torch.cuda.FloatTensor):
        multiTargetCode = Variable(torch.Tensor(multiTargetCode).cuda())

    # Calculate the weights of each target code.
    uniqueTargetNP, targetIndex, targetCounts = np.unique(multiTargetNP, return_index=True, return_counts=True, axis=0)
    targetWeights = targetCounts.astype(float) / targetCounts.sum()
    multiTargetCode = multiTargetCode[targetIndex]
    multiTargetCode = multiTargetCode.detach()
        
    X = np.array(img_t.unsqueeze(0))

    i = 0
    
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    from myLoss import multiTargetL1Loss as loss
    
    while i < l_inf_max:   
        adv_np = adv.cpu().data.numpy()
        adv_np[adv_np<0] = 0
        adv_np[adv_np>1] = 1
      
        inputs = randomly_input_diversity(adv_np[0], p = t_prob) 
        
        output_t = model(inputs)
        multiL1loss = loss(output_t, multiTargetCode, targetWeights)
        multiL1loss.backward(retain_graph=True)
        grad_var = inputs.grad
        
        adv = adv - eps * torch.sign(grad_var)
    
        i = i + 1 

    adv_np = adv.cpu().data.numpy()
    adv_np[adv_np<0] = 0
    adv_np[adv_np>1] = 1
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    return adv

def iFGSMMultiTargetEnsembleRes(model_list, img_t, multiTargetCode_list, eps=1.0/255, l_inf_max = 32):
    
    model_size = len(model_list)
    model_weights = np.ones([model_size])/model_size
    #model_weights = torch.Tensor(np.ones([model_size])/model_size).cuda()
    output_list = []
    targetCodes_list = []
    codesWeights_list = []
    
    for j in range(model_size):
        multiTargetCode = multiTargetCode_list[j]
        multiTargetNP = multiTargetCode
        if not isinstance(multiTargetCode, torch.cuda.FloatTensor):
            multiTargetCode = Variable(torch.Tensor(multiTargetCode).cuda())
        
        # Calculate the weights of each target code.
        uniqueTargetNP, targetIndex, targetCounts = np.unique(multiTargetNP, return_index=True, return_counts=True, axis=0)
        targetWeights = targetCounts.astype(float) / targetCounts.sum()
        multiTargetCode = multiTargetCode[targetIndex]
        targetCodes_list.append(multiTargetCode)
        codesWeights_list.append(targetWeights)
        
    from myLoss import multiTargetL1LossEnsRes as loss
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    for j in range(model_size):
        model_j = model_list[j]
        output_list.append(model_j(inputs))
    output_list = torch.stack(output_list[j])

    multiTargetCode = multiTargetCode.detach()
    multiL1loss = loss(output_list, model_weights, targetCodes_list, codesWeights_list)
    multiL1loss.backward(retain_graph=True)
    tmp = inputs.grad
        
    print('...targeted iterative FGSM begin....')

    i = 0
    while i < l_inf_max:
        print('epoch ', i, ' loss: ', multiL1loss.cpu().data.numpy())

        adv = inputs - eps * torch.sign(inputs.grad)
        tmp = adv.cpu().data.numpy()
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1
        inputs = Variable(torch.Tensor(tmp).cuda(), requires_grad=True)
        
        output_list = []
        for j in range(model_size):
            model_j = model_list[j]
            output_list.append(model_j(inputs))
        output_list = torch.stack(output_list[j])

        multiL1loss = loss(output_list, model_weights, targetCodes_list, codesWeights_list)
        multiL1loss.backward(retain_graph=True)
        i = i + 1 

    return inputs




def get_trans_img(img, job_dataset):
    # This function is being widely used.
    # It will be refactored next time.
    import numpy as np
    img = np.asarray(img)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
        
    if job_dataset == 'mnist':
        return img
    if job_dataset == 'coco':
        #mean=[0.485, 0.456, 0.406]
        #std=[0.229, 0.224, 0.225]
        r_mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255]
        r_std=[1/0.229, 1/0.224, 1/0.255]
        for i in range(3):
            #tmp[:,:,i] = tmp[:,:,i]*std[i]+mean[i] 
            print(i)
            tmp[:,:,i] = (tmp[:,:,i] - r_mean[i]) / r_std[i]
        tmp[tmp<0] = 0
        tmp[tmp>1] = 1        
        return img
            
    if job_dataset == 'cifar10':
        return img
    
    if 'imagenet' in job_dataset:
        return img
    if job_dataset == 'fashion_mnist':
        return img


def get_dsets_loader_imagenet(mode='test', data_batch_size=16):
    prep_dict = {}
    dsets = {}

    dset_loaders = {}
    config = {}

    prep_dict["train_set1"] = trans_train_resize_imagenet()
    prep_dict["database"] = trans_train_resize_imagenet()
    prep_dict["test"] = trans_train_resize_imagenet()

    config["data"] = {"database": {"list_path": "../data/imagenet/database.txt", "batch_size": data_batch_size}, \
                      "test": {"list_path": "../data/imagenet/test.txt", "batch_size": data_batch_size}}
    data_config = config["data"]

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                              transform=prep_dict["test"])
    dsets["database"] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                  transform=prep_dict["database"])

    dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                batch_size=data_config["test"]["batch_size"], \
                                                shuffle=False, num_workers=16)

    dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                                    batch_size=data_config["database"]["batch_size"], \
                                                    shuffle=False, num_workers=16)
    return dsets, dset_loaders

def get_dsets(job_dataset):
    # This function is being widely used.
    # It will be refactored next time.
    dsets = {}
    if job_dataset == 'mnist':
        root = '../../../../data/mnist'
        trans = trans_train_resize_mnist()
        dsets['test'] = dset.MNIST(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.MNIST(root=root, train=True, transform=trans, download=True)
    if job_dataset == 'cifar10':
        root = '../../../../data/cifar10'
        trans = trans_train_resize_cifar10()
        dsets['test'] = dset.CIFAR10(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.CIFAR10(root=root, train=True, transform=trans, download=True)
    if 'imagenet' in job_dataset:

        dsets, dset_loaders = get_dsets_loader_imagenet()
    if job_dataset == 'fashion_mnist':
        root = '../../../../data/fashion_mnist'
        trans = trans_train_resize_mnist()
        dsets['test'] = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)
        dsets['database'] = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
    return dsets

def get_round_adv(adv__):
    img_round = torch.round(adv__ * 255) / 255
    return img_round

def get_model_list(*names):
    model_list = []
    iters_list = {"ResNet18":33000, "ResNet34":10000, "ResNet50": 10000, "ResNet152":16000, "Inc_v4":36000}
    job_dataset = 'imagenet'
    
    for net in names:
        print(net)
        iters = iters_list[net]
        
        snapshot_path = '../snapshot/'+job_dataset+'_48bit_'+ net +'_hashnet/'
        model_path = snapshot_path + 'iter_'+str(iters)+'_model.pth.tar'
        model = torch.load(model_path)
        model = model.eval()
        model_list.append(model)
    return model_list
    
def get_multiTargetCode_list(net_name_list, index):
    multiTargetCode_list = []
    job_dataset = 'imagenet'
    
    for net in net_name_list:
        
        query_path = './save_for_load/'+net+'/'+job_dataset+'_test_output_code_label.npz'   
        tmp = np.load(query_path)
        output_test, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
        targetCodes = get_multi_code(code_test, multi_label_test, index)
        multiTargetCode_list.append(targetCodes)
    return multiTargetCode_list

def pgd_targetedAttack(model, img_t, targetCode, eps = 0.003, l_inf_max = 32, threshold_stop = 2, bShowProcess = False, bEarlyStop = False, adv_method = 'PGD'):
    # still under constructing
    adv = Variable(torch.Tensor(img_t).cuda(), requires_grad=True).unsqueeze(0)
    output = model(adv)
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()


    return img_t

def targetedAttack(model, img_t, targetCode, eps=1.0 / 255, l_inf_max=32, threshold_stop = 2, loss = nn.L1Loss(), decay_factor = 1.0, t_prob = 0.5, bShowProcess = False, bEarlyStop = False, adv_method = 'iFGSM'):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())

    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()

    print('...targeted %s begin....'%(adv_method))
    print('initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    i = 0
    grad_var = 0
    while i < l_inf_max:
        adv_np = adv.cpu().data.numpy()
        adv_np[adv_np<0] = 0
        adv_np[adv_np>1] = 1

        # inputs_adv is used as the input of model;
        # adv is used as the adv result of iter i.
        # They are different when using Diversity Inputs method

        if adv_method == 'iFGSM' or adv_method == 'miFGSM':
            inputs_adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
        elif adv_method == 'iFGSMDI' or adv_method == 'miFGSMDI':
            inputs_adv = randomly_input_diversity(adv_np[0], p=t_prob)

        output = model(inputs_adv)
        l1loss = loss(output, targetCode.detach())
        l1loss.backward(retain_graph=True)

        if adv_method == 'iFGSM':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSM':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'iFGSMDI':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSMDI':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'PGD':
            grad_var = inputs_adv.grad

        adv = adv - eps * torch.sign(grad_var)

        if bShowProcess:
            output = model(adv)
            oCodeValue = torch.sign(output).cpu().data.numpy()
            hamming_dist = np.sum(np.abs(tCodeValue - oCodeValue)) / 2
            print("epoch %d, Hamming Distance: %d"%(i, hamming_dist))
            if bEarlyStop:
                if hamming_dist <= threshold_stop:
                    break
        elif bEarlyStop:
            output = model(adv)
            oCodeValue = torch.sign(output).cpu().data.numpy()
            hamming_dist = np.sum(np.abs(tCodeValue - oCodeValue)) / 2
            if hamming_dist <= threshold_stop:
                break
        i = i + 1

    adv_np = adv.cpu().data.numpy()
    adv_np[adv_np < 0] = 0
    adv_np[adv_np > 1] = 1
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    output = model(adv)
    oCodeValue = torch.sign(output).cpu().data.numpy()
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    return adv

if __name__ == "__main__": 
    job_dataset = 'imagenet'
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = [ 'ResNet18', 'ResNet34', 'AlexNet']
    net = 'ResNet152'
    
    step = 1.0
    linf = 32
    adv_method = 'miFGSM'
    adv_method_list = ['iFGSM', 'iFGSMDI', 'iFGSMMT', 'iFGSMMTDI']

    dset_test, dset_database = load_dset_params(job_dataset)
    model, snapshot_path, query_path, database_path = load_net_params(net)
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    tmp = np.load(query_path)
    output_test, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    # set index for the targeted image
    index = 4
    
    # Load the advertised img
    ad_datapath = '../data/ad_dataset/ads/0/'
    datapath_dir = os.listdir(ad_datapath)
    #ad_imagepath = ad_datapath + '51.jpg'
    ad_imagepath = ad_datapath + datapath_dir[1000]

    img = default_loader(ad_imagepath)
    
    t = trans_train_resize_imagenet()
    img_t = t(img)
    
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    #targetCode = np.sign(model(inputs).cpu().data.numpy())
    targetCode = code_test[index]
    targetCodes = get_multi_code(code_test, multi_label_test, index)

    if 'MT' in adv_method:
        model_list = get_model_list('ResNet18', 'ResNet50', 'ResNet152')
        multiTargetCode_list = get_multiTargetCode_list(['ResNet18', 'ResNet50', 'ResNet152'], index)

    # adv__ = iFGSMTargetAttack(model, img_t, targetCode, eps=step / 255, l_inf_max=linf)

    adv__ = targetedAttack(model, img_t, targetCode, eps=step/255, l_inf_max=linf, bShowProcess=True, bEarlyStop=False, adv_method='iFGSMDI')
    if np.array_equal(adv__.cpu().data.numpy(), X):
        import warnings
        warnings.warn( "The adv generating failed!", UserWarning)
        
    # Rounding the adv__ to int
    adv__ = (adv__*255).round()/255
    
    img_num_by_class = get_img_num_by_class_from_img(adv__, model, code, multi_label, threshold=5)
    print(img_num_by_class, np.sum(img_num_by_class[:], axis=1))


    
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

