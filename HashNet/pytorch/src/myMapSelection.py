# -*- coding: utf-8 -*-
# To search a good feature map to do the intermediate feature adv attack.
# Turns out that it is almost impossible.


import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.autograd import Variable


from myGetAdvFeature import getConvLayerByIndex, getConvLayerByNumber
from myFeatureAnalyze import load_net_params, load_dset_params, load_net_inputs


import math


import matplotlib.pyplot as plt
Hashbit = 48

def getRestLayerByNumber(model, layer_number, net = 'ResNext101_32x4d'):
    # Input a layerIndex, 
    # Output a rest of Sequential object that follows the features' extracting model
    # This function is complementary with getConvLayerByNumber
    if net == 'ResNext101_32x4d':
        feature_layers = model.feature_layers
        if layer_number<=3:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:4]), *list(feature_layers[0][4])[:layer_number])
            #sub_model = nn.Sequential(*list(feature_layers[0][4])[layer_number:]
        elif layer_number<=3+4:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:5]), *list(feature_layers[0][5])[:layer_number-3])
        elif layer_number<=3+4+23:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:6]), *list(feature_layers[0][6])[:layer_number-3-4])
        elif layer_number<=33:
            sub_model = nn.Sequential(nn.Sequential(nn.Sequential(*list(feature_layers)[:0]), *list(feature_layers[0])[:7]), *list(feature_layers[0][7])[:layer_number-3-4-23])
        else:
            sub_model = nn.Sequential(nn.Sequential(feature_layers[1]))
        return sub_model.eval()
    
    if net == 'ResNet152':
        feature_layers = model.feature_layers
        if layer_number<=3:
            #sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:4]), *list(feature_layers[4])[:layer_number])
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[4])[layer_number:]), *list(feature_layers)[5:])
        elif layer_number <= 3+8:
            #sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:5]), *list(feature_layers[5])[:layer_number - 3])
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[5])[layer_number - 3:]), *list(feature_layers)[6:])
        elif layer_number <= 3+8+36:
            #sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:6]), *list(feature_layers[6])[:layer_number - 3-8])
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[6])[layer_number - 3-8:]), *list(feature_layers)[7:])
        else:
            #sub_model = nn.Sequential(nn.Sequential(*list(feature_layers)[:7]), *list(feature_layers[7])[:layer_number - 3-8-36])
            sub_model = nn.Sequential(nn.Sequential(*list(feature_layers[7])[layer_number - 3-8-36:]), *list(feature_layers)[8:])
        #sub_model_rest = nn.Sequential(sub_model, model.hash_layer)
        return sub_model.eval()
    
def rest_forward(model, out_rest):
    x = out_rest
    x = x.view(x.size(0), -1)
    y = model.hash_layer(x)
    if model.iter_num % model.step_size==0:
        model.scale = model.init_scale * (math.pow((1.+model.gamma*model.iter_num), model.power))
    y = model.activation(model.scale*y)
    
    return y

def flip_activation(out_fore, map_id):
    map_size = out_fore.shape[1]
    if map_id < map_size:
        out_flip = out_fore
        out_flip[0, map_id] = out_fore[0, map_id]

def zero2one_activation(out_fore, map_id):
    map_size = out_fore.shape[1]
    if map_id < map_size:
        #out_map = out_fore[0, map_id]
        out_modi = out_fore.clone()
        out_modi[0, map_id] = (out_fore[0, map_id] == 0).float()*2.8291
        return out_modi
    else:
        print('Error')
        return 0

    
def a2b_activation(out_fore, map_id, a=1, b=0):
    map_size = out_fore.shape[1]
    if map_id < map_size:
        #out_map = out_fore[0, map_id]
        out_modi = out_fore.clone()
        out_modi[0, map_id] = (out_fore[0, map_id] == a).float()*b
        return out_modi
    else:
        print('Error')
        return 0    
def threshold_a2b_activation(out_fore, map_id, a=1, b=0, method='more'):
    map_size = out_fore.shape[1]
    if map_id < map_size:
        #out_map = out_fore[0, map_id]
        out_modi = out_fore.clone()
        if method == 'more':
            out_modi[0, map_id] = (out_fore[0, map_id] >= a).float()*b
        elif method == 'less':
            out_modi[0, map_id] = (out_fore[0, map_id] < a).float()*b
        elif method == 'equal':
            out_modi[0, map_id] = (out_fore[0, map_id] == a).float()*b
        return out_modi
    else:
        print('Error')
        return 0 
    
def threshold_a2b_activation_nBlock(out_fore, block_id, N = 32, a=1, b=0, method='more'):
    map_size = out_fore.shape[1]
    block_size = map_size / N
    if block_id < block_size:
        #out_map = out_fore[0, map_id]
        out_modi = out_fore.clone()
        if method == 'more':
            out_modi[0, block_id*N:block_id*N+N] = (out_fore[0, block_id*N:block_id*N+N] >= a).float()*b
        elif method == 'less':
            out_modi[0, block_id*N:block_id*N+N] = (out_fore[0, block_id*N:block_id*N+N] < a).float()*b
        elif method == 'equal':
            out_modi[0, block_id*N:block_id*N+N] = (out_fore[0, block_id*N:block_id*N+N] == a).float()*b
        return out_modi
    else:
        print('Error')
        return 0 
    
def tryit(out_fore, map_id):
    map_size = out_fore.shape[1]
    if map_id < map_size/2:
        #out_map = out_fore[0, map_id]
        out_modi = out_fore.clone()
        out_modi[0, :map_size/2] = (out_fore[0, :map_size/2] == 0).float()*10
        return out_modi
    elif map_id >= map_size/2:
        out_modi = out_fore.clone()
        out_modi[0, map_size/2:] = (out_fore[0, map_size/2:] == 0).float()*10
        return out_modi
    else:
        print('Error')
        return 0

def grad_weigts(model1, sub_model_fore, sub_model_rest, img_t, targetCode, layer_number):
    if not isinstance(targetCode, torch.autograd.variable.Variable):
        targetCode = Variable(torch.Tensor(targetCode).cuda())
    X = np.array(img_t.unsqueeze(0))
    inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    
    if layer_number<=50:
        out_fore = sub_model_fore(inputs)
        out_rest = sub_model_rest(out_fore)   
        out_rest.register_hook(save_grad('out_rest'))
        out_final = rest_forward(model1,out_rest)
    else:
        out_rest = sub_model_fore(inputs)
        # tmp modi for resnext101
        #if net1 != 'ResNet152':
        #    out_rest = sub_model_rest(out_rest)   
        out_rest.register_hook(save_grad('out_rest'))
        out_final = rest_forward(model1,out_rest)
    
    loss = nn.MSELoss()
    
    l2loss = loss(out_final, targetCode.detach())
    l2loss.backward(retain_graph=True)
        
    return grads['out_rest']
  
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

    
def create_grads(index_sample, target_codes, dset_database, model1, sub_model_fore, sub_model_rest, layer_number):

    sample_size = index_sample.shape[0]
    grads_data = np.zeros([sample_size, 2048])
    for i in range(sample_size):
        index = index_sample[i]
        img_t = dset_database[index][0]
        targetCode = target_codes[i]
        a = grad_weigts(model1, sub_model_fore, sub_model_rest, img_t, targetCode, layer_number)
        grads_data[i] = a.squeeze().cpu().data.numpy().reshape([-1, 2048])
        
    return grads_data

def heat_map_grads(grads_data):
    data = grads_data
    xLabel = list(np.arange(start = 0, stop = data.shape[1], step = 10).astype(str))[:100]
    yLabel = list(np.arange(start = 0, stop = data.shape[0], step = 10).astype(str))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(list(range(data.shape[0])))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(list(range(data.shape[1])))
    ax.set_xticklabels(xLabel)
    im = ax.imshow(data, cmap=plt.cm.hot)

    plt.colorbar(im)
    plt.show()
    

if __name__ == "__main__": 
    job_dataset = 'imagenet'
    threshold = 2
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = ['ResNet18','ResNet34', 'AlexNet', 'ResNet152', 'ResNext101_32x4d']#'ResNext101_32x4d'
    
    bins = 16
    # flllowing segment sets the AD images as the inputs 
    ad_datapath = '../data/ad_dataset/ads/0/'
    inputs1 = load_net_inputs(ad_datapath, 0)
    #inputs2 = load_net_inputs(ad_datapath, 100)
    inputs2 = load_net_inputs(ad_datapath, 0)
    
    # following segment sets the ImageNet images as the inputs
    dset_test, dset_database = load_dset_params(job_dataset)
    inputs1 = Variable(dset_database[0][0].unsqueeze(0)).cuda()
    
    net1 = 'ResNet152'
    net2 = 'ResNext101_32x4d'
    #net2 = 'ResNet152'
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    tmp = np.load(query_path)
    _, _,  multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']    
    
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    tmp = np.load(query_path2)
    _, code_test2, _ = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    inputs1 = Variable(torch.Tensor(dset_database[0][0].unsqueeze(0)).cuda(), requires_grad=True)
    
    len_1 = 3+8+36+3  if net1 =='ResNet152' else 3+4+23+3
    
    layer_number = 51
    
    sub_model_fore = getConvLayerByNumber(model1, layer_number, net = net1)
    sub_model_rest = getRestLayerByNumber(model1, layer_number, net = net1)
    
    
    tmp = np.load(query_path)
    _, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']  
    img_t = dset_test[0][0]
    targetCode = code_test[0]
    

    
    #a = grad_weigts(model1, sub_model_fore, sub_model_rest, img_t, targetCode)
    
    index_sample = np.array([50])#np.arange(1)
    sample_size = index_sample.shape[0]
    target_codes = code[7876:7877]
    #target_codes = np.repeat(np.expand_dims(code[180], 0), sample_size, axis=0)
    
    grads_data = create_grads(index_sample, target_codes, dset_database, model1, sub_model_fore, sub_model_rest, layer_number)
    avg_block_size = 32
    grads_data_avgpool = grads_data.reshape([-1, 2048/avg_block_size, avg_block_size]).mean(-1)
    heat_map_grads(grads_data_avgpool[:, :])
      
        
    #print grads_data[:40].mean(), grads_data[:40].max(), grads_data[:40].min()
    #print grads_data[40:].mean(), grads_data[40:].max(), grads_data[40:].min()
    #a = grads_data
    b = grads_data
    from .myFeatureAnalyze import cal_cos
    c = np.array([cal_cos(a[i], b[i]) for i in range(sample_size)])
    print(c.mean())
    
    c = np.array([cal_cos(a[i], np.random.normal(size=b[i].shape)*b[i]) for i in range(40)])
    print(c.mean())
    
def main_job():
    ori_out = np.sign(model1(inputs1).cpu().data.numpy())
    
    out_fore = sub_model_fore(inputs1)
    
    map_size = out_fore.shape[1]
    #map_size =
    modi_hash_array = np.zeros([map_size, 48])
    '''    
    for i in range(map_size):
        #out_modi = tryit(out_fore, i)
        #out_modi = zero2one_activation(out_fore, i)
        
        out_modi = threshold_a2b_activation(out_fore, i, a=0.1, b=1, method='less')
        out_rest = sub_model_rest(out_modi)   
        out_final = rest_forward(model1,out_rest)
        out_hash_np = np.sign(out_final.cpu().data.numpy())
        modi_hash_array[i] = out_hash_np
    '''    
    N = 32    
    modi_hash_array = np.zeros([map_size/N, 48])
    for i in range(map_size/N):
        out_modi = threshold_a2b_activation_nBlock(out_fore, i, a=0.1, b=1, method='less')
        out_rest = sub_model_rest(out_modi)   
        out_final = rest_forward(model1,out_rest)
        out_hash_np = np.sign(out_final.cpu().data.numpy())
        modi_hash_array[i] = out_hash_np
    
        
    print(modi_hash_array.mean(axis=0))
    print(modi_hash_array.max(axis=0))
    print(modi_hash_array.min(axis=0))
    print(modi_hash_array.mean(axis=0) - code[0])
    
    a = modi_hash_array - code[0]
    a.shape
    hamming_dis = np.abs(a).sum(axis=1)/2
    hamming_dis.shape
    print(hamming_dis.mean(), hamming_dis.max(), hamming_dis.min())



