# -*- coding: utf-8 -*-


import numpy as np

import torch

from torch.autograd import Variable


from myGetAdvFeature import getConvLayerByIndex, getConvLayerByNumber

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance as emd

from publicFunctions import load_net_inputs, load_net_params, load_dset_params

import matplotlib.pyplot as plt
Hashbit = 48

from publicVariables import layer_index_value_list



def get_feature_np(sub_model, inputs):
    # convIndex is the index for submodel conv, starting from 1 

    layer_index_value = layer_index_value_list[net]
    model, snapshot_path, query_path, database_path = load_net_params(net)
    dset_test, dset_database = load_dset_params(job_dataset)
    
    layer_index = layer_index_value[-2]
    sub_model = getConvLayerByIndex(model, layer_index, net)
    feature_out = sub_model(inputs)
    feature_np = feature_out.cpu().data.numpy()
    return 0

def get_histogram_by_sub_model(sub_model, inputs, bins = 10):
    # the bins can be modified with other numbers
    feature_out = sub_model(inputs)
    feature_np = feature_out.cpu().data.numpy()
    print(feature_np.shape)
    his_feature = np.histogram(feature_np[feature_np<=1], bins=bins)
    his_normalized = his_feature[0].astype(float) / his_feature[0].sum()
    return his_normalized
    
def get_histograms_by_net(net, layer_index_value, inputs, bins = 10):
    layer_len = len(layer_index_value)
    histograms_by_layer = np.zeros([layer_len, bins])
    model, snapshot_path, query_path, database_path = load_net_params(net)
    
    for i in range(layer_len):
        layer_index = layer_index_value[i]
        sub_model = getConvLayerByIndex(model, layer_index, net)
        histograms_by_layer[i] = get_histogram_by_sub_model(sub_model, inputs, bins = bins)
    
    return histograms_by_layer

def get_features_by_net(net, layer_index_value, inputs):
    layer_len = len(layer_index_value)
    features_by_layer = {}
    model, snapshot_path, query_path, database_path = load_net_params(net)
    
    for i in range(layer_len):
        layer_index = layer_index_value[i]
        sub_model = getConvLayerByIndex(model, layer_index, net)
        feature_out = sub_model(inputs)
        features_by_layer[i] = feature_out
    return features_by_layer
    
def cal_emd(inputs1, inputs2, bins=10):
    if isinstance(inputs1, Variable):
        inputs1_np = inputs1.cpu().data.numpy()
        inputs2_np = inputs2.cpu().data.numpy()
    elif isinstance(inputs1, np.ndarray):
        inputs1_np = inputs1
        inputs2_np = inputs2
    inputs1_gt0 = inputs1_np[inputs1_np>0]
    inputs2_gt0 = inputs2_np[inputs2_np>0]
    inputs1_gt0_lt1 = inputs1_gt0[inputs1_gt0<=1]
    inputs2_gt0_lt1 = inputs2_gt0[inputs2_gt0<=1]
    
    inputs1_gt1_count =(inputs1_np>1).sum()
    inputs2_gt1_count =(inputs2_np>1).sum()
    
    input1_np_his = np.histogram(inputs1_gt0_lt1, bins=bins)
    input2_np_his = np.histogram(inputs2_gt0_lt1, bins=bins)
    input1_norm = input1_np_his[0].astype(float) / input1_np_his[0].sum()
    input2_norm = input2_np_his[0].astype(float) / input2_np_his[0].sum()
    input1_norm_with_gt1 = np.append(input1_norm, float(inputs1_gt1_count)/input1_np_his[0].sum())
    input2_norm_with_gt1 = np.append(input2_norm, float(inputs2_gt1_count)/input2_np_his[0].sum())

    emd_inputs = emd(input1_norm_with_gt1, input2_norm_with_gt1)
    return emd_inputs

def cal_l2(inputs1, inputs2):
    input1_np = inputs1.cpu().data.numpy() * 255
    input2_np = inputs2.cpu().data.numpy() * 255
    l2_dis = np.linalg.norm(input1_np - input2_np)
    return l2_dis, l2_dis/input1_np.size

def get_l2_dis(feature_out_1, feature_out_2):

    len_feature = len(feature_out_1)
    l2_distances = np.zeros([len_feature])
    l2_distances_avg = np.zeros([len_feature])
    
    for i in range(len_feature):
        feature1_np = feature_out_1[i].cpu().data.numpy() * 255
        feature2_np = feature_out_2[i].cpu().data.numpy() * 255
        l2_distances[i] = np.linalg.norm(feature1_np - feature2_np)
        l2_distances_avg[i] = np.linalg.norm(feature1_np - feature2_np) / feature1_np.size
    return l2_distances, l2_distances_avg

def cal_cos(inputs1, inputs2):
    if isinstance(inputs1, Variable):
        inputs1_np = inputs1.cpu().data.numpy()
        inputs2_np = inputs2.cpu().data.numpy()
    elif isinstance(inputs1, np.ndarray):
        inputs1_np = inputs1
        inputs2_np = inputs2
    cos_dis = cosine_similarity(inputs1_np.reshape([1, -1]), inputs2_np.reshape([1, -1]))
    return cos_dis

def clean_both_zero(inputs1, inputs2):
    if isinstance(inputs1, Variable):
        inputs1_np = inputs1.cpu().data.numpy()
        inputs2_np = inputs2.cpu().data.numpy()
    elif isinstance(inputs1, np.ndarray):
        inputs1_np = inputs1
        inputs2_np = inputs2
    index_not_both_zero = (inputs1_np!=0)+(inputs2_np!=0)
    return inputs1_np[index_not_both_zero], inputs2_np[index_not_both_zero]
    
    
def get_cos_dis(feature_out_1, feature_out_2):
    from sklearn.metrics.pairwise import cosine_similarity
    len_feature = len(feature_out_1)
    cos_distances = np.zeros([len_feature])
    
    for i in range(len_feature):
        feature1_np = feature_out_1[i].cpu().data.numpy()
        feature2_np = feature_out_2[i].cpu().data.numpy()
        cos_distances[i] = cosine_similarity(feature1_np.reshape([1, -1]) , feature2_np.reshape([1, -1]))
    return cos_distances

def main_backup():
    net1 = 'ResNet152'
    
    # convIndex is the index for submodel conv, starting from 1 
    from .publicVariables import layer_index_value_list
    layer_index_value1 = layer_index_value_list[net1]
    
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    dset_test, dset_database = load_dset_params(job_dataset)
    
    layer_index = layer_index_value1[-2]
    sub_model1 = getConvLayerByIndex(model1, layer_index, net1)
    feature_out1 = sub_model1(inputs)
    feature_np1 = feature_out1.cpu().data.numpy()
    
    net2 = 'ResNext101_32x4d'
    # convIndex is the index for submodel conv, starting from 1 

    layer_index_value2 = layer_index_value_list[net2]
    
    model2, snapshot_path, query_path, database_path = load_net_params(net2)
    dset_test, dset_database = load_dset_params(job_dataset)
    
    layer_index2 = layer_index_value2[-1]
    sub_model2 = getConvLayerByIndex(model2, layer_index2, net2)
    feature_out2 = sub_model2(inputs)
    feature_np2 = feature_out2.cpu().data.numpy()
    
    his_feature1 = np.histogram(feature_np1[feature_np1<=1])
    his_feature2 = np.histogram(feature_np2[feature_np2<=1])
    his_normalized_1 = his_feature1[0].astype(float) / his_feature1[0].sum()
    his_normalized_2 = his_feature2[0].astype(float) / his_feature2[0].sum()
    
    from scipy.stats import wasserstein_distance as emd
    emd1_2 = emd(his_normalized_1, his_normalized_2)
    emd2_1 = emd(his_normalized_2, his_normalized_1)
    print(emd1_2, emd2_1)
    

def emd_analyze():
    histograms1 = get_histograms_by_net(net1, layer_index_value1, inputs1, bins=bins)
    histograms2 = get_histograms_by_net(net2, layer_index_value2, inputs2, bins=bins)
    
    
    emd_AxB = np.zeros([A, B])
    for i in range(A):
        for j in range(B):
            emd_AxB[i,j] = emd(histograms1[i], histograms2[j])
            
    histogramsAB = np.concatenate([histograms1, histograms2])       
    emd_ABxAB = np.zeros([A+B, B+A])
    for i in range(A+B):
        for j in range(B+A):
            emd_ABxAB[i,j] = emd(histogramsAB[i], histogramsAB[j])
            
    emd_A = emd_ABxAB[:, :6]
    emd_B = emd_ABxAB[:, 6:]
    a = emd_A.mean(axis=1)
    b = emd_B.mean(axis=1)
    c = np.stack([a, b], axis=1)
    
    emd_inputs = cal_emd(inputs1, inputs2, bins=bins)
    emd_same = cal_emd(inputs1, inputs1, bins=bins)

def single_experiment():
    #layer_number_list = np.arange(3+4+23+3)
    features1 = {}
    features2 = {}
    l2_matrix = np.zeros([len_1, len_2]) - 1
    cos_matrix = np.zeros([len_1, len_2]) - 1
    emd_matrix = np.zeros([len_1, len_2]) - 1

    layer_number_list = np.arange(len_1)
    for layer_number in layer_number_list:
        sub_model = getConvLayerByNumber(model1, layer_number, net = net1)
        output_np = sub_model(inputs1).cpu().data.numpy()
        features1[layer_number] = output_np
        #print output_np.shape
    
    layer_number_list = np.arange(len_2)
    for layer_number in layer_number_list:
        sub_model = getConvLayerByNumber(model2, layer_number, net = net2)
        output_np = sub_model(inputs2).cpu().data.numpy()
        features2[layer_number] = output_np
        #print output_np.shape
    
    for i in range(len_1):
        for j in range(len_2):
            emd_matrix[i, j] = cal_emd( features1[i],  features2[j])
            if np.array_equal( features1[i].shape,  features2[j].shape):
                l2_matrix[i, j] = np.linalg.norm(features1[i]*255 - features2[j]*255) / features1[i].size
                cos_matrix[i, j] = cosine_similarity(features1[i].reshape([1, -1]), features2[j].reshape([1, -1]))
    if net1 == net2:
        emd_eye = np.array([emd_matrix[i, i] for i in range(len_1)])
        l2_eye = np.array([l2_matrix[i, i] for i in range(len_1)])
        cos_eye = np.array([cos_matrix[i, i] for i in range(len_1)])
        plt.figure(123)
        plt.subplot(3,1,1)
        plt.title('emd')
        plt.plot(emd_eye)
        plt.subplot(3,1,2)
        plt.title('l2')
        plt.plot(l2_eye)
        plt.subplot(3,1,3)
        plt.title('cos')
        plt.plot(cos_eye)
        
    '''
    img_inputs1 = get_trans_img(inputs1.cpu().data[0], job_dataset)
    img_inputs2 = get_trans_img(inputs2.cpu().data[0], job_dataset)
    plt.figure(888)
    plt.subplot(1,2,1)
    plt.imshow(img_inputs1)
    plt.subplot(1,2,2)
    plt.imshow(img_inputs2)
    '''
    return l2_matrix, cos_matrix, emd_matrix
    


def multi_images():
    # multi images
    # set 1-101 as the same classes
    # set 800, 810, 820,..., as the different classes 
    eye_only = True
    bClean = True
    
    inputs1 = Variable(torch.Tensor(dset_database[0][0].unsqueeze(0)).cuda(), requires_grad=True)
    len_same_samples = 100
    
    l2_matrix = np.zeros([len_1, len_2]) - 1
    cos_matrix = np.zeros([len_1, len_2]) - 1
    emd_matrix = np.zeros([len_1, len_2]) - 1

    
    same_l2_matrix = np.zeros([len_same_samples, len_1, len_2]) - 1
    same_cos_matrix = np.zeros([len_same_samples, len_1, len_2]) - 1
    same_emd_matrix = np.zeros([len_same_samples, len_1, len_2]) - 1
    same_emd_eye = np.zeros([len_same_samples, len_1])
    same_l2_eye = np.zeros([len_same_samples, len_1])
    same_cos_eye = np.zeros([len_same_samples, len_1])
    

    # get features1
    features1 = {}
    layer_number_list = np.arange(len_1)
    for layer_number in layer_number_list:
        sub_model = getConvLayerByNumber(model1, layer_number, net = net1)
        output_np = sub_model(inputs1).cpu().data.numpy()
        features1[layer_number] = output_np
        
    for kk in range(len_same_samples):    
        index = kk+1
        print(index)
        inputs2 = Variable(torch.Tensor(dset_database[index][0].unsqueeze(0)).cuda(), requires_grad=True)
        features2 = {}
        layer_number_list = np.arange(len_2)
        for layer_number in layer_number_list:
            sub_model = getConvLayerByNumber(model2, layer_number, net = net2)
            output_np = sub_model(inputs2).cpu().data.numpy()
            features2[layer_number] = output_np
        for i in range(len_1):
            for j in range(len_2):

                if eye_only and net1 == net2: 
                    if i == j:
                        if bClean:
                            feat1, feat2 = clean_both_zero(features1[i], features2[j])
                        else:
                            feat1, feat2 = features1[i], features2[j]
                        emd_matrix[i, j] = cal_emd(feat1,  feat2)
                        
                        l2_matrix[i, j] = np.linalg.norm(feat1*255 - feat2*255) / feat1.size
                        cos_matrix[i, j] = cosine_similarity(feat1.reshape([1, -1]), feat2.reshape([1, -1]))
                else:
                    feat1, feat2 = features1[i], features2[j]
                    emd_matrix[i, j] = cal_emd( feat1,  feat2)
                    if np.array_equal( feat1.shape,  feat2.shape):
                        l2_matrix[i, j] = np.linalg.norm(feat1*255 - feat2*255) / feat1.size
                        cos_matrix[i, j] = cosine_similarity(feat1.reshape([1, -1]), feat2.reshape([1, -1]))
                        
        if net1 == net2:
            emd_eye = np.array([emd_matrix[i, i] for i in range(len_1)])
            l2_eye = np.array([l2_matrix[i, i] for i in range(len_1)])
            cos_eye = np.array([cos_matrix[i, i] for i in range(len_1)])
            same_emd_eye[kk] = emd_eye
            same_l2_eye[kk] = l2_eye
            same_cos_eye[kk] = cos_eye
            
            
    len_diff_samples = 100
    diff_emd_eye = np.zeros([len_diff_samples, len_1])
    diff_l2_eye = np.zeros([len_diff_samples, len_1])
    diff_cos_eye = np.zeros([len_diff_samples, len_1])
    
    for kk in range(len_diff_samples):    
        index = kk*100+800
        print(index)
        inputs2 = Variable(torch.Tensor(dset_database[index][0].unsqueeze(0)).cuda(), requires_grad=True)
        features2 = {}
        layer_number_list = np.arange(len_2)
        for layer_number in layer_number_list:
            sub_model = getConvLayerByNumber(model2, layer_number, net = net2)
            output_np = sub_model(inputs2).cpu().data.numpy()
            features2[layer_number] = output_np
        for i in range(len_1):
            for j in range(len_2):

                if eye_only: 
                    if i == j:
                        if bClean:
                            feat1, feat2 = clean_both_zero(features1[i], features2[j])
                        else:
                            feat1, feat2 = features1[i], features2[j]
                        emd_matrix[i, j] = cal_emd(feat1,  feat2)
                        
                        l2_matrix[i, j] = np.linalg.norm(feat1*255 - feat2*255) / feat1.size
                        cos_matrix[i, j] = cosine_similarity(feat1.reshape([1, -1]), feat2.reshape([1, -1]))
                else:
                    feat1, feat2 = features1[i], features2[j]
                    emd_matrix[i, j] = cal_emd( feat1,  feat2)
                    if np.array_equal( feat1.shape,  feat2.shape):
                        l2_matrix[i, j] = np.linalg.norm(feat1*255 - feat2*255) / feat1.size
                        cos_matrix[i, j] = cosine_similarity(feat1.reshape([1, -1]), feat2.reshape([1, -1]))
                           
        if net1 == net2:
            emd_eye = np.array([emd_matrix[i, i] for i in range(len_1)])
            l2_eye = np.array([l2_matrix[i, i] for i in range(len_1)])
            cos_eye = np.array([cos_matrix[i, i] for i in range(len_1)])
            diff_emd_eye[kk] = emd_eye
            diff_l2_eye[kk] = l2_eye
            diff_cos_eye[kk] = cos_eye
    '''
    plt.figure(123)
    plt.subplot(3,1,1)
    plt.title('emd')
    plt.plot(same_emd_eye.mean(axis=0), label='same')
    plt.plot(diff_emd_eye.mean(axis=0), label='diff')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.title('l2')
    plt.plot(same_l2_eye.mean(axis=0), label='same')
    plt.plot(diff_l2_eye.mean(axis=0), label='diff')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.title('cos')
    plt.plot(same_cos_eye.mean(axis=0), label='same')    
    plt.plot(diff_cos_eye.mean(axis=0), label='diff')   
    plt.legend()
    
    '''
    return same_emd_eye, same_l2_eye, same_cos_eye, diff_emd_eye, diff_l2_eye, diff_cos_eye
    
    
def get_matrix_collection(l2_matrix):
    l2_matrix_collection = {}
    l2_matrix_collection[0] = l2_matrix[0:1, 0:1]    
    l2_matrix_collection[1] = l2_matrix[1:4, 1:4]    
    l2_matrix_collection[2] = l2_matrix[4:12, 4:8]    
    l2_matrix_collection[3] = l2_matrix[12:48, 8:31]
    l2_matrix_collection[4] = l2_matrix[48:, 31:]
    return l2_matrix_collection
    
    
    
def plot_figures():
    # this function can draw 
    # dPsN: different class picture same network
    # cPsN(or sPsN) : same class picture same network
    # dPdN:
    # sPdN:
    # cPdN:
    inputs1 = Variable(torch.Tensor(dset_database[0][0].unsqueeze(0)).cuda(), requires_grad=True)
    inputs2 = Variable(torch.Tensor(dset_database[1000][0].unsqueeze(0)).cuda(), requires_grad=True)
    l2_matrix_dPdN, cos_matrix_dPdN, emd_matrix_dPdN = single_experiment()
    
    inputs2 = Variable(torch.Tensor(dset_database[0][0].unsqueeze(0)).cuda(), requires_grad=True)
    l2_matrix_sPdN, cos_matrix_sPdN, emd_matrix_sPdN = single_experiment()
    
    inputs2 = Variable(torch.Tensor(dset_database[20][0].unsqueeze(0)).cuda(), requires_grad=True)
    l2_matrix_cPdN, cos_matrix_cPdN, emd_matrix_cPdN = single_experiment()
  
        
    plt.figure(123)
    plt.subplot(3,1,1)
    plt.title('L2')
    l2_matrix_sPdN[l2_matrix_sPdN==-1]=100
    plt.plot(diff_l2_eye.mean(axis=0)[5:45], label='dPsN')
    plt.plot(same_l2_eye.mean(axis=0)[5:45], label='sPsN')
    plt.plot(l2_matrix_sPdN.min(axis=1)[5:45], label='sPdN-min')
    plt.plot(np.array([l2_matrix_sPdN[i][l2_matrix_sPdN[i]!=100].mean() for i in range(50)])[5:45], label='sPdN-mean')  

    l2_matrix_dPdN[l2_matrix_dPdN==-1]=100
    plt.plot(l2_matrix_dPdN.min(axis=1)[5:45], label='dPdN-min')
    plt.plot(np.array([l2_matrix_dPdN[i][l2_matrix_dPdN[i]!=100].mean() for i in range(50)])[5:45], label='dPdN-mean')    

    l2_matrix_cPdN[l2_matrix_cPdN==-1]=100
    plt.plot(l2_matrix_cPdN.min(axis=1)[5:45], label='cPdN-min')
    plt.plot(np.array([l2_matrix_cPdN[i][l2_matrix_cPdN[i]!=100].mean() for i in range(50)])[5:45], label='cPdN-mean')    

    plt.legend()
    
    plt.figure(124)
    #plt.subplot(3,1,2)
    plt.title('cos')
    cos_matrix_sPdN[cos_matrix_sPdN==-1]=-100
    plt.plot(diff_cos_eye.mean(axis=0)[5:45], label='dPsN')
    plt.plot(same_cos_eye.mean(axis=0)[5:45], label='sPsN')
    plt.plot(cos_matrix_sPdN.max(axis=1)[5:45], label='sPdN-max')
    plt.plot(np.array([cos_matrix_sPdN[i][cos_matrix_sPdN[i]!=-100].mean() for i in range(50)])[5:45], label='sPdN-mean')  

    cos_matrix_dPdN[cos_matrix_dPdN==-1]=-100
    plt.plot(cos_matrix_dPdN.max(axis=1)[5:45], label='dPdN-max')
    plt.plot(np.array([cos_matrix_dPdN[i][cos_matrix_dPdN[i]!=-100].mean() for i in range(50)])[5:45], label='dPdN-mean')  

    cos_matrix_cPdN[cos_matrix_cPdN==-1]=-100
    plt.plot(cos_matrix_cPdN.max(axis=1)[5:45], label='cPdN-max')
    plt.plot(np.array([cos_matrix_cPdN[i][cos_matrix_cPdN[i]!=-100].mean() for i in range(50)])[5:45], label='cPdN-mean')  

    plt.legend()
    
    plt.figure(125)
    #plt.subplot(3,1,3)
    plt.title('emd')
    #emd_matrix_sPdN[emd_matrix_sPdN==-1]=100
    plt.plot(diff_emd_eye.mean(axis=0)[5:45], label='dPsN')
    plt.plot(same_emd_eye.mean(axis=0)[5:45], label='sPsN')
    emd_matrix_collection = get_matrix_collection(emd_matrix_sPdN)
    plt.plot(np.concatenate([emd_matrix_collection[i].min(axis=1) for i in range(len(emd_matrix_collection))])[5:45], label='sPdN-min')
    plt.plot(np.concatenate([emd_matrix_collection[i].mean(axis=1) for i in range(len(emd_matrix_collection))])[5:45], label='sPdN-mean')  

    emd_matrix_dPdN[emd_matrix_dPdN==-1]=100
    plt.plot(emd_matrix_dPdN.min(axis=1)[5:45], label='dPdN-min')
    plt.plot(np.array([emd_matrix_dPdN[i][emd_matrix_dPdN[i]!=100].mean() for i in range(50)])[5:45], label='dPdN-mean')  

    emd_matrix_cPdN[emd_matrix_cPdN==-1]=100
    plt.plot(emd_matrix_cPdN.min(axis=1)[5:45], label='cPdN-min')
    plt.plot(np.array([emd_matrix_cPdN[i][emd_matrix_cPdN[i]!=100].mean() for i in range(50)])[5:45], label='cPdN-mean')  

    plt.legend()

    
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

    # same net exp
    net1 = 'ResNet152'
    net2 = net1
    #net2 = 'ResNet152'
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    model2, snapshot_path, query_path, database_path = load_net_params(net2)
    
    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    
    from .publicVariables import layer_index_value_list
    layer_index_value1 = layer_index_value_list[net1][:-1]
    layer_index_value2 = layer_index_value_list[net2][:-1]
    A = len(layer_index_value1)
    B = len(layer_index_value2)
    len_1 = 3+8+36+3  if net1 =='ResNet152' else 3+4+23+3
    len_2 = 3+8+36+3  if net2 =='ResNet152' else 3+4+23+3
    # set net2 same as net1
    # same_emd_eye, same_l2_eye, same_cos_eye, diff_emd_eye, diff_l2_eye, diff_cos_eye = multi_images()

    
    

    
    from .publicVariables import layer_index_value_list
    layer_index_value1 = layer_index_value_list[net1][:-1]
    layer_index_value2 = layer_index_value_list[net2][:-1]
    A = len(layer_index_value1)
    B = len(layer_index_value2)
    
    # different net exp
    #net2 = 'ResNet152'
    model1, snapshot_path, query_path, database_path = load_net_params(net1)

    
    
    inputs1 = Variable(torch.Tensor(dset_database[0][0].unsqueeze(0)).cuda(), requires_grad=True)
    inputs2 = Variable(torch.Tensor(dset_database[100][0].unsqueeze(0)).cuda(), requires_grad=True)
    inputs2 = inputs1
    #inputs2 = adv__
    
    old_feature_layer = False
    if old_feature_layer:
        feature_out_1 = get_features_by_net(net1, layer_index_value1, inputs1)
        feature_out_2 = get_features_by_net(net2, layer_index_value2, inputs2)
        l2_distances, l2_distances_avg = get_l2_dis(feature_out_1, feature_out_2)
        cos_distances = get_cos_dis(feature_out_1, feature_out_2)
        
    l2_inputs, l2_inputs_avg = cal_l2(inputs1, inputs2)
    cos_inputs = cal_cos(inputs1, inputs2)
    emd_inputs = cal_emd(inputs1, inputs2, bins=10)
    print(l2_inputs_avg, cos_inputs[0], emd_inputs)
    
    len_1 = 3+8+36+3  if net1 =='ResNet152' else 3+4+23+3
    len_2 = 3+8+36+3  if net2 =='ResNet152' else 3+4+23+3
    
    # single images experiments
    #single_experiment()
    
    # 
    #plot_figures()