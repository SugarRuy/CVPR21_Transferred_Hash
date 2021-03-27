# -*- coding: utf-8 -*-
import math

import scipy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pre_process as prep
import torch.utils.data as util_data
from data_list import ImageList
from torch.autograd import Variable


import pickle
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from publicFunctions import NetworkSettings


def cal_intra_class_dis(code, multi_label_one_hot):
    num_class = multi_label_one_hot.shape[1]
    avg_dis_class = np.zeros([num_class, num_class])
    # hamming_dis_stat: 
    # mean, min, max, var 
    hamming_dis_stat = np.zeros([num_class, 4])
    hamming_dis_class = []
    class_index = []
    # optional: let all multi-label imgs invisible
    #database_label[database_label.sum(axis=1)>1] = 0
    print("Number of classes: ", num_class)
    for i in range(num_class):
        class_i_index = np.argwhere(multi_label_one_hot[:, i]==1).reshape([-1])
        class_index.append(class_i_index)
        class_i_code = code[class_i_index, :]

        for j in range(num_class):
            class_j_index = np.argwhere(multi_label_one_hot[:, j]==1).reshape([-1])
            class_j_code = code[class_j_index, :]
            hamming_dis = np.matmul(class_i_code, class_j_code.transpose()) * (-0.5) + 24
            if i == j:
                hamming_dis_class.append(hamming_dis)

            avg_dis_class[i, j] = np.mean(hamming_dis)
    return avg_dis_class, hamming_dis_stat, class_index

def get_retrieval_result_by_query_code(query_code, database_code, threshold=2, hashbit=48):
    query_size = query_code.shape[0]
    query_result = []
    hamming_dis = np.matmul(query_code, database_code.transpose()) * -0.5 + hashbit / 2
    for i in range(query_size):
        matched_index = np.argwhere(hamming_dis[i]<=threshold)
        query_result.append(matched_index)
    return np.array(query_result)

def get_retrieval_result_by_query_code_topN(query_code, database_code, topN=500, hashbit=48):
    query_size = query_code.shape[0]
    query_result = np.zeros([query_size, topN])
    hamming_dis = np.matmul(query_code, database_code.transpose()) * -0.5 + hashbit / 2
    for i in range(query_size):
        topN_index = np.argsort(hamming_dis[i])[:topN]
        query_result[i] = topN_index
    return query_result.astype(int)

def show_img(path):
    img = plt.imread(path)
    plt.imshow(img)
    
def get_multi_labels_from_vector(multi_label):
    img_num = multi_label.shape[0]
    multi_labels = []
    for i in range(img_num):
        multi_labels.append(np.argwhere(multi_label[i,:]==1))
    return multi_labels

def plot_by_result(query_result_with_index, mnist_train):
    for i in range(min(query_result_with_index.size, 200)):
        img_index = query_result_with_index[i][0]
        plt.subplot(10,20,i+1)
        plt.imshow(np.asarray(mnist_train[img_index][0]))
        
def count_by_query_result(query_result, multi_label):
    img_num = query_result.shape[0]
    img_num_by_class = np.zeros([img_num, int(multi_label.max()+1)] )
    if len(query_result.shape) == 3:
        if query_result.size == 0:
            return np.zeros([img_num, int(multi_label.max())+1])
        query_result = query_result.reshape([-1, query_result.shape[1]])

    for i in range(img_num):
        for j in range(int(multi_label.max()+1)):
            img_num_by_class[i, j] = np.argwhere(multi_label[query_result[i]] == j).shape[0]
            #print img_num_by_class[i, j] 
    return img_num_by_class


def get_query_code_batch(img_inputs, model, batch_size = 8):
    hash_bit = 48
    query_code = np.zeros([img_inputs.shape[0], hash_bit])
    i = 0
    while batch_size * i + batch_size < img_inputs.shape[0]:
        query_code[batch_size*i:batch_size*i+batch_size] = np.sign(model(img_inputs[batch_size*i:batch_size*i+batch_size]).data.cpu().numpy())
        i += 1
    query_code[batch_size*i:] = np.sign(model(img_inputs[batch_size*i:]).data.cpu().numpy())

    return query_code


def get_query_result_num_by_class(query_code, database_code, multi_label, threshold = 10):
    query_result = get_retrieval_result_by_query_code(query_code, database_code, threshold=threshold)
    img_num_by_class = count_by_query_result(query_result, multi_label)
    return img_num_by_class

def get_query_result_num_by_class_topN(query_code, database_code, multi_label, topN = 500):
    query_result = get_retrieval_result_by_query_code_topN(query_code, database_code, topN=topN)
    img_num_by_class = count_by_query_result(query_result, multi_label)
    return img_num_by_class

def get_img_num_by_class_from_img(img, model, database_code, multi_label, threshold = 10):
    oCodeValue = np.sign(model(img).cpu().data.numpy())
    return get_query_result_num_by_class(oCodeValue, database_code, multi_label, threshold)

def get_img_num_by_class_from_img_batch(imgs, model, database_code, multi_label, threshold = 10, batch_size = 32):
    class_num = int(multi_label.max()+1)
    if len(imgs.shape) == 4:
        img_num_by_class = np.zeros([imgs.shape[0], class_num])
        i = 0
        while batch_size*i+batch_size < imgs.shape[0]:
            img_num_by_class[batch_size*i:batch_size*i+batch_size] = get_img_num_by_class_from_img(imgs[batch_size*i:batch_size*i+batch_size], model, database_code, multi_label, threshold = threshold)
            i += 1
        img_num_by_class[batch_size*i:] =  get_img_num_by_class_from_img(imgs[batch_size*i:], model, database_code, multi_label, threshold = threshold)
            
    elif len(imgs.shape) == 5:
        img_num_by_class = np.zeros([imgs.shape[0], imgs.shape[1], class_num])
        i,j = 0,0
        for j in range(imgs.shape[1]):
            i = 0
            while batch_size*i+batch_size < imgs.shape[0]:
                img_num_by_class[batch_size*i:batch_size*i+batch_size, j] = get_img_num_by_class_from_img(imgs[batch_size*i:batch_size*i+batch_size, j], model, database_code, multi_label, threshold = threshold)
                i += 1
            img_num_by_class[batch_size*i:, j] =  get_img_num_by_class_from_img(imgs[batch_size*i:, j], model, database_code, multi_label, threshold = threshold)
    else:
        raise ValueError('ValueError: Images should be a 4-D or 5-D array')
    return img_num_by_class
        
def get_targeted_from_all_class(img_num_by_class, label):
    '''
    # might have bugs here, manually roll-back to former version
    if not np.all(img_num_by_class.shape[:2]==label.shape):
        if img_num_by_class.shape[0] == 1:
            label = np.expand_dims(label, axis=0)
        else:
            raise ValueError("Variable dimensions not match. %s!=%s"%(str(img_num_by_class.shape[:2]), str(label.shape)))
    '''
    if len(img_num_by_class.shape) == 3:
        i_max, j_max = img_num_by_class.shape[0], img_num_by_class.shape[1]
        img_num_targeted = np.zeros([i_max, j_max])-1
        for i in range(i_max):
            for j in range(j_max):
                img_num_targeted[i,j] = img_num_by_class[i,j, int(label[i,j])]
                #print i,j
    elif len(img_num_by_class.shape) == 2:
        j_max = img_num_by_class.shape[0]
        img_num_targeted = np.zeros([j_max])-1
        for j in range(j_max):
            img_num_targeted[j] = img_num_by_class[j, int(label[j])]
    return img_num_targeted
    
def get_img_num_by_class_from_img_topN(img, model, database_code, multi_label, topN = 500):
    oCodeValue = np.sign(model(img).cpu().data.numpy())
    return get_query_result_num_by_class_topN(oCodeValue, database_code, multi_label, topN)


def get_img_result_label_flags(img, model, database_code, multi_label, threshold = 10):
    oCodeValue = np.sign(model(img).cpu().data.numpy())
    distance_to_database = np.linalg.norm(database_code - oCodeValue, ord = 1, axis=1) / 2
    result_label = multi_label[distance_to_database<threshold].astype(int)
    img_result_label_flags = np.zeros([int(multi_label.max()+1)])
    for i in range(result_label.shape[0]):
        img_result_label_flags[result_label[i]] = 1
    return img_result_label_flags


def get_hashcode_from_img(img, model, batch_size=16):
    n_batches = math.ceil(img.shape[0] / batch_size)
    hashcode_list = []
    with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
        for i in range(n_batches):
            img_batch = img[i * batch_size:(i + 1) * batch_size]
            hashcode = model(img_batch).cpu().numpy()
            hashcode_list.append(hashcode)
            hashcode = np.vstack(hashcode_list)
    return np.sign(hashcode)

class HashHammingRetrievalSystem():
    # Wrap a hashnet model as a retrieval system model
    # Hamming ball

    def __init__(self, hashmodel, database_code, database_loader, threshold, batch_size):

        self.hashmodel = hashmodel
        self.database_code = database_code
        self.database_size = database_code.shape[0]
        self.hash_bit = database_code.shape[1]
        self.database_loader = database_loader
        self.threshold = threshold
        self.batch_size = batch_size

    def retrieval_hamming_ball(self, x):
        hashcode = get_hashcode_from_img(x, self.hashmodel, batch_size=self.batch_size)
        #hamm_dist = np.matmul(query_code, self.database_code.transpose()) * (-0.5) + self.hash_bit / 2
        #returned_index = np.argpartition(hamm_dist, self.K, axis=-1)
        returned_index_list = get_retrieval_result_by_query_code(hashcode, self.database_code, threshold=self.threshold, hashbit=self.hash_bit)
        return returned_index_list

    def retrieval_hamming_ball_code(self, hashcode, threshold=5):
        # Do not make the query size too big!
        # 500 is fine, 5k should be fine, but 130k will not.
        returned_index_list = get_retrieval_result_by_query_code(hashcode, self.database_code, threshold=threshold,
                                                                 hashbit=self.hash_bit)
        return returned_index_list


class RetrievalEvaluation():
    # Used to evaluate the results of retrieval.
    def __init__(self, retri_sys, database_label):
        self.retri_sys = retri_sys
        self.database_label = database_label

    def count_labels(self, query_result):
        return np.array([self.database_label[query_result[i]].shape[0] for i in range(len(query_result))])

    def cal_query_precision(self, query_result, query_label):
        query_size = len(query_result)
        query_precisions = np.zeros([query_size])
        for i in range(query_size):
            result_index_i = query_result[i]
            if result_index_i.size == 0:
                continue
            ground_true_label_i = query_label[i]
            result_label_i = database_label[result_index_i]
            matched_index_i = (result_label_i==ground_true_label_i)
            query_precisions[i] = matched_index_i.sum().astype(float) / result_label_i.size

        query_precisions = np.nan_to_num(query_precisions)
        return query_precisions

    def cal_raw_success_rate(self, query_result, target_label):
        raise NotImplementedError




if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyRetrieval')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='imagenet', help="dataset name")
    parser.add_argument('--net', type=str, default='SEResNet50', help="base network type")
    parser.add_argument('--th', type=int, default=5, help="retrieval_threshold")
    parser.add_argument('--mode', type=str, default='test', help="test or database")
    parser.add_argument('--snapshot_iter', type=int, default=-1, help="number of iterations the model has")

    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    retrieval_threshold = args.th
    query_mode = args.mode
    job_dataset = args.dataset
    net = args.net #'Inc_v3'#'ResNext101_32x4d' VGG19BN
    hash_bit = args.hash_bit
    R = 500
    if args.snapshot_iter == -1:
        from publicVariables import iters_list
        snapshot_iter = iters_list[net]
    else:
        snapshot_iter = args.snapshot_iter
    print(args)

    network_settings = NetworkSettings(job_dataset, hash_bit, net, snapshot_iter, batch_size=16)
    model = network_settings.get_model()
    dset_loaders = network_settings.get_dset_loaders()

    output, code, database_label = network_settings.get_out_code_label(part='database')
    query_output, query_code, query_label = network_settings.get_out_code_label(part='test')

    if query_mode == 'database':
        import random
        np.random.seed(0)
        random_index = np.random.randint(0, code.shape[0], int(query_code.shape[0] / 10))
        query_output, query_code, query_label = output[random_index], code[random_index], database_label[random_index]
    else:
        import random
        np.random.seed(0)
        random_index = np.random.randint(0, query_code.shape[0], int(query_code.shape[0] / 10))
        query_output, query_code, query_label = query_output[random_index], query_code[random_index], query_label[random_index]
        # when it is ImageNet, use C=100
    #multi_label_one_hot = make_one_hot(database_label, C=int(database_label.max()+1))
    #avg_dis_class, hamming_dis_class, class_index = cal_intra_class_dis(code[:], multi_label_one_hot[:])

    retri_sys = HashHammingRetrievalSystem(model, code, dset_loaders['database'], retrieval_threshold, batch_size=16)
    query_result = retri_sys.retrieval_hamming_ball_code(query_code)

    retri_eval = RetrievalEvaluation(retri_sys, database_label)

    #query_result = get_retrieval_result_by_query_code(query_code[:], code, retrieval_threshold)
    
    #query_result_count = np.array([database_label[query_result[i]].shape[0] for i in range(len(query_result))])
    query_result_count = retri_eval.count_labels(query_result)

    #query_precisions = np.array([np.sum(database_label[query_result[i]] == query_label[i]).astype(float) / database_label[query_result[i]].size for i in range(len(query_result))])
    #query_precisions = np.nan_to_num(query_precisions)
    query_precisions = retri_eval.cal_query_precision(query_result, query_label)

    print("Query Precision:", query_precisions.mean())
    a = np.array([database_label[query_result[i]].size for i in range(len(query_result))])
    print("Number of Query Returning No Result:", (a == 0).sum())

    weighted_query_precision = np.sum(query_result_count.astype(float) * query_precisions) / query_result_count.sum()
    print("Weighted Precision:", weighted_query_precision)

    
    
    
    