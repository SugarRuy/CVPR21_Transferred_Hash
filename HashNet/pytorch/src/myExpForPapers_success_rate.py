# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from torch.autograd import Variable

from myGetAdvVulnerable import get_target_targetedRetrievalNum, get_adv_black_retrieval_result, get_adv_code_diff_to_targeted

def success_rate_once(target_targetedNum_mat, adv_black_retrieval_result, adv_code_diff_to_targeted='', useCodeDiff=False):
    #if adv_code_diff_to_targeted is '':
    if useCodeDiff:
        index_valid_code_diff = adv_code_diff_to_targeted <= 5

        index_valid_gt_100 = (target_targetedNum_mat > 100 ) * index_valid_code_diff
        print("Valid index size: %d" % (index_valid_gt_100.sum()))

        index_black_gt_10 = adv_black_retrieval_result > 10
        print("Black Success Rate(gt10): %f=(%d/%d)" % (
            float(index_black_gt_10.sum()) / index_black_gt_10.size, index_black_gt_10.sum(), index_black_gt_10.size))

        index_valid_success_gt10 = index_valid_gt_100 * index_black_gt_10
        print("Black Success Rate(GT10)--Valid: %f=(%d/%d)" % (
            float(index_valid_success_gt10.sum()) / index_valid_gt_100.sum(), index_valid_success_gt10.sum(),
        index_valid_gt_100.sum()))

        return float(index_black_gt_10.sum()) / index_black_gt_10.size, float(
            index_valid_success_gt10.sum()) / index_valid_gt_100.sum()
    else:
        index_valid_gt_100 = target_targetedNum_mat > 100
        print("Valid index size: %d" %(index_valid_gt_100.sum()))

        index_black_gt_10 = adv_black_retrieval_result > 10
        print("Black Success Rate(gt10): %f=(%d/%d)" % (float(index_black_gt_10.sum()) / index_black_gt_10.size, index_black_gt_10.sum(), index_black_gt_10.size))

        index_valid_success_gt10 = index_valid_gt_100 * index_black_gt_10
        print("Black Success Rate(GT10)--Valid: %f=(%d/%d)" % (float(index_valid_success_gt10.sum()) / index_valid_gt_100.sum(), index_valid_success_gt10.sum(), index_valid_gt_100.sum()))

        return float(index_black_gt_10.sum()) / index_black_gt_10.size, float(index_valid_success_gt10.sum()) / index_valid_gt_100.sum()



def func_get_success_rate_matrix():
    print("linf:", linf)
    net1_values = ['ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'ResNet34', 'DenseNet161']
    #net1_values = ['ResNet101', 'ResNet152', 'ResNext101_32x4d']
    #net1_values = ['SEResNet50', 'ResNet34', 'DenseNet161']
    #net1_values = ['DenseNet161']
    net2_values = ['ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'ResNet34', 'DenseNet161'] #  'SEResNet50',
    #net2_values =['SEResNet50']# ['SEResNet50', 'DenseNet161']

    #adv_method_values = [ 'iFGSM', 'iFGSMDI', 'miFGSMDI', 'NAG']
    #adv_method_values = ['iFGSM', 'miFGSMDI']#['iFGSMDI', 'NAG']
    adv_method_values = [ 'FGSM']
    #dis_method_values = ['cW', 'fW']
    dis_method_values = ['cW']
    success_rate_matrix = np.zeros([len(net1_values), len(net2_values), len(adv_method_values), len(dis_method_values)])
    success_rate_matrix_valid = np.zeros([len(net1_values), len(net2_values), len(adv_method_values), len(dis_method_values)])
    for i0 in range(len(net1_values)):
        for i1 in range(len(net2_values)):
            for i2 in range(len(adv_method_values)):
                for i3 in range(len(dis_method_values)):
                    net1 = net1_values[i0]
                    net2 = net2_values[i1]
                    adv_method = adv_method_values[i2]
                    dis_method = dis_method_values[i3]
                    target_targetedNum_mat = get_target_targetedRetrievalNum(net1, net2, adv_method, step, linf, i_max, j_max, dis_method, job_dataset=job_dataset, allowLoad=allowLoadT)
                    adv_black_retrieval_result = get_adv_black_retrieval_result(net1, net2, adv_method, step, linf,i_max, j_max,dis_method, job_dataset=job_dataset, threshold=5, batch_size=8, allowLoad=allowLoadR)
                    adv_code_diff_to_targeted = get_adv_code_diff_to_targeted(net1, adv_method, step, linf, i_max, j_max, dis_method,  allowLoad=allowLoadD)

                    print('net1:%s, net2:%s, adv_method:%s, dis_method:%s'%(net1, net2, adv_method, dis_method))
                    success_rate_gt10, success_rate_gt10_valid = success_rate_once(target_targetedNum_mat, adv_black_retrieval_result, adv_code_diff_to_targeted, useCodeDiff = useCodeDiff)

                    success_rate_matrix[i0,i1,i2,i3] = success_rate_gt10
                    success_rate_matrix_valid[i0,i1,i2,i3] = success_rate_gt10_valid
    return success_rate_matrix, success_rate_matrix_valid


def print_success_rate_matrix(success_rate_matrix_valid):
    np.set_printoptions(formatter={'float': '{: .1f}'.format})
    for i in range(success_rate_matrix_valid.shape[0]):
        for j in range(success_rate_matrix_valid.shape[1]):
            print(i,j)
            print(np.around(success_rate_matrix_valid[i][j]*100, decimals=1))


def big_matrix(success_rate_matrix_valid_rounded, dis_method = 'cW'):
    #big_matrix = np.zeros([25, 10]) - 1.0
    useMat = True
    if useMat:
        big_matrix = np.zeros([25, 10]) - 1.0
        for i0 in range(success_rate_matrix_valid.shape[0]):
            for i1 in range(success_rate_matrix_valid.shape[1]):
                for i2 in range(success_rate_matrix_valid.shape[2]):
                    for i3 in range(success_rate_matrix_valid.shape[3]):
                        big_matrix[i0*5+i2, i1*2+i3] = success_rate_matrix_valid[i0,i1,i2,i3]
        print(big_matrix)
        return big_matrix
    else:
        big_dict = {}
        for i0 in range(success_rate_matrix_valid_rounded.shape[0]):
            big_dict[i0] = {}
            for i1 in range(success_rate_matrix_valid_rounded.shape[1]):
                big_dict[i0][i1] = success_rate_matrix_valid_rounded[i0, i1]
        return big_dict


def exp_train_loss_analyze(job_dataset):
    # An exp to see the tread of loss in training process
    net1a = 'ResNet152'#'ResNext101_32x4d'
    log_txt_path = '../snapshot/%s_48bit_%s_hashnet/log.txt'%(job_dataset, net1a)
    with open(log_txt_path) as f:
        lines = f.read().splitlines()
    loss_value = np.array([float(lines[ii][-5:]) for ii in range(len(lines))])
    k = 100
    loss_avg_k = np.array([loss_value[ii:ii+k].mean() for ii in range(loss_value.shape[0]-k+1)])
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.plot(loss_value)
    plt.subplot(2,2,2)
    plt.plot(loss_avg_k)
    print(net1a, "Loss std:",loss_value.std())


    net1b = 'ResNet101'#'DenseNet161'#'DenseNet161'
    log_txt_path = '../snapshot/%s_48bit_%s_hashnet/log.txt'%(job_dataset, net1b)
    with open(log_txt_path) as f:
        lines = f.read().splitlines()
    loss_value = np.array([float(lines[ii][-5:]) for ii in range(len(lines))])
    k = 100
    loss_avg_k = np.array([loss_value[ii:ii+k].mean() for ii in range(loss_value.shape[0]-k+1)])
    import matplotlib.pyplot as plt
    plt.subplot(2,2,3)
    plt.plot(loss_value)
    plt.subplot(2,2,4)
    plt.plot(loss_avg_k)
    print(net1b, "Loss std:",loss_value.std())


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpForPapers')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='miFGSMDI', help="adv method")
    parser.add_argument('--net1', type=str, default='ResNext101_32x4d', help="net1") # ResNext101_32x4d
    parser.add_argument('--net2', type=str, default='ResNext101_32x4d', help="net2")
    parser.add_argument('--allowLoad', type=str, default='True', help="is Loading allowed")
    parser.add_argument('--linf', type=int, default=8, help="linf")
    parser.add_argument('--allowLoadT', type=str, default='True', help="is Loading allowed Targeted Retrieval Result")
    parser.add_argument('--allowLoadR', type=str, default='True', help="is Loading allowed Retrieval Result")
    parser.add_argument('--allowLoadD', type=str, default='True', help="is Loading allowed Code Difference")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    adv_method = args.adv_method
    dis_method = args.dis_method
    net1 = args.net1
    net2 = args.net2
    i_max, j_max = 64, 32
    step = 8.0
    linf = args.linf
    job_dataset = 'imagenet'
    noiid = False

    allowLoad = False if args.allowLoad is not 'True' else True
    allowLoadT = False if args.allowLoadT is not 'True' else True
    allowLoadR = False if args.allowLoadR is not 'True' else True
    allowLoadD = False if args.allowLoadD is not 'True' else True

    # whether the code diff to be considered in the index
    useCodeDiff = True
    '''
    target_targetedNum_mat = get_target_targetedRetrievalNum(net1, net2, adv_method, step, linf, i_max, j_max, dis_method, job_dataset=job_dataset)
    adv_black_retrieval_result = get_adv_black_retrieval_result(net1, net2, adv_method, step, linf, i_max, j_max,
                                                            dis_method, job_dataset=job_dataset, threshold=5, batch_size=8)
    # success_rate_gt10, success_rate_gt10_valid = success_rate_once(target_targetedNum_mat, adv_black_retrieval_result)
    '''
    success_rate_matrix, success_rate_matrix_valid = func_get_success_rate_matrix()
    print_success_rate_matrix(success_rate_matrix_valid)
    success_rate_matrix_valid_rounded = np.around(success_rate_matrix_valid, decimals=1)
    #big_dict = big_matrix(success_rate_matrix_valid_rounded)
