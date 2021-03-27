# -*- coding: utf-8 -*-

# This file is to get the adv images for AD_attack jobs..
#
import numpy as np
import torch
import torch.nn as nn

from myExpForPapers_nag import EXPSettings
from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method, get_adv_by_method
from myGetAdvVulnerable import get_unique_index
from data_list import default_loader
from torch.autograd import Variable

import matplotlib.pyplot as plt

from myExtractCodeLabel import trans_train_resize_imagenet
from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class

from myGetAdvFeature import getConvLayerByNumber

from publicFunctions import load_net_inputs, load_net_params, load_dset_params, NetworkSettings

from myFeatureAnalyze import cal_cos, cal_emd, cal_l2
from publicFunctions import model_np_batch

from myRetrieval import get_img_num_by_class_from_img_batch
from myRetrieval import get_targeted_from_all_class
from myGetAdvVulnerable import estimate_subspace_size
from myMapSelection import getRestLayerByNumber



def func_enable_retrieval():
    # this is a function segment
    for i in range(i_max):
        print('id:%d' % (i))
        i_index = int(test_true_id_x[i_index_set[i]])
        inputs_ori = Variable(inputs_ori_tensor.cuda())[i].unsqueeze(0)
        inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())[i].unsqueeze(0)
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)

        label_targeted = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        label2_targeted = np.array([multi_label2[j_index_set[j]] for j in range(j_max)])
        if not bSaveBlackTargetedNum:
            X = np.stack([dset_database[j_index_set[j]][0] for j in range(j_max)])
            inputs_target = Variable(torch.Tensor(X).cuda(), requires_grad=True)

            # get the target's retrieval result fget_img_num_by_class_from_img_batchor each class on White and Black
            img_num_by_class_target = get_img_num_by_class_from_img_batch(inputs_target, model1, code, multi_label,
                                                                          threshold=threshold, batch_size=16)
            img_num_by_class_target_black = get_img_num_by_class_from_img_batch(inputs_target, model2, code2,
                                                                                multi_label2, threshold=threshold,
                                                                                batch_size=8)

            # get the adv's retrieval result for each class on White and Black
            img_num_by_class_adv = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label,
                                                                       threshold=threshold, batch_size=16)
        img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2,
                                                                         threshold=threshold, batch_size=8)
        if not bSaveBlackTargetedNum:
            # get the ori's retrieval result for each class on Black
            img_num_by_class_ori_black = get_img_num_by_class_from_img_batch(inputs_ori, model2, code2, multi_label2,
                                                                             threshold=threshold, batch_size=8)

            # get the target's retrieval result for targeted class only on White and Black
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_target, np.expand_dims(label_targeted, 0))
            img_num_target_black_targeted = get_targeted_from_all_class(img_num_by_class_target_black, np.expand_dims(label2_targeted, 0))

            # get the adv's retrieval result for targeted class only on White and Black
            img_num_adv_targeted = get_targeted_from_all_class(img_num_by_class_adv, np.expand_dims(label_targeted, 0))
        print(img_num_by_class_adv_black.shape)
        img_num_adv_black_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, np.expand_dims(label2_targeted, 0))
        if not bSaveBlackTargetedNum:
            # get the ori's retrieval result for targeted class only on Black
            img_num_by_class_ori_black_targeted = get_targeted_from_all_class(img_num_by_class_ori_black,
                                                                              np.expand_dims(label2_targeted, 0))


        # GUIDE:
        # Compare img_num_adv_black_targeted with img_num_target_targeted,
        # if one item in img_num_target_targeted is high enough, ignore it.
        # if we found one item has a great difference, we succeed.
        if not bSaveBlackTargetedNum:
            print(adv_method + ":")

            print("WhiteBox(%d imgs overall):" % (1 * j_max))
            print("", img_num_adv_targeted.sum(), (img_num_adv_targeted > 0).sum())

            print("BlackBox(%d imgs overall):" % (1 * j_max))
            print("", img_num_adv_black_targeted.sum(), (img_num_adv_black_targeted > 0).sum())



            code_adv_black = np.sign(model_np_batch(model2, inputs_adv, batch_size=8))
            code_ori_black = code_test2[i_index]
            code_targeted_black = code2[j_index_set]

            #
            code_ori_white = code_test[i_index]
            code_targeted_white = code[j_index_set]
            whiteHammingMatrix[i] = np.transpose(np.linalg.norm(code_ori_white - code_targeted_white, ord=0, axis=-1))
        blackTargetedNumMatrix[i] = img_num_adv_black_targeted
        if not bSaveBlackTargetedNum:
            code_diff_adv_target = np.transpose(np.linalg.norm(code_adv_black - code_targeted_black, ord=0, axis=-1))
            code_diff_ori_adv = np.linalg.norm(np.swapaxes(code_adv_black, 0, 1) - code_ori_black, ord=0, axis=-1)
            code_diff_ori_target = np.array(
                [np.transpose(np.linalg.norm(code_ori_black - code_targeted_black[j], ord=0, axis=-1)) for j in
                 range(j_max)])

            print(code_diff_adv_target.mean())
            print(code_diff_ori_adv.mean())
            print(code_diff_ori_target.mean())

            succeed_index = np.where(img_num_adv_black_targeted > 0)
            print(img_num_adv_black_targeted[img_num_adv_black_targeted > 0].astype(int))
            print(succeed_index[0], '\n', succeed_index[1])

            oriBlackCountMatrix[i] = img_num_by_class_ori_black_targeted[0]
            whiteMatrix[i][0], whiteMatrix[i][1] = img_num_adv_targeted.sum(), (img_num_adv_targeted > 0).sum()
            blackMatrix[i][0], blackMatrix[i][1] = img_num_adv_black_targeted.sum(), (
                    img_num_adv_black_targeted > 0).sum()
            distanceMatrix[i, 0], distanceMatrix[i, 1], distanceMatrix[i, 2] = \
                code_diff_adv_target.mean(), code_diff_ori_adv.mean(), code_diff_ori_target.mean()
            targetCountMatrix[i] = img_num_adv_black_targeted[img_num_adv_black_targeted > 0].astype(int)
            succeedIndexXMatrix[i], succeedIndexYMatrix[i] = succeed_index[0], succeed_index[1]
    if not bSaveBlackTargetedNum:
        print("dis_method:%s" % (dis_method))
        print("retrieval num of ori in blackbox:\n", oriBlackCountMatrix.astype(int))
        print("retrieval num and sample size of adv in whitebox:\n", whiteMatrix.astype(int).transpose())
        print("retrieval num and sample size of adv in blackbox:\n", blackMatrix.astype(int).transpose())
        print("distanceMatrix of adv in blackbox:\n", distanceMatrix.transpose())
        print("attack percentage(i-level, white and black):%f,%f" % (
            float((whiteMatrix[:, 1] > 0).sum()) / i_max, float((blackMatrix[:, 1] > 0).sum()) / i_max))
        print("attack percentage(i*j-level, white and black):%f,%f" % (
            float((whiteMatrix[:, 1]).sum()) / i_max / j_max, float((blackMatrix[:, 1]).sum()) / i_max / j_max))

    if bSaveWhiteHamming:
        np.save(path_whiteHamming, whiteHammingMatrix)
        print('Save white hamming distance matrix file to: %s'%(path_whiteHamming))
    if bSaveBlackTargetedNum:
        np.save(path_blackTargetedNum, blackTargetedNumMatrix)
        print('Save black targeted number file to: %s'%(path_blackTargetedNum))

def get_adv_imgs():
    adv_imgs = np.zeros([i_max, j_max, 3, 224, 224])
    for i in range(i_max):
        i_index = int(test_true_id_x[i_index_set[i]])
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])]

        if noiid:
            ad_image_num = ad_imagelist[i_index]
            ad_imagepath = ad_datapath + ad_image_num
            img = default_loader(ad_imagepath)
            t = trans_train_resize_imagenet()
            img_t = t(img)
        else:
            img_t = dset_test[i_index][0]

        for j in range(j_max):
            j_index = int(j_index_set[j])

            X = np.array(img_t.unsqueeze(0))
            inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
            targetCode = code[j_index]

            print("i:%d,j:%d" % (i, j))
            print("i_index:%d,j_index:%d" % (i_index, j_index))

            adv__ = get_adv_by_method(model1, img_t, targetCode, adv_method, step, linf, bShowProcess=True)

            adv_imgs[i, j] = adv__.cpu().data.numpy()

    np.save(npy_path, adv_imgs)
    print("Save adv_imgs file to: %s"%(npy_path))
    return adv_imgs

def get_npy_name_path_by_noiid(noiid):
    if noiid:
        if 'iFGSMLF' in adv_method:
            npy_name = '/%s_imgs_layerNum_%d_step%1.1f_linf%d_%dx%d_noiid_%s.npy' % (
                adv_method, layer_number, step, linf, i_max, j_max, dis_method)
            npy_path = 'save_for_load/' + net1 + npy_name
        else:
            npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_noiid_%s.npy' % (
                adv_method, step, linf, i_max, j_max, dis_method)
            npy_path = 'save_for_load/' + net1 + npy_name
    else:
        if 'iFGSMLF' in adv_method:
            npy_name = '/%s_imgs_layerNum_%d_step%1.1f_linf%d_%dx%d_%s.npy' % (
                adv_method, layer_number, step, linf, i_max, j_max, dis_method)
            npy_path = 'save_for_load/' + net1 + npy_name
        else:
            npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
            npy_path = 'save_for_load/' + net1 + npy_name
    return npy_name, npy_path


def func_enable_retrieval_class_only():
    global i
    num_classes = 100
    path_white_retrieval_num_by_class = './save_for_load/distanceADVRetrieval/%s/retrieval_num_by_class_white_%s_black_%s_%s.npy' % (
    adv_method, net1, net1, dis_method)
    path_black_retrieval_num_by_class = './save_for_load/distanceADVRetrieval/%s/retrieval_num_by_class_white_%s_black_%s_%s.npy' % (
    adv_method, net1, net2, dis_method)
    if os.path.exists(path_black_retrieval_num_by_class):
        img_num_by_class_adv_mat = np.load(path_white_retrieval_num_by_class)
        img_num_by_class_adv_black_mat = np.load(path_black_retrieval_num_by_class)
    else:
        img_num_by_class_adv_mat = np.zeros([i_max, j_max, num_classes])
        img_num_by_class_adv_black_mat = np.zeros([i_max, j_max, num_classes])
        for i in range(i_max):
            print('id:%d' % (i))
            i_index = int(test_true_id_x[i_index_set[i]])
            inputs_ori = Variable(inputs_ori_tensor.cuda())[i].unsqueeze(0)
            inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())[i].unsqueeze(0)
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)

            label_targeted = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
            label2_targeted = np.array([multi_label2[j_index_set[j]] for j in range(j_max)])
            X = np.stack([dset_database[j_index_set[j]][0] for j in range(j_max)])
            inputs_target = Variable(torch.Tensor(X).cuda(), requires_grad=True)

            # get the adv's retrieval result for each class on White and Black
            img_num_by_class_adv = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label,
                                                                       threshold=threshold, batch_size=16)
            img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2,
                                                                             threshold=threshold, batch_size=8)
            img_num_by_class_adv_mat[i] = img_num_by_class_adv
            img_num_by_class_adv_black_mat[i] = img_num_by_class_adv_black
        np.save(path_white_retrieval_num_by_class, img_num_by_class_adv_mat)
        np.save(path_black_retrieval_num_by_class, img_num_by_class_adv_black_mat)
    label_ori = multi_label_test[test_true_id_x[i_index_set.astype(int)]]
    label_ori_mat = np.repeat(label_ori, j_max).reshape([i_max, j_max])
    label_ori_target_mat = np.zeros([i_max, j_max])
    for i in range(i_max):
        i_index = int(test_true_id_x[i_index_set[i]])
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_ori_target_mat[i] = np.array([multi_label2[j_index_set[j]] for j in range(j_max)])
    label_by_result_adv_white = np.argmax(img_num_by_class_adv_mat, axis=-1)
    index_no_result_white = img_num_by_class_adv_mat.sum(axis=-1) == 0
    label_by_result_adv_white[index_no_result_white] = -1
    label_by_result_adv_black = np.argmax(img_num_by_class_adv_black_mat, axis=-1)
    index_no_result_black = img_num_by_class_adv_black_mat.sum(axis=-1) == 0
    label_by_result_adv_black[index_no_result_black] = -1
    print(index_no_result_white.sum(), index_no_result_black.sum())
    matched_mat_white = label_ori_mat == label_by_result_adv_white
    matched_mat_black = label_ori_mat == label_by_result_adv_black
    matched_mat_white_black = label_by_result_adv_white == label_by_result_adv_black
    matched_mat_white_target = label_by_result_adv_white == label_ori_target_mat
    matched_mat_black_target = label_ori_target_mat == label_by_result_adv_black
    print(matched_mat_white.sum(), matched_mat_black.sum(), matched_mat_white_black.sum(), \
        matched_mat_white_target.sum(), matched_mat_black_target.sum())
    path_untarget_result_id = './save_for_load/distanceADVRetrieval/%s/untargeted_result_white_%s_black_%s_%s.npy' % (
    adv_method, net1, net2, dis_method)
    np.save(path_untarget_result_id, matched_mat_black_target)


def get_noised_result(adv_imgs, ori_imgs, perturbation_ratio=0.25, noise_level=10):

    # imgs_test = np.load('save_for_load/imgs_test.npy')
    # target_img_mat = np.load('save_for_load/target_imgs.npy')
    random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
    adv_imgs_noised = np.stack(
        [(adv_imgs[i] - ori_imgs[i]) * perturbation_ratio + ori_imgs[i] for i in range(i_max)]) + random_noise / 255
    # adv_imgs_noised = adv_imgs

    X = Variable(torch.Tensor(adv_imgs_noised)).cuda()
    noised_img_num_result = get_img_num_by_class_from_img_batch(X, model1, code, multi_label, threshold=5,
                                                                batch_size=16)
    label_targeted = np.zeros([i_max, j_max])
    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        label_targeted[i] = label_targeted_i
    # retrieval_result is a i_max*j_max matrix, which contains the number of targeted imgs of each input images.
    noised_adv_white_retrieval_result = get_targeted_from_all_class(noised_img_num_result, label_targeted)
    return noised_adv_white_retrieval_result


def calculate_estimate_subspace_size():


    ori_imgs = inputs_ori_tensor.cpu().numpy()
    lower_bound_matrix = np.zeros([i_max, j_max])
    upper_bound_matrix = np.zeros([i_max, j_max])

    for i in range(10):
        ori_img = ori_imgs[i]
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted = np.array([multi_label[j_index_set[jj]] for jj in range(j_max)])
        for j in range(j_max):
            print("i=",i,";j=",j)
            adv_img = adv_imgs[i,j]
            target_label = label_targeted[j]
            perturbation_ratio_bound = estimate_subspace_size(adv_img, ori_img, model1, target_label, code, multi_label)
            lower_bound_matrix[i, j] = perturbation_ratio_bound[:, 0].mean()
            upper_bound_matrix[i, j] = perturbation_ratio_bound[:, 1].mean()
    print(lower_bound_matrix, upper_bound_matrix)
    return lower_bound_matrix, upper_bound_matrix

if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpGetAdvVulnerable')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='iFGSM', help="adv method")
    parser.add_argument('--net1', type=str, default='ResNext101_32x4d', help="net1")
    parser.add_argument('--net2', type=str, default='ResNext101_32x4d', help="net2")
    parser.add_argument('--step', type=float, default=1.0, help="step size")
    parser.add_argument('--linf', type=int, default=32, help="linf(FGSM) or the maximum step(mostly used)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    job_dataset = 'imagenet'
    threshold = 5
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = ['ResNet18', 'ResNet34', 'AlexNet', 'ResNet101', 'ResNet152', 'ResNext101_32x4d']
    # 'ResNext101_32x4d', 'VGG19BN', 'DenseNet161'
    adv_method_values = ['FGSM', 'iFGSM', 'iFGSMDI', 'miFGSMDI']
    adv_method = args.adv_method

    # c stands for close, f stands for far
    # W stands for White and B stands for Black
    dis_method_value = ['cW', 'fW', 'cB', 'fB', 'cWcB', 'cWfB', 'fWcB', 'fWfB']
    dis_method = args.dis_method

    i_max = 64
    j_max = 32

    # flllowing segment sets the AD images as the inputs
    ad_datapath = '../data/ad_dataset/ads/0/'

    # following segment sets the ImageNet images as the inputs
    hash_bit = 48

    dset_test, dset_database = load_dset_params(job_dataset)
    # same net exp
    net1 = args.net1
    net2 = args.net2

    from publicVariables import iters_list

    network_settings1 = NetworkSettings(job_dataset, hash_bit, net1, snapshot_iter=iters_list[net1], batch_size=16)
    model1 = network_settings1.get_model()
    dset_loaders = network_settings1.get_dset_loaders()
    output, code, multi_label = network_settings1.get_out_code_label(part='database')
    _, code_test, multi_label_test = network_settings1.get_out_code_label(part='test')

    network_settings2 = NetworkSettings(job_dataset, hash_bit, net2, snapshot_iter=iters_list[net2], batch_size=16)
    model2 = network_settings2.get_model()
    _, code2, multi_label2 = network_settings2.get_out_code_label(part='database')
    _, code_test2, _  = network_settings2.get_out_code_label(part='test')
    # adv
    step = args.step
    linf = args.linf
    early_stop = False
    early_stop_name = 'early' if early_stop else 'noearly'

    exp_settings = EXPSettings(net1, net2, dis_method, i_max, j_max, step=step, linf=linf)
    #i_index_set, j_index_matrix = exp_settings.cal_index_set_matrix(multi_label_test, code_test2, code2, multi_label2,
    #                                                                code_test, code, multi_label)
    i_index_set, j_index_matrix = exp_settings.cal_index_set_matrix_white(code_test, code, multi_label)
    inputs_ori_tensor = exp_settings.cal_inputs_ori_tensor(dset_test=dset_loaders['test'].dataset)
    i_index_set = i_index_set.astype(int)

    test_true_id_x = exp_settings.test_true_id_x
    test_true_label_y = exp_settings.test_true_label_y

    noiid = False
    if noiid:
        ad_imagelist = os.listdir(ad_datapath)

    # some enable flags for function blocks
    bSaveWhiteL2 = False
    bSaveWhiteCos = False
    bSaveWhiteEMD = False
    bSaveWhiteHopping = False
    bSaveWhiteHamming = False
    bNeedClean = False
    bSaveBlackTargetedNum = False # set to True if want to save sth to path_blackTargetedNum.
    bEnableRetrieval = False # set to True if bSaveBlackTargetedNum is True
    bEnableRetrievalClassOnly = False


    bSaveDistOriAdv = False
    bAnaylzeByDistOriAdv = False
    bDoRetrievalAddedRandomNoise = False

    bDoCalculateEstimateSubspaceSize = False

    path_whiteHamming = 'save_for_load/distanceADVRetrieval/hamming_%s.npy' % ( dis_method)
    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_%s.npy' % (adv_method, net1, net2, dis_method)

    path_advGeoDist = 'save_for_load/distanceADVRetrieval/%s_advGeoDist_%s.npy'%(adv_method, dis_method)

    print('Overall size:%d' % (test_true_id_x.shape[0]))

    if 'iFGSMLF' in adv_method:
        layer_number = 51
        sub_model = getConvLayerByNumber(model1, layer_number, net1)
    if 'iFGSMLFW' in adv_method:
        # map_weights = getWeightsByTargetFeat(targetFeature_final)
        sub_model_fore = getConvLayerByNumber(model1, layer_number, net=net1)
        sub_model_rest = getRestLayerByNumber(model1, layer_number, net=net1)

    npy_name, npy_path = get_npy_name_path_by_noiid(noiid)

    if os.path.exists(npy_path):
        bLoad = True
    else:
        bLoad = False

    if not bLoad:
        adv_imgs = get_adv_imgs()
    else:
        adv_imgs = np.load(npy_path)

    oriBlackCountMatrix = np.zeros([i_max])
    whiteMatrix = np.zeros([i_max, 2])
    blackMatrix = np.zeros([i_max, 2])
    distanceMatrix = np.zeros([i_max, 3])
    targetCountMatrix = {}
    succeedIndexXMatrix, succeedIndexYMatrix = {}, {}
    # Calculate the distance and targeted retrieval result for each (source, target) adv.
    whiteHammingMatrix = np.zeros([i_max, j_max])
    blackTargetedNumMatrix = np.zeros([i_max, j_max])

    bEnablePlotDistanceScatter = False


    if bEnableRetrieval:
        func_enable_retrieval()


    if bEnableRetrievalClassOnly:
        func_enable_retrieval_class_only()






    '''
    tmp_inputs = Variable(torch.Tensor(adv_imgs[-1,-1:])).cuda()
    tmp_out = model1(tmp_inputs)
    adv_out = tmp_out.cpu().data.numpy()
    target = code[108759]
    distance = np.linalg.norm(target-adv_out, ord=0, axis=1)
    print(distance)
    '''