# -*- coding: utf-8 -*-

# This file is to get the adv images for AD_attack jobs..

import scipy as sp
import numpy as np
import torch
from torch.autograd import Variable

from myExpForPapers_nag import EXPSettings

from myGetAdvFeature import getConvLayerByNumber
from publicFunctions import load_net_inputs, load_net_params, load_dset_params, NetworkSettings
from myGetAdvFeature import iFGSMTargetLayerFeatures, iFGSMTargetFeatures


def get_adv_by_method(model1, img_t, targetFeature, targetCode, adv_method, sub_model, step, linf):
    # choose the attack method of adv_method variable
    # adv_method: name of the adversarial method
    # step: step size
    # linf: maximum l_infinite difference
    if adv_method == 'iFGSMF':
        adv__ = iFGSMTargetFeatures(model1, img_t, targetFeature, targetCode, eps=step/255, l_inf_max=linf)
    if adv_method == 'iFGSMLF':
        adv__ = iFGSMTargetLayerFeatures(model1, img_t, targetFeature, targetCode, sub_model, eps=step/255, l_inf_max=linf)


    return adv__

if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpGetAdvVulnerable')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='iFGSMF', help="adv method")
    parser.add_argument('--net1', type=str, default='ResNet101', help="net1")
    parser.add_argument('--net2', type=str, default='ResNext101_32x4d', help="net2")
    parser.add_argument('--step', type=float, default=1.0, help="step size")
    parser.add_argument('--linf', type=int, default=32, help="linf(FGSM) or the maximum step(mostly used)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    adv_method = args.adv_method
    net1 = args.net1
    net2 = args.net2

    job_dataset = 'imagenet'
    threshold = 5
    hash_bit = 48


    dis_method = args.dis_method
    i_max = 64
    j_max = 32
    step = args.step
    linf = args.linf

    # following segment sets the ImageNet images as the inputs
    dset_test, dset_database = load_dset_params(job_dataset)
    from publicVariables import iters_list
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    network_settings1 = NetworkSettings(job_dataset, hash_bit, net1, snapshot_iter=iters_list[net1], batch_size=16)
    model1 = network_settings1.get_model()
    dset_loaders = network_settings1.get_dset_loaders()
    output, code, multi_label = network_settings1.get_out_code_label(part='database')
    _, code_test, multi_label_test = network_settings1.get_out_code_label(part='test')
    
    network_settings2 = NetworkSettings(job_dataset, hash_bit, net2, snapshot_iter=iters_list[net2], batch_size=16)
    model2 = network_settings2.get_model()
    _, code2, multi_label2 = network_settings2.get_out_code_label(part='database')
    _, code_test2, _ = network_settings2.get_out_code_label(part='test')

    exp_settings = EXPSettings(net1, net2, dis_method, i_max, j_max, step=step, linf=linf)
    i_index_set, j_index_matrix = exp_settings.cal_index_set_matrix_white(code_test, code, multi_label)
    inputs_ori_tensor = exp_settings.cal_inputs_ori_tensor(dset_test=dset_loaders['test'].dataset)
    i_index_set = i_index_set.astype(int)
    test_true_id_x = exp_settings.test_true_id_x
    test_true_label_y = exp_settings.test_true_label_y





    early_stop = False

    
    layer_number = 51
    if adv_method == 'iFGSMF':
        sub_model = model1.feature_layers
    else:
        sub_model = getConvLayerByNumber(model1, layer_number, net1)

    bLoad = True

    if 'iFGSMLF' in adv_method:
        npy_name = '/%s_imgs_layerNum_%d_step%1.1f_linf%d_%dx%d.npy' % (
        adv_method, layer_number, step, linf, i_max, j_max)
        npy_path = 'save_for_load/' + net1 + npy_name
    else:
        npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
        npy_path = 'save_for_load/' + net1 + npy_name
    
    if not os.path.exists('save_for_load/' + net1):
        os.makedirs('save_for_load/' + net1)
        
    if os.path.exists(npy_path):
        bLoad = True
    else:
        bLoad = False

    
    if not bLoad:
        adv_imgs = np.zeros([i_max, j_max, 3, 224, 224])
        for i in range(i_max):
            i_index = int(test_true_id_x[i_index_set[i]])
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])]

            img_t = dset_test[i_index][0]
            for j in range(j_max):
                j_index = int(j_index_set[j])
                img_targeted = dset_database[j_index][0]

                X = np.array(img_targeted.unsqueeze(0))
                inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
                targetCode = code[j_index]
                targetFeature = sub_model(inputs)

                print("i:%d,j:%d" % (i, j))
                print("i_index:%d,j_index:%d" % (i_index, j_index))

                adv__ = get_adv_by_method(model1, img_t, targetFeature, targetCode, adv_method, sub_model, step, linf)

                adv_imgs[i, j] = adv__.cpu().data.numpy()

        np.save(npy_path, adv_imgs)
        print("Save adv_imgs file to: %s" % (npy_path))
    else:
        adv_imgs = np.load(npy_path)



    
