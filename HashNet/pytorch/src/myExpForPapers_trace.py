# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method, get_unique_index
from publicFunctions import load_net_inputs, load_net_params, load_dset_params
from myGetAdv import randomly_input_diversity
from torch.autograd import Variable
from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class, get_query_result_num_by_class

import torch.optim as optim
optim_dict = {"SGD": optim.SGD}

def func_targetedAttack_cornell_lambda_trace(model, img_t, targetCode, targetCode_black, model2, eps=1.0 / 255, l_inf_max=32, threshold_stop=2, decay_factor=1.0,
                                t_prob=0.5, bShowProcess=False, bEarlyStop=False, random_noise_level=16, var_lambda = 1.0,
                                noise_distribution='uniform', adv_method='iFGSM', frozenVar=False):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())

    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()
    oCodeValue_black = np.sign(model2(adv).cpu().data.numpy())

    # Trace_data_mat, each column
    # 0: total loss,
    # 1: loss1; 2: loss2; 3: lambda; 4: var_v
    # 5: l1_nosied_to_x_white, 6:l1_noised_to_target_white
    # 7: l1_nosied_to_x_black,
    # 8: l1_adv_to_x_white, 9: l1_adv_to_target_white
    # 10: l1_adv_to_x_black
    # 11: l1_noised_to_target_black, 12: l1_adv_to_target_black
    trace_data_mat = np.zeros([l_inf_max, 13])
    print('...targeted %s begin....'%(adv_method))
    print('initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    i = 0
    grad_var = 0
    '''
    if noise_distribution == 'uniform':
        random_noise = np.random.uniform(-random_noise_level, random_noise_level, adv.shape)
    elif noise_distribution == 'normal' or noise_distribution == 'Gaussian':
        random_noise = np.random.normal(loc=0.0, scale=random_noise_level / 3, size=adv.shape)
        random_noise = np.clip(random_noise, -random_noise_level, random_noise_level)
    '''
    T_h = 5.0
    #var_grad_lambda = Variable(torch.Tensor(np.array([var_lambda])).cuda(), requires_grad=True)
    var_v = np.random.uniform(0, threshold) / 48*2
    #var_grad_v = Variable(torch.Tensor(np.array([var_v])).cuda(), requires_grad=True)
    t = eps*255*10
    #print 'initial var_gar_lambda', var_grad_lambda
    optim_method = 'iFGSM'

    while i < l_inf_max:
        adv_np = adv.cpu().data.numpy()
        adv_np[adv_np<0] = 0
        adv_np[adv_np>1] = 1
        #var_lambda = float(var_grad_lambda.cpu().data.numpy()[0])
        #var_grad_v.zero_grad()
        # inputs_adv is used as the input of model;
        # adv is used as the adv result of iter i.
        # They are different when using Diversity Inputs method

        if adv_method == 'iFGSM' or adv_method == 'miFGSM':
            inputs_adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
        elif adv_method == 'iFGSMDI' or adv_method == 'miFGSMDI':
            inputs_adv = randomly_input_diversity(adv_np[0], p=t_prob)

        output = model(inputs_adv)
        loss = nn.L1Loss()
        loss1 = loss(output, targetCode.detach())

        if noise_distribution == 'uniform':
            random_noise = np.random.uniform(-random_noise_level, random_noise_level, adv.shape)
        elif noise_distribution == 'normal' or noise_distribution == 'Gaussian':
            random_noise = np.random.normal(loc=0.0, scale=random_noise_level / 3, size=adv.shape)
            random_noise = np.clip(random_noise, -random_noise_level, random_noise_level)

        adv_intermediate_noised = inputs_adv + Variable(torch.Tensor(random_noise).cuda())
        output_noised = model(adv_intermediate_noised)

        ord_value = 0
        output_noised_np = np.sign(output_noised.cpu().data.numpy())
        output_noised_np_black = np.sign(model2(adv_intermediate_noised).cpu().data.numpy())
        #output_noised_np = output_noised.cpu().data.numpy()
        #output_noised_np_black = model(adv_intermediate_noised).cpu().data.numpy()
        l1_nosied_to_x_white = np.linalg.norm(output_noised_np-oCodeValue, ord=ord_value, axis=1)
        l1_nosied_to_target_white = np.linalg.norm(output_noised_np-tCodeValue, ord=ord_value, axis=1)
        l1_nosied_to_x_black = np.linalg.norm(output_noised_np_black-oCodeValue_black, ord=ord_value, axis=1)

        #output_np = output.cpu().data.numpy()
        output_np = np.sign(output.cpu().data.numpy())
        output_np_black = np.sign(model2(inputs_adv).cpu().data.numpy())

        l1_adv_to_x_white = np.linalg.norm(output_np-oCodeValue, ord=ord_value, axis=1)
        l1_adv_to_target_white = np.linalg.norm(output_np-tCodeValue, ord=ord_value, axis=1)
        l1_adv_to_x_black = np.linalg.norm(output_np_black-oCodeValue_black, ord=ord_value, axis=1)

        l1_noised_to_target_black = np.linalg.norm(output_noised_np_black-targetCode_black, ord=ord_value, axis=1)
        l1_adv_to_target_black = np.linalg.norm(output_np_black-targetCode_black, ord=ord_value, axis=1)


        #loss2 = loss(output, output_noised.detach())
        loss2 = T_h/48.0*2 - loss(output_noised, targetCode.detach())
        #loss2 = 0

        if not frozenVar:
            total_loss = loss1 + var_lambda * (loss2 - var_v) + t / 2 * torch.pow(loss2 - var_v, 2)
        else:
            total_loss = loss1 - var_lambda * loss2
        total_loss.backward(retain_graph=True)
        #print "loss1, loss2, total:", loss1.data.cpu().numpy()[0], total_loss.data.cpu().numpy()[0]
        print("loss1, loss2, total:", loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], total_loss.data.cpu().numpy()[0], var_lambda, var_v)
        print("loss_noise:", loss(output_noised, targetCode.detach()).cpu().data.numpy()[0])
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
        if not frozenVar:
            # optimize lambda
            adv_new = Variable(adv.data)

            output_new = model(adv_new)
            adv_intermediate_noised_new = adv_new + Variable(torch.Tensor(random_noise).cuda())
            output_noised_new = model(adv_intermediate_noised_new)

            #var_grad_lambda = Variable(torch.Tensor(np.array([var_lambda])).cuda(), requires_grad=True)
            loss1_new = loss(output_new, targetCode.detach())
            loss2_new = T_h/48*2 - loss(output_noised_new, targetCode.detach())

            var_v = float((loss2_new+ var_lambda / t).data.cpu().numpy()[0])
            var_v = 0 if var_v <= 0 else var_v
            var_lambda = var_lambda + t * float(loss2_new.data.cpu().numpy()[0])
            var_lambda = 0 if var_lambda<=0 else var_lambda

        if bShowProcess:
            #output = model(adv)
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

        # trace data
        trace_data_mat[i] = np.array([total_loss.data.cpu().numpy()[0], loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], var_lambda, var_v, \
                                      l1_nosied_to_x_white, l1_nosied_to_target_white, l1_nosied_to_x_black, \
                                      l1_adv_to_x_white, l1_adv_to_target_white, l1_adv_to_x_black, \
                                      l1_noised_to_target_black, l1_adv_to_target_black])
        i = i + 1

    adv_np = adv.cpu().data.numpy()
    adv_np[adv_np < 0] = 0
    adv_np[adv_np > 1] = 1
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    output = model(adv)
    oCodeValue = torch.sign(output).cpu().data.numpy()
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    return adv, trace_data_mat


def func_generate_alg2_adv(step_size, step_max, random_noise_level, noise_distribution, var_lambda):
    img_size = 224
    adv_imgs_cornell = np.zeros([i_max, j_max, 3, img_size, img_size])
    trace_data_big_mat = np.zeros([i_max, j_max, step_max, 13])
    for i in range(i_max):
        i_index = int(test_true_id_x[i_index_set[i]])
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])]
        img_t = dset_test[i_index][0]
        for j in range(j_max):
            print("i, j", i, j)
            j_index = int(j_index_set[j])
            targetCode = code[j_index]
            targetCode_black = code2[j_index]
            adv_cornell, trace_data_mat = func_targetedAttack_cornell_lambda_trace(model1, img_t, targetCode, targetCode_black, eps=step_size / 255,
                                                      l_inf_max=step_max, decay_factor=1.0, t_prob=0.5,
                                                      var_lambda=var_lambda,
                                                      random_noise_level=random_noise_level / 255,
                                                      noise_distribution=noise_distribution, model2=model2,
                                                                                   frozenVar=bFrozenVar)
            # print adv_imgs_better[i,j].shape, adv_better.shape

            adv_imgs_cornell[i, j] = adv_cornell.data.cpu().numpy()[0]
            trace_data_big_mat[i, j] = trace_data_mat
        if bPartTest:
            return  adv_imgs_cornell, trace_data_big_mat
    return adv_imgs_cornell, trace_data_big_mat


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpForPapers: Version for Algorithm 2')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='iFGSM', help="adv method")
    parser.add_argument('--var_lambda', type=float, default=1.0, help="lbd to balance loss1 and loss2")
    parser.add_argument('--noise', type=str, default='uniform', help="noise distribution")
    parser.add_argument('--noise_level', type=float, default=32.0, help="random_noise_level")
    parser.add_argument('--net1', type=str, default='ResNet152', help="net1") # ResNext101_32x4d
    parser.add_argument('--net2', type=str, default='ResNext101_32x4d', help="net2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    job_dataset = 'imagenet'
    threshold = 5
    job_values = ['mnist', 'cifar10', 'fashion_mnist']
    net_values = ['ResNet18', 'ResNet34', 'AlexNet', 'ResNet152', 'ResNext101_32x4d']  # 'ResNext101_32x4d'
    adv_method_values = ['FGSM', 'iFGSM', 'iFGSMDI', 'miFGSMDI']
    adv_method = args.adv_method

    # c stands for close, f stands for far
    # W stands for White and B stands for Black
    dis_method_value = ['cW', 'fW', 'cB', 'fB', 'cWcB', 'cWfB', 'fWcB', 'fWfB']
    dis_method = args.dis_method

    i_max = 64#64
    j_max = 32#32

    dset_test, dset_database = load_dset_params(job_dataset)
    net1 = args.net1
    net2 = args.net2
    model1, snapshot_path, query_path, database_path = load_net_params(net1)

    tmp = np.load(database_path)
    output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    tmp = np.load(query_path)
    _, code_test, multi_label_test = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    tmp = np.load(query_path2)
    _, code_test2, _ = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    import os

    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_%s.npy' % (
        adv_method, net1, net2, dis_method)
    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net2)
    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy, multi_label_test,
                                                  code_test2, code2, multi_label2, code_test, code, multi_label)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                   max_dis=18, min_dis=12)

    id_size = test_true_id_x.shape[0]
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    step = 1.0
    linf = 32#64#
    early_stop = False
    early_stop_name = 'early' if early_stop else 'noearly'

    # using test set as the original
    # adv_imgs were produced in myExpGetAdvVulnerable.py
    inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])



    # exp here
    step_size = 1.0
    step_max = linf
    random_noise_level = args.noise_level
    noise_distribution = args.noise
    var_lambda = args.var_lambda
    bLoad = True
    # need run all the data or just j_max amount of data
    bPartTest = False
    bFrozenVar = True

    if bLoad == False:
        adv_imgs_cornell, trace_data_mat = func_generate_alg2_adv(step_size, step_max, random_noise_level, noise_distribution, var_lambda)

        if not bPartTest:
            #os.makedirs('save_for_load/trace/')
            np.save('save_for_load/trace/adv_imgs.npy', adv_imgs_cornell)
            np.save('save_for_load/trace/trace_data_mat.npy', trace_data_mat)
    else:
        adv_imgs_cornell = np.load('save_for_load/trace/adv_imgs.npy')
        trace_data_mat = np.load('save_for_load/trace/trace_data_mat.npy')

    inputs_adv_cornell = Variable(torch.Tensor(adv_imgs_cornell).cuda())
    if not bPartTest:
        img_num_by_class_adv_cornell_white = get_img_num_by_class_from_img_batch(inputs_adv_cornell, model1, code, multi_label, threshold=5, batch_size=8)
        img_num_by_class_adv_cornell_black = get_img_num_by_class_from_img_batch(inputs_adv_cornell, model2, code2, multi_label2, threshold=5, batch_size=8)
        img_num_target_targeted_white = np.zeros([i_max, j_max])
        img_num_target_targeted_black = np.zeros([i_max, j_max])
        img_num_target_ori = np.zeros([i_max, j_max])
        for i in range(i_max):
            #i_index = int(test_true_id_x[i_index_set[i]])
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
            label_targeted = multi_label[j_index_set]

            img_num_target_targeted_white[i] = get_targeted_from_all_class(img_num_by_class_adv_cornell_white[i], label_targeted)
            img_num_target_targeted_black[i] = get_targeted_from_all_class(img_num_by_class_adv_cornell_black[i], label_targeted)
            targetCodes = code[j_index_set]
            img_num_by_class_ori_white = get_query_result_num_by_class(targetCodes, code, multi_label, threshold=5)
            img_num_target_ori[i] = get_targeted_from_all_class(img_num_by_class_ori_white, label_targeted)#get_targeted_from_all_class(img_num_by_class_adv_cornell_white[i], label_targeted)
        img_num_target_ori = img_num_target_ori.astype(int)
        print("img_num_target_targeted_white:", (img_num_target_targeted_white>=100).sum())
        print("img_num_target_targeted_black:", (img_num_target_targeted_black>=10).sum())
        print("img_num_target_targeted_black(valid):", (img_num_target_targeted_black[img_num_target_targeted_white>=100]>=10).sum())

    #success_index = img_num_target_targeted_black>10
    if not bPartTest:
        success_index = trace_data_mat[:,:,-1,12] <= 6
        #success_index = np.ones([i_max, j_max]).astype(bool)
    else:
        success_index = np.zeros([i_max, j_max]).astype(bool)
        success_index[0] = np.ones([1, j_max]).astype(bool)

    from scipy import io
    io.savemat('./trace_data_mat.mat', {'trace_data_mat':trace_data_mat})

    total_loss_traced =  trace_data_mat[success_index,:,0].reshape([-1,step_max]).mean(axis=0)
    loss1_traced = trace_data_mat[success_index,:,1].reshape([-1,step_max]).mean(axis=0)
    loss2_traced = trace_data_mat[success_index,:,2].reshape([-1,step_max]).mean(axis=0)
    lambda_traced = trace_data_mat[0,:,:,3].reshape([-1,step_max]).mean(axis=0)
    v_traced = trace_data_mat[0, :, :, 4].reshape([-1,step_max]).mean(axis=0)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(total_loss_traced)
    plt.plot(loss1_traced)
    plt.plot(loss2_traced)
    #plt.plot(lambda_traced)
    #plt.plot(v_traced)

    l1_nosied_to_x_white_traced = trace_data_mat[success_index, :, 5].reshape([-1,step_max]).mean(axis=0)
    l1_noised_to_target_white_traced = trace_data_mat[success_index, :, 6].reshape([-1,step_max]).mean(axis=0)
    l1_nosied_to_x_black_traced = trace_data_mat[success_index, :, 7].reshape([-1,step_max]).mean(axis=0)
    l1_noised_to_target_black_traced = trace_data_mat[success_index, :, 11].reshape([-1,step_max]).mean(axis=0)

    plt.figure(2)
    plt.plot(l1_nosied_to_x_white_traced)
    plt.plot(l1_noised_to_target_white_traced)
    plt.plot(l1_nosied_to_x_black_traced)
    plt.plot(l1_noised_to_target_black_traced)

    l1_adv_to_x_white_traced = trace_data_mat[success_index, :, 8].reshape([-1,step_max]).mean(axis=0)
    l1_adv_to_target_white_traced = trace_data_mat[success_index, :, 9].reshape([-1,step_max]).mean(axis=0)
    l1_adv_to_x_black_traced = trace_data_mat[success_index, :, 10].reshape([-1,step_max]).mean(axis=0)
    l1_adv_to_target_black_traced = trace_data_mat[success_index, :, 12].reshape([-1,step_max]).mean(axis=0)

    plt.figure(3)
    plt.plot(l1_adv_to_x_white_traced)
    plt.plot(l1_adv_to_target_white_traced)
    plt.plot(l1_adv_to_x_black_traced)
    plt.plot(l1_adv_to_target_black_traced)

    if not bPartTest:
        plt.figure(4)
        for i in range(64):
            l1_nosied_to_x_white_traced = trace_data_mat[i, :, :, 5].reshape([-1, step_max]).mean(axis=0)
            l1_noised_to_target_white_traced = trace_data_mat[i, :, :, 6].reshape([-1, step_max]).mean(axis=0)
            l1_nosied_to_x_black_traced = trace_data_mat[i, :, :, 7].reshape([-1, step_max]).mean(axis=0)
            l1_noised_to_target_black_traced = trace_data_mat[i, :, :, 11].reshape([-1, step_max]).mean(axis=0)
            plt.subplot(8, 8, i+1)
            plt.plot(l1_nosied_to_x_white_traced)
            plt.plot(l1_noised_to_target_white_traced)
            plt.plot(l1_nosied_to_x_black_traced)
            plt.plot(l1_noised_to_target_black_traced)

        plt.figure(5)
        for i in range(64):
            l1_adv_to_x_white_traced = trace_data_mat[i, :, :, 8].reshape([-1, step_max]).mean(axis=0)
            l1_adv_to_target_white_traced = trace_data_mat[i, :, :, 9].reshape([-1, step_max]).mean(axis=0)
            l1_adv_to_x_black_traced = trace_data_mat[i, :, :, 10].reshape([-1, step_max]).mean(axis=0)
            l1_adv_to_target_black_traced = trace_data_mat[i, :, :, 12].reshape([-1, step_max]).mean(axis=0)
            plt.subplot(8, 8, i + 1)
            plt.plot(l1_adv_to_x_white_traced)
            plt.plot(l1_adv_to_target_white_traced)
            plt.plot(l1_adv_to_x_black_traced)
            plt.plot(l1_adv_to_target_black_traced)

