# Basically the same as myExpForPapers.py
import numpy as np
import torch
import torch.nn as nn
import os

from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method, get_adv_by_method
from myGetAdvVulnerable import get_unique_index

from torch.autograd import Variable



from myRetrieval import get_query_code_batch, get_retrieval_result_by_query_code, get_img_num_by_class_from_img_batch, \
    get_targeted_from_all_class

from publicFunctions import  NetworkSettings
from myGetAdv import randomly_input_diversity
from publicVariables import iters_list
from myExpForPapers import get_diff

import torch.optim as optim
optim_dict = {"SGD": optim.SGD}





def func_eval_adv_imgs(adv_imgs, model, code, test_true_label_y):
    # use it to evaluate the adv_imgs
    inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())

    better_img_num_result = get_img_num_by_class_from_img_batch(inputs_adv, model, code, multi_label2,
                                                                threshold=threshold, batch_size=8)
    label_targeted = np.zeros([i_max, j_max])

    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([multi_label2[j_index_set[j]] for j in range(j_max)])
        label_targeted[i] = label_targeted_i

    better_adv_black_retrieval_result = get_targeted_from_all_class(better_img_num_result, label_targeted)

    return better_adv_black_retrieval_result


def func_targetedAttack_nag(model, img_t, targetCode, eps=1.0 / 255, l_inf_max=32, threshold_stop=2,
                            decay_factor=1.0,
                            t_prob=0.5, bShowProcess=False, bEarlyStop=False, random_noise_level=16, var_lambda=1.0,
                            noise_distribution='uniform', adv_method='iFGSM'):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())

    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()

    print('...targeted %s begin....' % (adv_method))
    print('initial distance', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    i = 0
    grad_var = 0

    while i < l_inf_max:
        adv_np = adv.cpu().data.numpy()
        adv_np[adv_np < 0] = 0
        adv_np[adv_np > 1] = 1

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

        # loss2 = loss(output, output_noised.detach())
        loss2 = loss(output_noised, targetCode.detach())
        # loss2 = 0

        total_loss = loss1 + var_lambda * loss2
        total_loss.backward(retain_graph=True)
        # print "loss1, loss2, total:", loss1.data.cpu().numpy()[0], total_loss.data.cpu().numpy()[0]
        print("loss1, loss2, total:", loss1.data.cpu().numpy(), loss2.data.cpu().numpy(),
              total_loss.data.cpu().numpy())
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
            # output = model(adv)
            oCodeValue = torch.sign(output).cpu().data.numpy()
            hamming_dist = np.sum(np.abs(tCodeValue - oCodeValue)) / 2
            print("epoch %d, Hamming Distance: %d" % (i, hamming_dist))
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


def func_targetedAttack_nag_lambda(model, img_t, targetCode, eps=1.0 / 255, l_inf_max=32, threshold_stop=2,
                                   decay_factor=1.0,
                                   t_prob=0.5, bShowProcess=False, bEarlyStop=False, random_noise_level=16,
                                   var_lambda=1.0,
                                   noise_distribution='uniform', adv_method='iFGSM'):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())

    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    tCodeValue = targetCode.cpu().data.numpy()
    oCodeValue = torch.sign(output).cpu().data.numpy()

    print('...targeted %s begin....' % (adv_method))
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
    # var_grad_lambda = Variable(torch.Tensor(np.array([var_lambda])).cuda(), requires_grad=True)
    # var_v = np.random.uniform(0, threshold) / 48*2
    var_v = 0
    # var_grad_v = Variable(torch.Tensor(np.array([var_v])).cuda(), requires_grad=True)
    # t = eps*255*10
    t = 1.0
    # print 'initial var_gar_lambda', var_grad_lambda
    optim_method = 'iFGSM'

    if optim_method == 'adam':
        adv_np = adv.cpu().data.numpy()
        inputs_adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
        optimizer = optim.Adam([inputs_adv], lr=0.001)
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

        loss2 = T_h / 48.0 * 2 - loss(output_noised, targetCode.detach())

        total_loss = loss1 + var_lambda * (loss2 - var_v) + t / 2 * torch.pow(loss2 - var_v, 2)
        total_loss.backward(retain_graph=True)
        optimizer.step()

    elif optim_method == 'iFGSM':
        while i < l_inf_max:
            adv_np = adv.cpu().data.numpy()
            adv_np[adv_np < 0] = 0
            adv_np[adv_np > 1] = 1
            # var_lambda = float(var_grad_lambda.cpu().data.numpy()[0])
            # var_grad_v.zero_grad()
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

            # loss2 = loss(output, output_noised.detach())
            loss2 = T_h / 48.0 * 2 - loss(output_noised, targetCode.detach())
            # loss2 = 0

            total_loss = loss1 + var_lambda * (loss2 - var_v) + t / 2 * torch.pow(loss2 - var_v, 2)
            total_loss.backward(retain_graph=True)
            # print "loss1, loss2, total:", loss1.data.cpu().numpy()[0], total_loss.data.cpu().numpy()[0]
            print("loss1, loss2, total:", loss1.data.cpu().numpy(), loss2.data.cpu().numpy(),
                  total_loss.data.cpu().numpy(), var_lambda, var_v)
            print("loss_noise:", loss(output_noised, targetCode.detach()).cpu().data.numpy())
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

            # optimize lambda
            adv_new = Variable(adv.data)

            output_new = model(adv_new)
            adv_intermediate_noised_new = adv_new + Variable(torch.Tensor(random_noise).cuda())
            output_noised_new = model(adv_intermediate_noised_new)

            # var_grad_lambda = Variable(torch.Tensor(np.array([var_lambda])).cuda(), requires_grad=True)
            loss1_new = loss(output_new, targetCode.detach())
            loss2_new = T_h / 48 * 2 - loss(output_noised_new, targetCode.detach())

            var_v = float((loss2_new + var_lambda / t).data.cpu().numpy())
            var_v = 0 if var_v <= 0 else var_v
            # var_lambda = var_lambda + t * float(loss2_new.data.cpu().numpy()[0])
            var_lambda = var_lambda + t * float(loss2_new.data.cpu().numpy() - var_v)
            var_lambda = 0 if var_lambda <= 0 else var_lambda

            if bShowProcess:
                # output = model(adv)
                oCodeValue = torch.sign(output).cpu().data.numpy()
                hamming_dist = np.sum(np.abs(tCodeValue - oCodeValue)) / 2
                print("epoch %d, Hamming Distance: %d" % (i, hamming_dist))
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


def func_apply_alg2_nag(step_size, step_max, random_noise_level, noise_distribution, var_lambda, test_true_id_x='', test_true_label_y='', dset_test=''):
    # type: (float, float, float or int, str) -> ndarray

    # path_nagAdv = 'save_for_load/distanceADVRetrieval/%s/nagAdv_white_%s_stepsize_%s_stepmax_%s_noiselevel_%s_distribution_%s_lambda_%s_%s.npy' % (
    #    adv_method, net1, str(step_size), str(step_max), str(random_noise_level), noise_distribution, str(var_lambda), dis_method)
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % ('NAG', step_size, step_max, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    path_nagAdv = npy_path
    img_size = 224
    if os.path.exists(path_nagAdv):
        adv_imgs_nag = np.load(path_nagAdv)
    else:
        adv_imgs_nag = np.zeros([i_max, j_max, 3, img_size, img_size])
        for i in range(i_max):
            i_index_i = int(i_index_set[i])
            true_index_x_i = int(test_true_id_x[i_index_i])
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_i])]
            img_t = dset_test[true_index_x_i][0]
            for j in range(j_max):
                print("i, j", i, j)
                j_index = int(j_index_set[j])
                targetCode = code[j_index]

                #adv_nag = func_targetedAttack_nag_lambda(model1, img_t, targetCode, eps=step_size / 255,
                print("i_index:%d,j_index:%d" % (i_index_i, j_index))
                adv_nag = func_targetedAttack_nag(model1, img_t, targetCode, eps=step_size / 255,
                                                             l_inf_max=step_max, decay_factor=1.0, t_prob=0.5,
                                                             var_lambda=var_lambda,
                                                             random_noise_level=random_noise_level / 255,
                                                             noise_distribution=noise_distribution)
                # print adv_imgs_better[i,j].shape, adv_better.shape
                adv_imgs_nag[i, j] = adv_nag.data.cpu().numpy()
        np.save(path_nagAdv, adv_imgs_nag)
    return adv_imgs_nag





def get_random_noise_orthogonal(adv_imgs, ori_imgs, noise_distribution='uniform'):
    # the result is correct
    random_noise_orthogonal = np.zeros_like(adv_imgs)
    for i in range(adv_imgs.shape[0]):
        for j in range(adv_imgs.shape[1]):
            if noise_distribution == 'uniform':
                random_noise = np.random.random_sample(size=adv_imgs[0, 0].shape).astype(float) * 2 - 1.0
            elif noise_distribution == 'Gaussian':
                random_noise = np.random.normal(size=adv_imgs[0, 0].shape).astype(float)
            random_noise = random_noise.reshape([-1])
            adv_direction = (adv_imgs[i, j] - ori_imgs[i]).reshape([-1])
            random_noise /= get_diff(random_noise, np.zeros_like(random_noise))
            diff = adv_direction.astype(np.float32)
            diff /= get_diff(adv_direction, np.zeros_like(adv_direction))
            random_noise -= np.dot(random_noise, diff) * diff
            # for k, channel in enumerate(diff):
            #    random_noise[k] -= np.dot(random_noise[k], channel) * channel
            random_noise_orthogonal[i, j] = random_noise.reshape([3, 224, 224])
            # print (random_noise.reshape([-1]) * adv_direction.reshape([-1])).sum()
    return random_noise_orthogonal


def get_noised_result(model1, adv_imgs, ori_imgs, perturbation_ratio=0.25, noise_level=10, is_orthogonal=False,
                      noise_distribution='uniform', test_true_label_y=''):
    # directly copied from myExpGetAdvVulnerable.py
    # imgs_test = np.load('save_for_load/imgs_test.npy')
    # target_img_mat = np.load('save_for_load/target_imgs.npy')
    if not is_orthogonal:
        if noise_distribution == 'uniform':
            random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
        elif noise_distribution == 'Gaussian':
            # using 3-pi to define the max value of the noise
            random_noise = np.random.normal(0, noise_level / 3, size=adv_imgs.shape).astype(float)
            random_noise = np.clip(random_noise, -noise_level, noise_level)
    else:
        if noise_level == 0:
            random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
        # Get the orthogonal projection of random noise and amplify it to designated number
        else:
            random_noise = get_random_noise_orthogonal(adv_imgs, ori_imgs, noise_distribution)
            random_noise = np.clip(random_noise, -noise_level * 3, noise_level * 3)
            random_noise /= random_noise.max()
            random_noise = np.clip(random_noise, -random_noise.max(), random_noise.max())
            random_noise *= noise_level
    print("Random Noise Real Range:[%f, %f]" % (random_noise.min(), random_noise.max()))
    adv_imgs_noised = np.stack(
        [(adv_imgs[i] - ori_imgs[i]) * perturbation_ratio + ori_imgs[i] for i in range(i_max)]) + random_noise / 255
    # adv_imgs_noised = adv_imgs
    from .myRetrieval import get_img_num_by_class_from_img_batch
    from .myRetrieval import get_targeted_from_all_class
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


def func_IRRN(model, adv_imgs, ori_imgs, perturbation_level_arr, noise_level_arr, index_valid_gt_100, N=10,
              is_orthogonal=False, noise_distribution='uniform', test_true_label_y=''):
    # N is the minimal numbers of retrieval results to make a query to be considered as an adv
    # N = 10 # N is pre-set
    # from myExpForPapers
    p_size = len(perturbation_level_arr)
    n_size = len(noise_level_arr)
    irrn = np.zeros([p_size, n_size])

    for i in range(p_size):
        perturbation_ratio = perturbation_level_arr[i]
        for j in range(n_size):
            noise_level = noise_level_arr[j]
            noised_adv_white_retrieval_result = get_noised_result(model, adv_imgs, ori_imgs,
                                                                  perturbation_ratio=perturbation_ratio,
                                                                  noise_level=noise_level, is_orthogonal=is_orthogonal,
                                                                  noise_distribution=noise_distribution,
                                                                  test_true_label_y=test_true_label_y)
            index_noised_gtN = noised_adv_white_retrieval_result > N
            index_noised_gtN_valid = index_noised_gtN * index_valid_gt_100

            irrn[i][j] = float(index_noised_gtN_valid.sum()) / index_valid_gt_100.sum()
            print(
                "perturbation:%f, noise:%d, irrn:%f, valid_white:%d, valid_all:%d, orthogonal:%s, noise_distribution:%s" % (
                    perturbation_ratio, noise_level, irrn[i][j], index_noised_gtN_valid.sum(), index_valid_gt_100.sum(),
                    str(is_orthogonal), noise_distribution))
    return irrn


def func_do_irrn(adv_imgs):
    ori_imgs = inputs_ori_tensor.cpu().numpy()
    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s_%s.npy' % (
        net1, adv_method, dis_method)
    white_targeted_retrieval_result = np.load(target_targeted_retrieval_num_path)
    index_valid_gt_100 = white_targeted_retrieval_result > 100

    perturbation_level_arr = np.array([0, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0])
    # ([0.0])  # ([1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0]) #
    noise_level_arr = np.array([0, 8, 16, 24, 32])

    irrn = func_IRRN(model1, adv_imgs, ori_imgs, perturbation_level_arr, noise_level_arr, index_valid_gt_100, N=10,
                     is_orthogonal=False, noise_distribution='Gaussian', test_true_label_y=exp_settings.test_true_label_y)
    return irrn


class EXPSettings():
    def __init__(self, net1, net2, dis_method, i_max, j_max, step=1.0, linf=32, max_dis=18, min_dis=12, hash_bit=48):

        self.net1 = net1
        self.net2 = net2
        self.dis_method = dis_method
        self.i_max = i_max
        self.j_max = j_max
        self.step = step
        self.linf = linf
        self.max_dis = max_dis
        self.min_dis = min_dis
        self.hash_bit = hash_bit

        path_to_save = 'save_for_load/distanceADVRetrieval/'
        if not os.path.exists(path_to_save): os.makedirs(path_to_save)
        self.path_white_test_dis_npy = '%s/test_dis_%s.npy' % (path_to_save, net1)
        self.path_black_test_dis_npy = '%s/test_dis_%s.npy' % (path_to_save, net2)

    def get_test_dis_white(self, code_test, code, multi_label):
        if os.path.exists(self.path_white_test_dis_npy):
            test_dis_white = np.load(self.path_white_test_dis_npy)
        else:
            test_dis_white = np.ones([500, 100]) * 49
            for i in range(500):
                for j in range(100):
                    if int(multi_label_test[i]) != j:
                        a = np.linalg.norm(code_test[i:i + 1] - code[multi_label == j], ord=0, axis=-1)
                        test_dis_white[i, j] = a.mean()
            np.save(self.path_white_test_dis_npy, test_dis_white)
        return test_dis_white

    def choose_index_by_dis_method_white(self, test_dis_white):
        hashbit = self.hash_bit
        max_dis = self.max_dis
        min_dis = self.min_dis
        no_same_cate_index = (test_dis_white < hashbit).astype(int)
        if self.dis_method == 'cW':
            test_true_index = (test_dis_white < max_dis).astype(int) * (test_dis_white > min_dis).astype(int)
        if self.dis_method == 'fW':
            test_true_index = (test_dis_white > max_dis).astype(int)
        test_true_index = test_true_index * no_same_cate_index
        test_true_id_list = np.where(test_true_index == 1)
        test_true_id_x = test_true_id_list[0]
        test_true_label_y = test_true_id_list[1]
        return test_true_id_x, test_true_label_y

    def cal_index_set_matrix_white(self, code_test, code, multi_label):
        i_max = self.i_max
        j_max = self.j_max
        path_white_test_dis_npy = self.path_white_test_dis_npy
        test_dis_white = self.get_test_dis_white(code_test, code, multi_label)
        test_true_id_x, test_true_label_y = self.choose_index_by_dis_method_white(test_dis_white)
        id_size = test_true_id_x.shape[0]
        i_index_set = np.arange(0, id_size, id_size / (i_max), dtype=np.int)[:i_max]
        j_index_matrix = get_unique_index(code, multi_label, j_max)
        self.test_true_id_x = test_true_id_x
        self.test_true_label_y = test_true_label_y
        self.i_index_set = i_index_set.astype(int)
        self.j_index_matrix = j_index_matrix
        return self.i_index_set, self.j_index_matrix

    def cal_index_set_matrix(self, multi_label_test, code_test2, code2, multi_label2, code_test, code, multi_label):
        # Select the vulnerable index.
        # Deprecated
        i_max = self.i_max
        j_max = self.j_max
        path_white_test_dis_npy = self.path_white_test_dis_npy
        path_black_test_dis_npy = self.path_black_test_dis_npy

        test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy,
                                                      multi_label_test,
                                                      code_test2, code2, multi_label2, code_test, code, multi_label)
        test_true_id_x, test_true_label_y = choose_index_by_dis_method(self.dis_method, test_dis_white, test_dis_black,
                                                                       max_dis=self.max_dis, min_dis=self.min_dis)
        id_size = test_true_id_x.shape[0]
        #i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]
        i_index_set = np.arange(0, id_size, id_size / (i_max), dtype=np.int)[:i_max]
        j_index_matrix = get_unique_index(code, multi_label, j_max)
        self.test_true_id_x = test_true_id_x
        self.test_true_label_y = test_true_label_y
        self.i_index_set = i_index_set.astype(int)
        self.j_index_matrix = j_index_matrix
        return self.i_index_set, self.j_index_matrix

    def get_index_set_matrix(self):
        return self.i_index_set, self.j_index_matrix

    def cal_inputs_ori_tensor(self, dset_test=''):
        inputs_ori_list = []
        for i in range(self.i_max):
            i_index_i = int(self.i_index_set[i])
            test_true_id_x_i = int(self.test_true_id_x[i_index_i])
            image_i = dset_test[test_true_id_x_i][0]
            inputs_ori_list.append(image_i)
        inputs_ori_tensor = torch.stack(inputs_ori_list)
        #    inputs_ori_tensor = torch.stack([dset_test[self.test_true_id_x[self.i_index_set[i]]][0] for i in range(self.i_max)])
        self.inputs_ori_tensor = inputs_ori_tensor
        return self.inputs_ori_tensor

    def get_inputs_ori_tensor(self):
        return self.inputs_ori_tensor



class RetrievalExp:
    # Do retrieval exp
    def __init__(self, retri_sys):
        self.retri_sys = retri_sys
        pass

    def select_(self):
        pass

    def whitebox_retrieval(self):
        pass

class IRRNexp:
    def __init__(self, minN=10, is_orth=False, noise_distribution='uniform'):
        self.minN = minN
        self.is_orth = is_orth
        self.noise_distribution = noise_distribution
        raise NotImplementedError

class AttackFlow:
    def __init__(self):
        raise NotImplementedError



def exp_alg2_nag():
    step_size = step
    step_max = linf
    random_noise_level = args.noise_level
    noise_distribution = args.noise
    var_lambda = args.var_lambda
    isBlackBox = False
    isIRRN = False
    adv_imgs_nag = func_apply_alg2_nag(step_size, step_max, random_noise_level, noise_distribution, var_lambda, \
                                           test_true_id_x=exp_settings.test_true_id_x, \
                                           test_true_label_y=exp_settings.test_true_label_y, \
                                           dset_test=dset_loaders['test'].dataset)
    return adv_imgs_nag




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
    parser.add_argument('--net1', type=str, default='ResNet152', help="net1")  # ResNext101_32x4d
    parser.add_argument('--net2', type=str, default='ResNext101_32x4d', help="net2(NOT Necessary and Won't change to results!)")
    parser.add_argument('--l_inf_max', type=float, default=32, help="l_inf_max")
    parser.add_argument('--step_size', type=float, default=1.0, help="step_size")

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

    i_max = 64
    j_max = 32

    step = args.step_size
    linf = args.l_inf_max

    net1 = args.net1
    net2 = args.net2

    hash_bit = 48
    snapshot_iter1 = iters_list[net1]
    network_settings1 = NetworkSettings(job_dataset, hash_bit, net1, snapshot_iter1, batch_size=16)
    model1 = network_settings1.get_model()
    dset_loaders = network_settings1.get_dset_loaders()
    _, code, multi_label = network_settings1.get_out_code_label(part='database')
    _, code_test, multi_label_test = network_settings1.get_out_code_label(part='test')
    dset_database = dset_loaders['database'].dataset

    snapshot_iter2 = iters_list[net2]
    network_settings2 = NetworkSettings(job_dataset, hash_bit, net2, snapshot_iter2, batch_size=16)
    model2 = network_settings2.get_model()
    _, code2, multi_label2 = network_settings2.get_out_code_label(part='database')
    _, code_test2, _ = network_settings2.get_out_code_label(part='test')

    exp_settings = EXPSettings(net1, net2, dis_method, i_max, j_max, step=step, linf=linf)
    #i_index_set, j_index_matrix = exp_settings.cal_index_set_matrix(multi_label_test, code_test2, code2, multi_label2, code_test, code, multi_label)
    i_index_set, j_index_matrix =  exp_settings.cal_index_set_matrix_white(code_test, code, multi_label)
    inputs_ori_tensor = exp_settings.cal_inputs_ori_tensor(dset_test=dset_loaders['test'].dataset)


    adv_imgs_nag = exp_alg2_nag()

    #irrn_nag = func_do_irrn(adv_imgs_nag)