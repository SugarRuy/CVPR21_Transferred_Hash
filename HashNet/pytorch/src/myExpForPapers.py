import numpy as np
import torch
import torch.nn as nn

from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method, get_adv_by_method
from myGetAdvVulnerable import get_unique_index
from data_list import default_loader
from torch.autograd import Variable

import matplotlib.pyplot as plt

from myRetrieval import get_query_code_batch, get_retrieval_result_by_query_code

from publicFunctions import load_net_inputs, load_net_params, load_dset_params
from myRetrieval import get_img_num_by_class_from_img_batch
from myRetrieval import get_targeted_from_all_class
from myGetAdv import randomly_input_diversity

def get_diff(sample_1, sample_2):
    if True:
        # My own version
        # fuck the boundary attack. I made my own version using p_2 = p_1 - (p_1 * d_1) * d_1 !!!
        #sample_1 = sample_1.reshape(3, 224, 224)
        #sample_2 = sample_2.reshape(3, 224, 224)
        diff = np.linalg.norm((sample_1 - sample_2).astype(np.float32))
        return diff

    if False:
        # the same from boundary attack
        sample_1 = sample_1.reshape(3, 224, 224)
        sample_2 = sample_2.reshape(3, 224, 224)
        diff = []
        for i, channel in enumerate(sample_1):
            diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
        print(np.array(diff).shape)
        return np.array(diff)





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
            #for k, channel in enumerate(diff):
            #    random_noise[k] -= np.dot(random_noise[k], channel) * channel
            random_noise_orthogonal[i, j] = random_noise.reshape([3,224,224])
            #print (random_noise.reshape([-1]) * adv_direction.reshape([-1])).sum()
    return random_noise_orthogonal


def get_noised_result(model1, adv_imgs, ori_imgs, perturbation_ratio=0.25, noise_level=10, is_orthogonal = False,
                      noise_distribution='uniform'):
    # directly copied from myExpGetAdvVulnerable.py
    # imgs_test = np.load('save_for_load/imgs_test.npy')
    # target_img_mat = np.load('save_for_load/target_imgs.npy')
    if not is_orthogonal:
        if noise_distribution == 'uniform':
            random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
        elif noise_distribution == 'Gaussian':
            # using 3-pi to define the max value of the noise
            random_noise = np.random.normal(0, noise_level / 3 , size=adv_imgs.shape).astype(float)
            random_noise = np.clip(random_noise, -noise_level, noise_level)
    else:
        if noise_level == 0:
            random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
        # Get the orthogonal projection of random noise and amplify it to designated number
        else:
            random_noise = get_random_noise_orthogonal(adv_imgs, ori_imgs, noise_distribution)
            random_noise = np.clip(random_noise, -noise_level*3, noise_level*3)
            random_noise /= random_noise.max()
            random_noise = np.clip(random_noise, -random_noise.max(), random_noise.max())
            random_noise *= noise_level
    print("Random Noise Real Range:[%f, %f]"%(random_noise.min(), random_noise.max()))
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


def func_IRRN(model, adv_imgs, ori_imgs, perturbation_level_arr, noise_level_arr, index_valid_gt_100, index_black_gt_1,
              N=10, is_orthogonal = False,noise_distribution='uniform'):
    # N is the minimal numbers of retrieval results to make a query to be considered as an adv
    # N = 10 # N is pre-set
    p_size = len(perturbation_level_arr)
    n_size = len(noise_level_arr)
    irrn = np.zeros([p_size, n_size])

    for i in range(p_size):
        perturbation_ratio = perturbation_level_arr[i]
        for j in range(n_size):
            noise_level = noise_level_arr[j]
            noised_adv_white_retrieval_result = get_noised_result(model, adv_imgs, ori_imgs,
                                                                  perturbation_ratio=perturbation_ratio,
                                                                  noise_level=noise_level,
                                                                  is_orthogonal=is_orthogonal,
                                                                  noise_distribution=noise_distribution)
            index_noised_gtN = noised_adv_white_retrieval_result > N
            index_noised_gtN_valid = index_noised_gtN * index_valid_gt_100
            index_valid_noised_gtN_success_gt1 = index_noised_gtN_valid * index_black_gt_1
            irrn[i][j] = float(index_noised_gtN_valid.sum()) / index_valid_gt_100.sum()
            print("perturbation:%f, noise:%d, irrn:%f, valid_white:%d, valid_all:%d, orthogonal:%s, noise_distribution:%s" % (
                perturbation_ratio, noise_level, irrn[i][j], index_noised_gtN_valid.sum(), index_valid_gt_100.sum(), str(is_orthogonal), noise_distribution))
    return irrn


def exp_IRRN():
    # As its name says, this function is for experiment for applying IRRN method in adv_imgs to evaluate the robustness-to-random-noise

    ori_imgs = inputs_ori_tensor.cpu().numpy()

    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s.npy' % (
        net1, dis_method)
    white_targeted_retrieval_result = np.load(target_targeted_retrieval_num_path)
    index_valid_gt_100 = white_targeted_retrieval_result > 100

    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_%s.npy' % (
        adv_method, net1, net2, dis_method)
    black_retrieval_result = np.load(path_blackTargetedNum)
    index_black_gt_1 = black_retrieval_result > 1

    perturbation_level_arr = np.array([1.0/2, 1.0])
        #([0, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0])
        #([0.0])  # ([1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0]) #
    noise_level_arr = np.array([0, 4, 8, 16, 24, 32, 64])
    #noise_level_arr = np.array([16, 32, 64])
    irrn = func_IRRN(model1, adv_imgs, ori_imgs, perturbation_level_arr, noise_level_arr, index_valid_gt_100,
                     index_black_gt_1, N=10, is_orthogonal=False, noise_distribution='Gaussian')
    print("experiment on IRRN for white: %s, black: %s, adv_method: %s, dis_method: %s" % (
        net1, net2, adv_method, dis_method))
    return irrn


def func_single_RRN(model, adv_img, source_img_index, radius_candidates_array, candidates_size=8, N=10):
    # NOTE: This file is not debugged. To be debugged when the computational resource is available

    # N is the minimal numbers of retrieval results to make a query to be considered as an adv
    # N = 10 # N is pre-set

    radius_array_size = len(radius_candidates_array)
    label_targeted = np.zeros([i_max, j_max])
    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        label_targeted[i] = label_targeted_i

    for i_radius in range(radius_array_size):
        noise_level = radius_candidates_array[i_radius]
        random_noise = np.random.randint(-noise_level, noise_level + 1,
                                         np.concatenate((candidates_size, adv_img.shape), axis=None)).astype(float)
        adv_imgs_noised = adv_img + random_noise / 255

        X = Variable(torch.Tensor(adv_imgs_noised)).cuda()
        noised_img_num_result = get_img_num_by_class_from_img_batch(X, model, code, multi_label, threshold=5,
                                                                    batch_size=16)

        # bugs over the following code
        label_noised = np.zeros([candidates_size])
        label_noised = label_targeted[source_img_index][:candidates_size]
        noised_adv_white_retrieval_result = get_targeted_from_all_class(noised_img_num_result, label_noised)
        index_noised_gtN = noised_adv_white_retrieval_result > N

        print("Pass Number SUM:", index_noised_gtN.sum())
        if index_noised_gtN.sum() == candidates_size:
            print("Stop at:", radius_candidates_array[i_radius])
            return radius_candidates_array[i_radius]
        else:
            print("Continue at:", radius_candidates_array[i_radius])

    if i_radius == radius_array_size:
        return 0  # default radius


def func_matrix_RRN():
    rrn = {}

    return rrn


def exp_RRN():
    ori_imgs = inputs_ori_tensor.cpu().numpy()

    radius_candidates_array = 32 - np.arange(32)
    source_img_index = 0
    rrn = func_single_RRN(model1, adv_imgs[source_img_index][13], source_img_index, radius_candidates_array)

    return rrn


def exp_pairs_difference():
    # Implement exps in myExpForPapers_diff.py
    # describe experiments here
    # Here we want to show that pairing process in HashNet really makes difference,
    # the exps we want to do are:
    # 1. Training HashNet#A, HashNet#B with different pairs data on ImageNet(Hash Models)
    # 2. Training ResNet152#A, ResNet152#B with different shuffled data on ImageNet(Classification Models)
    # 3. Comparing the difference in Black-Box Attack between Hash#B to Hash#A and Res#B to Res#A.
    # 4. Statement holds true if success rate of Res#B to Res#A is significantly better than result of Hash#B to Hash#A
    return


def func_success_rate():
    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_%s.npy' % (
        adv_method, net1, net2, dis_method)
    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s.npy' % (
        net1, dis_method)

    blackTargetedNumMatrix = np.load(path_blackTargetedNum)
    white_targeted_retrieval_result = np.load(target_targeted_retrieval_num_path)

    index_valid_gt_100 = white_targeted_retrieval_result > 100

    print((blackTargetedNumMatrix[index_valid_gt_100] > 100).sum(), index_valid_gt_100.sum())
    return


def exp_success_rate():
    func_success_rate()
    return


def func_adv_hamming_distance():
    path_code_tar_adv_ori = 'save_for_load/distanceADVRetrieval/%s/code_tar_adv_ori_white_%s_black_%s_%s.npz' % (
        adv_method, net1, net2, dis_method)
    if os.path.exists(path_code_tar_adv_ori):
        code_tar_adv_ori = np.load(path_code_tar_adv_ori)
        code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black = \
            code_tar_adv_ori['arr_0'], code_tar_adv_ori['arr_1'], code_tar_adv_ori['arr_2'], \
            code_tar_adv_ori['arr_3'], code_tar_adv_ori['arr_4'], code_tar_adv_ori['arr_5']

    else:
        print("adv_method:%s, dis_method:%s"%(adv_method, dis_method))
        hash_bit = 48
        code_targeted_white = np.zeros([i_max, j_max, hash_bit])
        code_source_white = np.zeros([i_max, hash_bit])
        code_adv_white = np.zeros([i_max, j_max, hash_bit])
        code_targeted_black = np.zeros([i_max, j_max, hash_bit])
        code_source_black = np.zeros([i_max, hash_bit])
        code_adv_black = np.zeros([i_max, j_max, hash_bit])
        for i in range(i_max):
            print("Process: %d"%(i))
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
            i_index = int(test_true_id_x[i_index_set[i]])

            code_targeted_white_i = np.array([code[j_index_set[j]] for j in range(j_max)])
            code_targeted_white[i] = code_targeted_white_i
            code_targeted_black_i = np.array([code2[j_index_set[j]] for j in range(j_max)])
            code_targeted_black[i] = code_targeted_black_i

            code_source_white[i] = code_test[i_index]
            code_source_black[i] = code_test2[i_index]

            adv_inputs = Variable(torch.Tensor(adv_imgs[i])).cuda()
            batch_size = 8
            code_adv_white[i] = get_query_code_batch(img_inputs = adv_inputs, model = model1, batch_size = batch_size)
            code_adv_black[i] = get_query_code_batch(img_inputs = adv_inputs, model = model2, batch_size = batch_size)
        np.savez(path_code_tar_adv_ori, code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black)
    return code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black


def func_code_adv_result_black(code_adv, database_code2, multi_label2, targeted_labels, threshold=5):
    # get code_adv, return the a set of codes in targeted class for each code in code_adv
    hash_bit = 48
    result_code_set = {}
    for i in range(i_max):
        result_code_set_i = {}
        target_database_code2 = database_code2[multi_label2==targeted_labels[i]]
        query_result_i = get_retrieval_result_by_query_code(code_adv[i], target_database_code2, threshold)
        for j in range(j_max):
            query_result_i_j = query_result_i[j]
            result_codes = target_database_code2[np.array(query_result_i_j).astype(int)]
            result_code_set_i[j] = result_codes

        result_code_set[i] = result_code_set_i
    return result_code_set


def func_draw_adv_hamming_distance(code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black, code_adv_result_black):
    hamming_adv_ori_white = np.zeros([i_max, j_max])
    hamming_adv_tar_white = np.zeros([i_max, j_max])
    hamming_tar_ori_white = np.zeros([i_max, j_max])
    hamming_adv_ori_black = np.zeros([i_max, j_max])
    hamming_adv_tar_black = np.zeros([i_max, j_max])
    hamming_tar_ori_black = np.zeros([i_max, j_max])

    hamming_adv_result_black = np.zeros([i_max, j_max])



    for i in range(i_max):
        hamming_adv_ori_white[i] = np.linalg.norm(code_adv_white[i] - code_source_white[i], axis=1, ord=0)
        hamming_adv_tar_white[i] = np.linalg.norm(code_targeted_white[i] - code_adv_white[i], axis=1, ord=0)
        hamming_tar_ori_white[i] = np.linalg.norm(code_targeted_white[i] - code_source_white[i], axis=1, ord=0)
        hamming_adv_ori_black[i] = np.linalg.norm(code_adv_black[i] - code_source_black[i], axis=1, ord=0)
        hamming_adv_tar_black[i] = np.linalg.norm(code_targeted_black[i] - code_adv_black[i], axis=1, ord=0)
        hamming_tar_ori_black[i] = np.linalg.norm(code_targeted_black[i] - code_source_black[i], axis=1, ord=0)


    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_%s.npy' % (
        adv_method, net1, net2, dis_method)
    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s.npy' % (
        net1, dis_method)

    blackTargetedNumMatrix = np.load(path_blackTargetedNum)
    white_targeted_retrieval_result = np.load(target_targeted_retrieval_num_path)
    valid_index = white_targeted_retrieval_result > 100

    th_hard_index = hamming_adv_tar_black <= 5

    plt.title(adv_method+'/'+dis_method)
    if dis_method == 'cW':
        #index_by_18 = hamming_tar_ori_white <= 18
        index_by_18 = hamming_tar_ori_white <= 48
    else:
        index_by_18 = hamming_tar_ori_white > 18

    # index_final = (blackTargetedNumMatrix < 10) * index_by_18 * valid_index
    # plt.plot(hamming_tar_ori_white[index_final], hamming_tar_ori_black[index_final], 'o', alpha=0.2, fillstyle='none')
    # print index_final.sum()
    plt.xlabel("hamming_tar_ori_white")
    plt.ylabel("hamming_adv_tar_black")
    index_final = (hamming_adv_tar_black <= 5) * index_by_18 * valid_index * (blackTargetedNumMatrix >= 10)
    succeeded_plot = plt.plot(hamming_tar_ori_white[index_final], hamming_adv_tar_black[index_final], 'ro', alpha=0.3, fillstyle='none', label='lte_5_succeeded_plot')
    print(index_final.sum())
    '''
    index_final = (hamming_adv_tar_black > 5 ) * index_by_18 * valid_index * (blackTargetedNumMatrix >= 10)
    succeeded_plot = plt.plot(hamming_tar_ori_white[index_final], hamming_adv_tar_black[index_final], 'bo', alpha=0.3,
                              fillstyle='none', label='gt_5_succeeded_plot')
    print index_final.sum()
    '''
    index_final = (hamming_adv_tar_black > 5) * index_by_18 * valid_index * (blackTargetedNumMatrix >= 10)
    for i in range(i_max):
        for j in range(j_max):
            if index_final[i,j] == 1:
                hamming_adv_result_black[i, j] = np.linalg.norm(code_adv_black[i, j] - np.array(code_adv_result_black[i][j]).squeeze(),
                                                            axis=1, ord=0).mean()
    succeeded_plot = plt.plot(hamming_tar_ori_white[index_final], hamming_adv_result_black[index_final], 'bo', alpha=0.3,
                              fillstyle='none', label='gt_5_succeeded_plot_new')
    print(index_final.sum())


    index_final = (hamming_adv_tar_black > 5) * index_by_18 * valid_index *(blackTargetedNumMatrix < 10)
    failed_plot = plt.plot(hamming_tar_ori_white[index_final], hamming_adv_tar_black[index_final], 'bx', alpha=0.3, fillstyle='none', label='gt_5_failed_plot')
    print(index_final.sum())

    index_final = (hamming_adv_tar_black <= 5) * index_by_18 * valid_index *(blackTargetedNumMatrix < 10)
    failed_plot = plt.plot(hamming_tar_ori_white[index_final], hamming_adv_tar_black[index_final], 'rx', alpha=0.3,
                           fillstyle='none', label='lte_5_failed_plot')
    print(index_final.sum())

    #plt.legend([succeeded_plot, failed_plot], ['succeeded', 'failed'])
    plt.legend()


    '''
    plt.plot(hamming_tar_ori_white, hamming_adv_ori_white, 'bo', alpha=0.1)
    plt.plot(hamming_tar_ori_white, hamming_adv_tar_white, 'ro', alpha=0.1)
    plt.plot(hamming_tar_ori_black, hamming_adv_ori_black, 'bx', alpha=0.1)
    plt.plot(hamming_tar_ori_black, hamming_adv_tar_black, 'rx', alpha=0.1)
    '''
    return


def exp_adv_hamming_distance():
    # experiment for showing the hamming distances between adv, tar and ori.
    code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black = func_adv_hamming_distance()
    targeted_labels = np.array([int(test_true_label_y[i_index_set[i]]) for i in range(i_max)] )
    code_adv_result_black = func_code_adv_result_black(code_adv_black, code2, multi_label2, targeted_labels, threshold=5)
    func_draw_adv_hamming_distance(code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black, code_adv_result_black)

    return code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black



def func_get_traced_adv_img(model, img_t, targetCode, eps=1.0 / 255, l_inf_max=32, threshold_stop=2, loss=nn.L1Loss(),
                       decay_factor=1.0, t_prob=0.5, bShowProcess=False, bEarlyStop=False, adv_method='iFGSM'):
    if not isinstance(targetCode, torch.cuda.FloatTensor):
        targetCode = Variable(torch.Tensor(targetCode).cuda())

    traced_adv_imgs = {}

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

        traced_adv_imgs[i] = adv_np
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
    traced_adv_imgs[i] = adv_np
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    output = model(adv)
    oCodeValue = torch.sign(output).cpu().data.numpy()
    print('...Final Hamming distance : ', np.sum(np.abs(tCodeValue - oCodeValue)) / 2)
    return np.array([traced_adv_imgs[i][0] for i in range(l_inf_max+1)]) # traced_adv_imgs#np.array(list(traced_adv_imgs.items()))


def func_trace_adv_code():
    path_traced_adv_codes = 'save_for_load/distanceADVRetrieval/%s/traced_adv_codes_%s_black_%s_%s.npz' % (
        adv_method, net1, net2, dis_method)
    if os.path.exists(path_traced_adv_codes):
        code_tar_adv_ori = np.load(path_traced_adv_codes)
        traced_adv_code_white, traced_adv_code_black = \
            code_tar_adv_ori['arr_0'], code_tar_adv_ori['arr_1']
    else:
        hash_bit = 48
        traced_adv_code_white = np.zeros([i_max, j_max, linf + 1, hash_bit])
        traced_adv_code_black = np.zeros([i_max, j_max, linf + 1, hash_bit])
        for i in range(i_max):
            i_index = int(test_true_id_x[i_index_set[i]])
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])]
            img_t = dset_test[i_index][0]
            for j in range(j_max):
                j_index = int(j_index_set[j])
                targetCode = code[j_index]

                print("i,j:", i, j)
                traced_adv_imgs = func_get_traced_adv_img(model1, img_t, targetCode, eps=step/255, l_inf_max=linf, bShowProcess=False, bEarlyStop=False, adv_method=adv_method)

                traced_adv_img_inputs = Variable(torch.Tensor(traced_adv_imgs)).cuda()
                traced_adv_code_white[i, j] = get_query_code_batch(traced_adv_img_inputs, model1)
                traced_adv_code_black[i, j] = get_query_code_batch(traced_adv_img_inputs, model2)
        np.savez(path_traced_adv_codes, traced_adv_code_white, traced_adv_code_black)
    #traced_adv_code_white, trace_adv_code_black = func_traced_adv_imgs()
    return traced_adv_code_white, traced_adv_code_black


def func_draw_traced_adv_hamming_distance(code_targeted_white, code_source_white, code_targeted_black, code_source_black, traced_adv_code_white, traced_adv_code_black):
    traced_hamming_adv_ori_white = np.zeros([linf+1, i_max, j_max])
    traced_hamming_adv_tar_white = np.zeros([linf+1, i_max, j_max])
    traced_hamming_tar_ori_white = np.zeros([linf+1, i_max, j_max])
    traced_hamming_adv_ori_black = np.zeros([linf+1, i_max, j_max])
    traced_hamming_adv_tar_black = np.zeros([linf+1, i_max, j_max])
    traced_hamming_tar_ori_black = np.zeros([linf+1, i_max, j_max])
    for i in range(i_max):
        for k in range(linf+1):
            code_adv_white = traced_adv_code_white[:,:,k,:]
            code_adv_black = traced_adv_code_black[:,:,k,:]
            traced_hamming_adv_ori_white[k, i] = np.linalg.norm(code_adv_white[i] - code_source_white[i], axis=1, ord=0)
            traced_hamming_adv_tar_white[k, i] = np.linalg.norm(code_targeted_white[i] - code_adv_white[i], axis=1, ord=0)
            traced_hamming_tar_ori_white[k, i] = np.linalg.norm(code_targeted_white[i] - code_source_white[i], axis=1, ord=0)
            traced_hamming_adv_ori_black[k, i] = np.linalg.norm(code_adv_black[i] - code_source_black[i], axis=1, ord=0)
            traced_hamming_adv_tar_black[k, i] = np.linalg.norm(code_targeted_black[i] - code_adv_black[i], axis=1, ord=0)
            traced_hamming_tar_ori_black[k, i] = np.linalg.norm(code_targeted_black[i] - code_source_black[i], axis=1, ord=0)

    mat_dict = {"traced_hamming_adv_ori_white":traced_hamming_adv_ori_white,
                "traced_hamming_adv_tar_white":traced_hamming_adv_tar_white,
                "traced_hamming_tar_ori_white":traced_hamming_tar_ori_white,
                "traced_hamming_adv_ori_black":traced_hamming_adv_ori_black,
                "traced_hamming_adv_tar_black":traced_hamming_adv_tar_black,
                "traced_hamming_tar_ori_black":traced_hamming_tar_ori_black}
    from scipy import io
    path_traced_hamming = 'save_for_load/distanceADVRetrieval/%s/traced_hamming_%s_black_%s_%s.mat' % (
        adv_method, net1, net2, dis_method)
    io.savemat(path_traced_hamming, mat_dict)


def exp_trace_hamming_distance():
    code_targeted_white, code_source_white, _, code_targeted_black, code_source_black, _ = func_adv_hamming_distance()
    traced_adv_code_white, traced_adv_code_black = func_trace_adv_code()
    func_draw_traced_adv_hamming_distance(code_targeted_white, code_source_white, code_targeted_black, code_source_black, traced_adv_code_white, traced_adv_code_black)

    return traced_adv_code_white, traced_adv_code_black

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpForPapers')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='FGSM', help="adv method")
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

    i_max = 64
    j_max = 32

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

    step = 8.0#1.0
    linf = 8#16
    early_stop = False
    early_stop_name = 'early' if early_stop else 'noearly'

    # using test set as the original
    # adv_imgs were produced in myExpGetAdvVulnerable.py
    inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    adv_imgs = np.load(npy_path)

    #
    irrn = exp_IRRN()
    # rrn = exp_RRN()
    # exp_success_rate()
    #code_targeted_white, code_source_white, code_adv_white, code_targeted_black, code_source_black, code_adv_black = exp_adv_hamming_distance()
    #traced_adv_code_white, traced_adv_code_black = exp_trace_hamming_distance()
    print(irrn)
