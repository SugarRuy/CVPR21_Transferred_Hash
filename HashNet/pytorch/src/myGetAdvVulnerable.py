import os

import numpy as np
import torch
from torch.autograd import Variable

from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class
# use it to get the adv_imgs retrieval result
from publicFunctions import load_net_inputs, load_net_params, load_dset_params


def get_unique_index(code, multi_label, j_max):
    '''

    Args:
        code: query_code of the database
        multi_label: labels of the database
        j_max: maximum sample size in one class

    Returns:
        unique_index_matrix: A matrix containing indexes of all the targeted images in the database
    '''
    max_cat = int(multi_label.max()) + 1
    unique_index_matrix = np.zeros([max_cat, j_max]) - 1
    for i in range(max_cat):
        index_cat_i = np.where(multi_label == i)[0]
        unique_index_i, indics_i = np.unique(code[index_cat_i], return_index=True, axis=0)
        if unique_index_i.shape[0] <= 32:
            return 0
        else:
            chosen_index = np.arange(0, unique_index_i.shape[0], int(unique_index_i.shape[0] / j_max))
            unique_index_matrix[i] = index_cat_i[indics_i[chosen_index[:j_max].astype(int)]].astype(int)

    return unique_index_matrix

def get_test_dis_white(white_test_dis_npy_path, multi_label_test=None, code_test=None, code=None, multi_label=None):
    if os.path.exists(white_test_dis_npy_path):
        test_dis_white = np.load(white_test_dis_npy_path)
    else:
        test_dis_white = np.ones([500, 100]) * 49
        for i in range(500):
            for j in range(100):
                if int(multi_label_test[i]) != j:
                    a = np.linalg.norm(code_test[i:i + 1] - code[multi_label == j], ord=0, axis=-1)
                    test_dis_white[i, j] = a.mean()
        np.save(white_test_dis_npy_path, test_dis_white)
    return test_dis_white

def get_test_dis(white_test_dis_npy_path, black_test_dis_npy_path, multi_label_test=None, code_test2=None, code2=None, multi_label2=None, code_test=None, code=None, multi_label=None):
    # use the first 500 code in test set, which covers all 100 classes, as the source
    # use all the 100 classes code in train set as the target classes
    # get a 500*100 mat by calculating the average distance from the source to target
    if os.path.exists(white_test_dis_npy_path):
        test_dis_white = np.load(white_test_dis_npy_path)
    else:
        test_dis_white = np.ones([500, 100]) * 49
        for i in range(500):
            for j in range(100):
                if int(multi_label_test[i]) != j:
                    a = np.linalg.norm(code_test[i:i + 1] - code[multi_label == j], ord=0, axis=-1)
                    test_dis_white[i, j] = a.mean()
        np.save(white_test_dis_npy_path, test_dis_white)

    if os.path.exists(black_test_dis_npy_path):
        test_dis_black = np.load(black_test_dis_npy_path)
    else:
        test_dis_black = np.ones([500, 100]) * 49
        for i in range(500):
            for j in range(100):
                if int(multi_label_test[i]) != j:
                    a = np.linalg.norm(code_test2[i:i + 1] - code2[multi_label2 == j], ord=0, axis=-1)
                    test_dis_black[i, j] = a.mean()
        np.save(black_test_dis_npy_path, test_dis_black)

    return test_dis_white, test_dis_black


def choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black, max_dis=18, min_dis=12):
    '''

    Args:
        dis_method: the specific method to choose index
        test_dis_white: a matrix of hamming distance from one test image to a database targeted class, in whitebox net1.
        test_dis_black: a matrix of hamming distance from one test image to a database targeted class, in blackbox net2.
        max_dis: a threshold dividing the close and far
        min_dis: a threshold identifying the outlier distances.

    Returns:
        test_true_id_x: the qualified id of test images in test set
        test_true_label_y: the corresponding class label of targeted database images in database.
    '''
    hashbit = 48
    no_same_cate_index = (test_dis_white < hashbit).astype(int) * (test_dis_black < hashbit).astype(int)
    if dis_method == 'cW':
        test_true_index = (test_dis_white < max_dis).astype(int) * (test_dis_white > min_dis).astype(int)
    if dis_method == 'fW':
        test_true_index = (test_dis_white > max_dis).astype(int)
    if dis_method == 'cB':
        test_true_index = (test_dis_black < max_dis).astype(int) * (test_dis_black > min_dis).astype(int)
    if dis_method == 'fB':
        test_true_index = (test_dis_black > max_dis).astype(int)
    if dis_method == 'cWcB':
        test_true_index = (test_dis_white < max_dis).astype(int) * (test_dis_white > min_dis).astype(int) * (
                test_dis_black < max_dis).astype(int) * (test_dis_black > min_dis).astype(int)
    if dis_method == 'cWfB':
        test_true_index = (test_dis_white < max_dis).astype(int) * (test_dis_white > min_dis).astype(int) * (
                test_dis_black > max_dis).astype(int)
    if dis_method == 'fWcB':
        test_true_index = (test_dis_white > max_dis).astype(int) * (test_dis_black < max_dis).astype(int) * (
                test_dis_black > min_dis).astype(int)
    if dis_method == 'fWfB':
        test_true_index = (test_dis_white > max_dis).astype(int) * (test_dis_black > max_dis).astype(int)
    test_true_index = test_true_index * no_same_cate_index
    test_true_id_list = np.where(test_true_index == 1)
    test_true_id_x = test_true_id_list[0]
    test_true_label_y = test_true_id_list[1]
    return test_true_id_x, test_true_label_y

def get_adv_by_method(model1, img_t, targetCode, adv_method, step=1.0, linf=32.0, bShowProcess=False, bEarlyStop=False):
    from myGetAdv import targetedAttack
    # choose the attack method of adv_method variable
    # adv_method: name of the adversarial method
    # step: step size
    # linf: maximum l_infinite difference
    if adv_method == 'iFGSM':
        adv__ = targetedAttack(model1, img_t, targetCode, eps=step/255, l_inf_max=linf, bShowProcess=bShowProcess, bEarlyStop=bEarlyStop, adv_method='iFGSM')
    if adv_method == 'iFGSMDI':
        adv__ = targetedAttack(model1, img_t, targetCode, eps=step/255, l_inf_max=linf, bShowProcess=bShowProcess, bEarlyStop=bEarlyStop, adv_method='iFGSMDI')
    if adv_method == 'miFGSMDI':
        adv__ = targetedAttack(model1, img_t, targetCode, eps=step/255, l_inf_max=linf, bShowProcess=bShowProcess, bEarlyStop=bEarlyStop, adv_method='miFGSMDI')
    if adv_method == 'ori':
        adv__ = Variable(img_t)
    if adv_method == 'FGSM':
        adv__ = targetedAttack(model1, img_t, targetCode, eps=step / 255, l_inf_max=1, bShowProcess=bShowProcess,
                               bEarlyStop=bEarlyStop, adv_method='iFGSM')

    return adv__

def get_target_retrival_result():

    return

def get_target_imgs(j_index_matrix, test_true_label_y, i_index_set, dset_database, bSave = True, path_target_imgs = None):
    # returns the target images as an i_max*j_max matrix
    i_max = i_index_set.shape[0]
    j_max = j_index_matrix.shape[1]
    target_img_mat = np.zeros([i_max, j_max, 3, 224, 224])
    for i in range(i_max):
        for j in range(j_max):
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
            target_img = dset_database[j_index_set[j]][0].cpu().numpy()
            target_img_mat[i, j] = target_img
    if bSave and path_target_imgs is not None:
        np.save(path_target_imgs, target_img_mat)

    return target_img_mat

def get_hold_l_inf_adv(adv_imgs, ori_imgs, l_inf_max = 32):
    # make sure l_inf(adv_imgs, ori_imgs) <= l_inf_max, which is required by ADV's usabilty.
    noise_img = adv_imgs - ori_imgs
    noise_img[noise_img>=l_inf_max] = l_inf_max
    noise_img[noise_img<=-l_inf_max] = -l_inf_max
    return ori_imgs+noise_img

def is_adv_attack_success(model, adv_img, target_label, database_code, multi_label, checking_method=''):
    # not finished,
    # it may always returns true
    if len(adv_img==3):
        adv_img_variable = Variable(torch.Tensor(adv_img)).cuda().unsqueeze(0)
    elif len(adv_img == 4):
        adv_img_variable = Variable(torch.Tensor(adv_img)).cuda()

    if checking_method == '':
        from myRetrieval import get_img_result_label_flags
        img_result_label_flags = get_img_result_label_flags(adv_img_variable, model, database_code, multi_label, threshold = 5)
        if img_result_label_flags[int(target_label)] == 1:
            return True
        else:
            return False
    return True

def estimate_subspace_size(adv_img, ori_img, model, target_label, database_code, multi_label, noise_number = 10, noise_level = 10, lower_iter_max = 10, upper_iter_max = 10):
    # Test passed.
    # Hard-coded
    noised_adv_imgs = np.zeros([noise_number, 3, 224, 224])

    # generate qualified noised adv images
    i = 0
    while i < noise_number:
        random_noise = np.random.randint(-noise_level, noise_level + 1, adv_img.shape).astype(float)
        noised_adv = adv_img + random_noise / 255
        if is_adv_attack_success(model, adv_img, target_label, database_code, multi_label):
            noised_adv_imgs[i] = noised_adv
            i += 1

    # calculate the upper-bound and lower-bound of the perturbation ratio for each noised adv
    perturbation_ratio_bound = np.ones([noise_number, 2])
    for i in range(noise_number):
        noised_perturbation = noised_adv_imgs[i] - ori_img
        # lower-bound, simple version using 0.1 interval
        lower_iter = 0
        lower_ratio_array = np.arange(lower_iter_max).astype(float)/10.0 + 0
        while lower_iter < lower_iter_max:
            #print "lower_iter:", lower_iter
            noised_adv = ori_img + lower_ratio_array[lower_iter] * noised_perturbation
            if is_adv_attack_success(model, noised_adv, target_label, database_code, multi_label):
                perturbation_ratio_bound[i, 0] = lower_ratio_array[lower_iter]
                break
            else:
                lower_iter += 1
        # upper-bound, simple version using 0.1 interval
        upper_iter = upper_iter_max - 1
        upper_ratio_array = np.arange(upper_iter_max).astype(float)/10.0 + 1.1
        while upper_iter >= 0:
            noised_adv = ori_img + upper_ratio_array[upper_iter] * noised_perturbation
            if is_adv_attack_success(model, noised_adv, target_label, database_code, multi_label):
                perturbation_ratio_bound[i, 1] = upper_ratio_array[upper_iter]
                break
            else:
                upper_iter -= 1

    return perturbation_ratio_bound


def get_target_targetedRetrievalNum(net1, net2, adv_method, step, linf, i_max, j_max, dis_method, job_dataset='', allowLoad=True):
    # returns the targeted retrieval number of original targets imgs.
    # The result has no relation with the adv method
    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s.npy' % (
        net1, dis_method)
    if os.path.exists(target_targeted_retrieval_num_path) and allowLoad:
        target_targetedNum_mat = np.load(target_targeted_retrieval_num_path)
        print('target_targeted_retrieval_num_path already exists in:', target_targeted_retrieval_num_path)
        return target_targetedNum_mat
    else:
        #npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
        #npy_path = 'save_for_load/' + net1 + npy_name
        path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net1)
        path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net2)
        dset_test, dset_database = load_dset_params(job_dataset)
        model1, snapshot_path, query_path, database_path = load_net_params(net1)

        tmp = np.load(database_path)
        _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
        test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
        test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                        max_dis=18, min_dis=12)


        id_size = test_true_id_x.shape[0]
        print('id size:',id_size)
        i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]

        #inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])
        j_index_matrix = get_unique_index(code, multi_label, j_max)

        target_img_mat = get_target_imgs(j_index_matrix, test_true_label_y, i_index_set, dset_database)

        from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class
        inputs_targets = Variable(torch.Tensor(target_img_mat).cuda(), requires_grad=True)

        img_num_by_class_target = get_img_num_by_class_from_img_batch(inputs_targets, model1, code, multi_label,
                                                                      threshold=5, batch_size=16)
        target_targetedNum_mat = np.zeros([i_max, j_max])
        for i in range(i_max):
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
            label_targeted = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_target[i], label_targeted)
            target_targetedNum_mat[i] = img_num_target_targeted
        print('new blackTargetedNum:', target_targeted_retrieval_num_path)
        np.save(target_targeted_retrieval_num_path, target_targetedNum_mat)
        return target_targetedNum_mat


def get_adv_black_retrieval_result(net1, net2, adv_method, step, linf, i_max, j_max, dis_method, job_dataset='', threshold=5, batch_size=8, allowLoad=True):
    # save/load and return  the black box retrieval result for specific adv_imgs
    # the adv_imgs is loaded in this function

    path_blackTargetedNum = 'save_for_load/distanceADVRetrieval/%s/targetedNum_white_%s_black_%s_step%1.1f_linf%d_%s.npy' % (
        adv_method, net1, net2, step, linf, dis_method)
    if os.path.exists(path_blackTargetedNum) and allowLoad:
        adv_black_retrieval_result = np.load(path_blackTargetedNum)
        print('path_blackTargetedNum already exists in:', path_blackTargetedNum)
        return adv_black_retrieval_result

    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net2)
    dset_test, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    tmp2 = np.load(database_path2)
    _, code2, multi_label2 = tmp2['arr_0'], tmp2['arr_1'], tmp2['arr_2']
    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                   max_dis=18, min_dis=12)
    id_size = test_true_id_x.shape[0]
    print('id size:', id_size)
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]

    inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    print('load adv_imgs from:', npy_path)
    adv_imgs = np.load(npy_path)

    inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())
    better_img_num_result = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2,
                                                                     threshold=threshold, batch_size=batch_size)
    label_targeted = np.zeros([i_max, j_max])

    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        label_targeted[i] = label_targeted_i

    adv_black_retrieval_result = get_targeted_from_all_class(better_img_num_result, label_targeted)
    print('new blackTargetedNum:', path_blackTargetedNum)
    np.save(path_blackTargetedNum, adv_black_retrieval_result)

    return adv_black_retrieval_result


def imgs_to_file(adv_imgs='', pics_root_path=''):
    from publicFunctions import load_net_inputs, load_net_params, load_dset_params
    import matplotlib.pyplot as plt
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net2)
    dset_test, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                   max_dis=18, min_dis=12)
    id_size = test_true_id_x.shape[0]
    print('id size:', id_size)
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    adv_imgs = np.load(npy_path)
    pics_root_path = './save_for_load/pics/%s_imgs_step%1.1f_linf%d_%dx%d_%s/' %(adv_method, step, linf, i_max, j_max, dis_method)
    if not os.path.exists(pics_root_path):
        os.makedirs(pics_root_path)

    for i in range(i_max):
        for j in range(j_max):
            print('i,j:',i,j)
            file_name_full = pics_root_path + '/' + 'i%s_j%s.jpg'%(str(i), str(j))
            img_array = np.moveaxis(adv_imgs[i, j], 0, -1)
            plt.imsave(file_name_full, img_array)

    return


def get_adv_code_diff_to_targeted(net1, adv_method, step, linf, i_max, j_max, dis_method, allowLoad=True):

    path_whiteCodeDiff = 'save_for_load/distanceADVRetrieval/%s/whiteCodeDiff_white_%s_step%1.1f_linf%d_%s.npy' % (
        adv_method, net1, step, linf, dis_method)
    if os.path.exists(path_whiteCodeDiff) and allowLoad:
        adv_code_diff_to_targeted = np.load(path_whiteCodeDiff)
        print('whiteCodeDiff already exists in:', path_whiteCodeDiff)
        return adv_code_diff_to_targeted

    from publicFunctions import load_net_inputs, load_net_params, load_dset_params
    from myRetrieval import get_query_code_batch
    import matplotlib.pyplot as plt
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy' % (net1)

    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                   max_dis=18, min_dis=12)
    id_size = test_true_id_x.shape[0]
    print('id size:', id_size)
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    adv_imgs = np.load(npy_path)


    code_targeted = np.zeros([i_max, j_max, 48])
    adv_code_mat = np.zeros([i_max, j_max, 48])

    adv_code_diff_to_targeted = np.zeros([i_max, j_max])

    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([code[j_index_set[j]] for j in range(j_max)])
        code_targeted[i] = label_targeted_i

        img_inputs = Variable(torch.Tensor(adv_imgs[i])).cuda()
        adv_code = get_query_code_batch(img_inputs, model1, batch_size=16)
        adv_code_mat[i] = adv_code
        adv_code_diff_to_targeted[i] = np.linalg.norm(adv_code - label_targeted_i, ord=0, axis=1)

    np.save(path_whiteCodeDiff, adv_code_diff_to_targeted)
    return adv_code_diff_to_targeted

def main_func():

    from publicFunctions import load_net_inputs, load_net_params, load_dset_params
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name
    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net2)
    dset_test, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                    max_dis=18, min_dis=12)
    id_size = test_true_id_x.shape[0]
    print('id size:',id_size)
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]

    inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    adv_imgs = np.load(npy_path)
    ori_imgs = inputs_ori_tensor.cpu().numpy()
    # imgs_test = np.load('save_for_load/imgs_test.npy')
    # target_img_mat = np.load('save_for_load/target_imgs.npy')
    target_img_mat = get_target_imgs(j_index_matrix, test_true_label_y, i_index_set, dset_database)

    from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class
    inputs_targets = Variable(torch.Tensor(target_img_mat).cuda(), requires_grad=True)
    img_num_by_class_target = get_img_num_by_class_from_img_batch(inputs_targets, model1, code, multi_label,
                                                                  threshold=5, batch_size=16)
    target_targetedNum_mat = np.zeros([i_max, j_max])
    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_target[i], label_targeted)
        target_targetedNum_mat[i] = img_num_target_targeted

    target_targeted_retrieval_num_path = './save_for_load/%s/target_targetedRetrievalNum_%s_%s.npy' % (net1, adv_method, dis_method)
    np.save(target_targeted_retrieval_num_path, target_targetedNum_mat)

    '''
    for i in range(i_max):
        for j in range(j_max):
            j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
            target_img = target_img_mat[i, j]
    #target_result = get_target_retrival_result(model1, )
    
    i, j = 3, 9
    ori_img = ori_imgs[i]
    adv_img = adv_imgs[i,j]
    j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
    label_targeted = np.array([database_label[j_index_set[j]] for j in range(j_max)])
    target_label =  label_targeted[j]
    perturbation_ratio_bound = estimate_subspace_size(adv_img, ori_img, model1, target_label, code, database_label)
    '''

    return


if __name__ == "__main__":
    dis_method_value = ['cW', 'fW', 'cB', 'fB', 'cWcB', 'cWfB', 'fWcB', 'fWfB']
    dis_method = dis_method_value[0]
    i_max, j_max = 64, 32
    adv_method = 'iFGSM'
    step = 1.0
    linf = 32
    job_dataset = 'imagenet'
    net1 = 'DenseNet161'
    net2 = 'ResNet152'
    noiid = False

    #main_func()
    #adv_black_retrieval_result = get_adv_black_retrieval_result(net1, net2, adv_method, step, linf, i_max, j_max, dis_method, job_dataset='imagenet', threshold=5, batch_size=8)
    #imgs_to_file()
    code_diff = get_adv_code_diff_to_targeted(net1, adv_method, step, linf, i_max, j_max, dis_method, allowLoad=True)

