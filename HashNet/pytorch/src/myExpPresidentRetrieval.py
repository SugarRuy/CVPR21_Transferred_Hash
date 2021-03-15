import os

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from PIL import Image
from .publicFunctions import load_net_inputs, load_net_params, load_dset_params
from .myRetrieval import get_retrieval_result_by_query_code, count_by_query_result, get_img_num_by_class_from_img_batch, get_targeted_from_all_class, get_query_code_batch, get_query_result_num_by_class
from .myExtractCodeLabel import trans_train_resize_imagenet
from .myExpForPapers_nag import func_targetedAttack_nag
from .myGetAdv import targetedAttack

def get_query_avg_dis(query_code, code, multi_label):
    query_avg_dis = np.zeros([100])
    for i in range(100):
        hamming_dis = np.linalg.norm(query_code - code[multi_label == i], ord=0, axis=-1)
        query_avg_dis[i] = hamming_dis.mean()
    return query_avg_dis

def get_query_dis(query_code, code):

    return

def exp_president_query():

    from .myGetAdv import get_trans_img
    _, dset_database = load_dset_params(job_dataset)

    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    #
    img_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/trump005.jpg'
    #img_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/biden002.jpg'
    t = trans_train_resize_imagenet()
    img_pil = Image.open(img_path)
    img_tensor = t(img_pil)
    inputs = Variable(img_tensor.cuda()).unsqueeze(0)

    query_code = np.sign(model1(inputs).cpu().data.numpy())
    query_avg_dis = get_query_avg_dis(query_code, code, multi_label)

    print(np.where(query_avg_dis<18))
    show_K_random_imgs_from_class_N(dset_database, multi_label, N=64, K=32)

    #query_result = get_retrieval_result_by_query_code(query_code, code, threshold=5).reshape([query_code.shape[0], -1])
    query_result = get_retrieval_result_by_query_code(query_code, code, threshold=5)

    img_num_by_class = count_by_query_result(query_result, multi_label)
    #show_random_imgs(dset_database, query_result, K=32)
    show_100_class_imgs(dset_database, multi_label)
    return query_code


def show_K_random_imgs_from_class_N(dset_database, multi_label, N=83, K=32):
    np.random.seed(0)


    index_in_N = np.where(multi_label==N)[0]
    #index_in_N_size_K = np.random.choice(index_in_N, K)
    index_in_N_size_K = index_in_N[:K].astype(int)
    for i in range(K):
        img_np = np.moveaxis(dset_database[index_in_N_size_K[i]][0].cpu().numpy(), 0, -1)
        plt.subplot(4, 8, i+1)
        plt.imshow(img_np)

def show_100_class_imgs(dset_database, multi_label):

    K = 1
    for i in range(100):
        N = i
        index_in_N = np.where(multi_label == N)[0]
        index_in_N_size_K = index_in_N[:K].astype(int)
        img_np = np.moveaxis(dset_database[index_in_N_size_K[0]][0].cpu().numpy(), 0, -1)
        plt.subplot(10, 10, i+1)
        plt.imshow(img_np)

def show_random_imgs(dset_database, query_result, K=32):

    print("Size of Query Result:", query_result.shape[0])
    np.random.seed(0)
    showing_size = query_result.shape[0] if query_result.shape[0]<=32 else K
    #random_index = np.random.sample(query_result, showing_size)
    #random_index_in_result = query_result[random_index].astype(int)
    random_index_in_result = np.arange(showing_size)
    for i in range(showing_size):
        index_i = random_index_in_result[i]
        img_np = np.moveaxis(dset_database[index_i][0].cpu().numpy(), 0, -1)
        plt.subplot(4,8,i+1)
        plt.imshow(img_np)


def func_get_adv_by_method(model, img_t, targetCode, eps=1.0 / 255, l_inf_max=32, threshold_stop=2, decay_factor=1.0,
                                t_prob=0.5, bShowProcess=False, bEarlyStop=False, random_noise_level=16, var_lambda = 1.0,
                                noise_distribution='uniform', adv_method='iFGSM'):
    if adv_method == 'NAG':
        adv__ = func_targetedAttack_nag(model, img_t, targetCode, eps=eps,
                                        l_inf_max=l_inf_max, decay_factor=1.0, t_prob=0.5,
                                        var_lambda=var_lambda, random_noise_level=float(random_noise_level) / 255,
                                        noise_distribution=noise_distribution)
    if adv_method == 'iFGSM':
        adv__ = targetedAttack(model, img_t, targetCode, eps=eps, l_inf_max=l_inf_max, bShowProcess=bShowProcess,
                               bEarlyStop=bEarlyStop, adv_method='iFGSM')
    if adv_method == 'iFGSMDI':
        adv__ = targetedAttack(model, img_t, targetCode, eps=eps, l_inf_max=l_inf_max, bShowProcess=bShowProcess,
                               bEarlyStop=bEarlyStop, adv_method='iFGSMDI')
    if adv_method == 'miFGSMDI':
        adv__ = targetedAttack(model, img_t, targetCode, eps=eps, l_inf_max=l_inf_max, bShowProcess=bShowProcess,
                               bEarlyStop=bEarlyStop, adv_method='miFGSMDI')
    if adv_method == 'ori':
        adv__ = Variable(img_t)
    if adv_method == 'FGSM':
        adv__ = targetedAttack(model, img_t, targetCode, eps=eps, l_inf_max=1, bShowProcess=bShowProcess,
                               bEarlyStop=bEarlyStop, adv_method='iFGSM')

    return adv__

def func_get_target_advs_by_target_label(model, img_t, label_targeted, code, multi_label, size_target_code, adv_method='NAG'):
    # args incompleted, not import-able.

    np.random.seed(0)
    adv_imgs = np.zeros([size_target_code, 3, 224, 224])
    targeted_code_unique = np.unique(code[multi_label==label_targeted], axis=0)
    #random_index = np.random.sample(size_target_code, np.arange(targeted_code_unique.shape[0]))
    random_index = np.random.choice(targeted_code_unique.shape[0], size_target_code, replace=False)
    for i in range(size_target_code):
        targetCode = targeted_code_unique[random_index[i]]
        adv__ = func_get_adv_by_method(model, img_t, targetCode, random_noise_level=32, var_lambda=var_lambda, adv_method=adv_method)
        adv_imgs[i] = adv__.cpu().data.numpy()
    return adv_imgs, targeted_code_unique[random_index]


def exp_president_attack():
    step_size, step_max = step, linf
    _, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    # trump005 is a good example for transferability with all methods
    #img_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/trump005.jpg'

    root_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/'

    if president == 'biden':
        img_name = 'biden001'
        img_path = root_path + '%s.jpg'%(img_name)
    else:
        img_name = 'trump006'
        img_path = root_path + '%s.jpg'%(img_name)
    t = trans_train_resize_imagenet()
    img_pil = Image.open(img_path)
    img_tensor = t(img_pil)
    inputs = Variable(img_tensor.cuda()).unsqueeze(0)

    # query with the original image on the whitebox to see what it is from the beginning
    query_code_white = np.sign(model1(inputs).cpu().data.numpy())
    query_avg_dis_white = get_query_avg_dis(query_code_white, code, multi_label)
    close_white_class = np.where(query_avg_dis_white < 18)
    print(close_white_class)
    query_result_white = get_retrieval_result_by_query_code(query_code_white, code, threshold=5)

    # query with the original image on the blackbox to see what it is from the beginning
    query_code_black = np.sign(model2(inputs).cpu().data.numpy())
    query_result_black = get_retrieval_result_by_query_code(query_code_black, code2, threshold=5)


    label_targeted = close_white_class[0][targeted_class_index]
    size_target_code = 32

    #adv_img = func_get_adv_by_method()
    adv_imgs, targetCodes = func_get_target_advs_by_target_label(model1, img_tensor, label_targeted, code, multi_label, size_target_code)
    inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())
    img_num_by_class_adv_target = get_query_result_num_by_class(targetCodes, code, multi_label, threshold=5)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_target,  np.ones([size_target_code]) * label_targeted)

    print(img_name)
    print("WhiteClose:", close_white_class, "Targeted Label:",label_targeted)
    print("Original Retrieval Result(on WhiteBox):", query_result_white.size)
    print("Original Retrieval Result in Target Class:", (multi_label[query_result_white.reshape([-1])]==label_targeted).sum())
    print("Original Retrieval Result(on BlackBox):", query_result_black.size)
    print("Original Retrieval Result in Target Class:", (multi_label[query_result_black.reshape([-1])] == label_targeted).sum())

    print("Targeted Code Result:", img_num_target_targeted.astype(int))

    '''
    adv_code = get_query_code_batch(inputs_adv, model1)
    adv_query_avg_dis = get_query_avg_dis(adv_code[0], code, database_label)
    adv_close_white_class = np.where(adv_query_avg_dis < 18)
    print adv_close_white_class
    print np.linalg.norm(adv_code-query_code, ord=0, axis=-1)
    print [np.linalg.norm(adv_code[i]-targetCodes[i], ord=0, axis=-1) for i in range(size_target_code)]
    query_result = get_retrieval_result_by_query_code(adv_code, code, threshold=5)
    '''



    img_num_by_class_adv_white = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label, threshold=5, batch_size=8)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_white, np.ones([size_target_code]) * label_targeted)
    print("White Box Targeted Result:", img_num_target_targeted.astype(int))
    print("White Box Success:", (img_num_target_targeted > 10).sum())

    img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2, threshold=5, batch_size=8)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, np.ones([size_target_code])*label_targeted)
    print("Black Box Targeted Result:", img_num_target_targeted.astype(int))
    print("Black Box Success:", (img_num_target_targeted>10).sum())


    saved_path = root_path + 'saved_img/%s/'%(president)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    #save_ori_img()
    ori_img_path = saved_path + '%s_original.jpg'%(img_name)
    ori_img_np = img_tensor.numpy()
    plt.imsave(ori_img_path, np.moveaxis(ori_img_np, 0, -1))

    # save_adv_imgs()
    size_adv_imgs = adv_imgs.shape[0]
    for i in range(size_adv_imgs):
        adv_img_path = saved_path + '%s_adv_%d_%d.jpg' % (img_name, label_targeted, i)
        adv_img_np = adv_imgs[i]
        plt.imsave(adv_img_path, np.moveaxis(adv_img_np, 0, -1))

    #save_white_close_imgs()
    #size_close_white = close_white_class.size
    size_to_save_close_white = 8
    close_white_indices = np.where(multi_label == label_targeted)[0]
    for i in range(size_to_save_close_white):
        cW_img_path = saved_path + 'cW_%d_%d.jpg'%(label_targeted, i)

        close_white_img_np = dset_database[close_white_indices[i]][0].cpu().numpy()
        plt.imsave(cW_img_path, np.moveaxis(close_white_img_np, 0, -1))


    # calculate success number:
    adv_code_black = get_query_code_batch(inputs_adv, model2)
    query_result_black = get_retrieval_result_by_query_code(adv_code_black, code2, threshold=5)
    all_result_black_reshape = np.concatenate([query_result_black[i] for i in range(query_result_black.shape[0])]).reshape([-1])
    query_result_white_unique = np.unique(all_result_black_reshape)
    print("success:%d, close: %d, ratio:%f"%(query_result_white_unique.size, close_white_indices.size, float(query_result_white_unique.size)/close_white_indices.size))


def see_class_name():
    return


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpForPapers')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='miFGSMDI', help="adv method")
    parser.add_argument('--net1', type=str, default='ResNext101_32x4d', help="net1") # ResNext101_32x4d
    parser.add_argument('--net2', type=str, default='ResNet152', help="net2")
    parser.add_argument('--allowLoad', type=str, default='True', help="is Loading allowed")
    parser.add_argument('--linf', type=int, default=32, help="linf")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    adv_method = args.adv_method
    dis_method = args.dis_method
    net1 = args.net1
    net2 = args.net2
    i_max, j_max = 64, 32
    step = 1.0
    linf = args.linf
    job_dataset = 'imagenet'
    noiid = False


    net1 = 'ResNext101_32x4d' #'ResNet152'
    net2 = 'ResNet152' #'ResNext101_32x4d'
    president = 'trump'
    targeted_class_index = 2
    #out_code = exp_president_query()
    #exp_president_attack()

    var_lambda = 1.0

    step_size, step_max = step, linf
    _, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    # trump005 is a good example for transferability with all methods
    #img_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/trump005.jpg'

    root_path = '/home/yxiao/Dropbox/Yanru Research/Ads-Blackbox/Exp_record/president/'

    if president == 'biden':
        img_name = 'biden001'
        img_path = root_path + '%s.jpg'%(img_name)
    else:
        img_name = 'trump003'
        img_path = root_path + '%s.jpg'%(img_name)
    t = trans_train_resize_imagenet()
    img_pil = Image.open(img_path)
    img_tensor = t(img_pil)
    inputs = Variable(img_tensor.cuda()).unsqueeze(0)

    # query with the original image on the whitebox to see what it is from the beginning
    query_code_white = np.sign(model1(inputs).cpu().data.numpy())
    query_avg_dis_white = get_query_avg_dis(query_code_white, code, multi_label)
    close_white_class = np.where(query_avg_dis_white < 18)
    print(close_white_class)
    query_result_white = get_retrieval_result_by_query_code(query_code_white, code, threshold=5)

    # query with the original image on the blackbox to see what it is from the beginning
    query_code_black = np.sign(model2(inputs).cpu().data.numpy())
    query_result_black = get_retrieval_result_by_query_code(query_code_black, code2, threshold=5)


    label_targeted = close_white_class[0][targeted_class_index]
    size_target_code = 32
    close_white_indices = np.where(multi_label == label_targeted)[0]

    #adv_img = func_get_adv_by_method()
    adv_imgs, targetCodes = func_get_target_advs_by_target_label(model1, img_tensor, label_targeted, code, multi_label, size_target_code)
    inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())
    img_num_by_class_adv_target = get_query_result_num_by_class(targetCodes, code, multi_label, threshold=5)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_target,  np.ones([size_target_code]) * label_targeted)

    print(img_name)
    print("WhiteClose:", close_white_class, "Targeted Label:",label_targeted)
    print("Original Retrieval Result(on WhiteBox):", query_result_white.size)
    print("Original Retrieval Result in Target Class:", (multi_label[query_result_white.reshape([-1])]==label_targeted).sum())
    print("Original Retrieval Result(on BlackBox):", query_result_black.size)
    print("Original Retrieval Result in Target Class:", (multi_label[query_result_black.reshape([-1])] == label_targeted).sum())

    print("Targeted Code Result:", img_num_target_targeted.astype(int))

    '''
    adv_code = get_query_code_batch(inputs_adv, model1)
    adv_query_avg_dis = get_query_avg_dis(adv_code[0], code, database_label)
    adv_close_white_class = np.where(adv_query_avg_dis < 18)
    print adv_close_white_class
    print np.linalg.norm(adv_code-query_code, ord=0, axis=-1)
    print [np.linalg.norm(adv_code[i]-targetCodes[i], ord=0, axis=-1) for i in range(size_target_code)]
    query_result = get_retrieval_result_by_query_code(adv_code, code, threshold=5)
    '''


    # retrieve to see if the img_num of targeted class is small or zero
    img_num_by_class_adv_white = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label, threshold=5, batch_size=8)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_white, np.ones([size_target_code]) * label_targeted)
    print("White Box Targeted Result:", img_num_target_targeted.astype(int))
    print("White Box Success:", (img_num_target_targeted > 10).sum())

    img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2, threshold=5, batch_size=8)
    img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, np.ones([size_target_code])*label_targeted)
    print("Black Box Targeted Result:", img_num_target_targeted.astype(int))
    print("Black Box Success:", (img_num_target_targeted>10).sum())


    saved_path = root_path + 'saved_img/%s/'%(president)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    #save_ori_img()
    ori_img_path = saved_path + '%s_original.jpg'%(img_name)
    ori_img_np = img_tensor.numpy()
    plt.imsave(ori_img_path, np.moveaxis(ori_img_np, 0, -1))

    # save_adv_imgs()
    size_adv_imgs = adv_imgs.shape[0]
    for i in range(size_adv_imgs):
        adv_img_path = saved_path + '%s_adv_%d_%d.jpg' % (img_name, label_targeted, i)
        adv_img_np = adv_imgs[i]
        plt.imsave(adv_img_path, np.moveaxis(adv_img_np, 0, -1))

    #save_white_close_imgs()
    #size_close_white = close_white_class.size
    size_to_save_close_white = 8
    for i in range(size_to_save_close_white):
        cW_img_path = saved_path + 'cW_%d_%d.jpg'%(label_targeted, i)

        close_white_img_np = dset_database[close_white_indices[i]][0].cpu().numpy()
        plt.imsave(cW_img_path, np.moveaxis(close_white_img_np, 0, -1))


    # calculate success number:
    adv_code_black = get_query_code_batch(inputs_adv, model2)
    query_result_black = get_retrieval_result_by_query_code(adv_code_black, code2, threshold=5)
    all_result_black_reshape = np.concatenate([query_result_black[i] for i in range(query_result_black.shape[0])]).reshape([-1])
    query_result_white_unique = np.unique(all_result_black_reshape)
    query_result_white_unique = query_result_white_unique[multi_label[query_result_white_unique]==label_targeted]
    print("success:%d, close: %d, ratio:%f"%(query_result_white_unique.size, close_white_indices.size, float(query_result_white_unique.size)/close_white_indices.size))
