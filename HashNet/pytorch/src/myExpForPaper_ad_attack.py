
import numpy as np
import torch
from torch.autograd import Variable

from publicFunctions import load_net_inputs, NetworkSettings


from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class, get_query_result_num_by_class, get_query_code_batch, get_retrieval_result_by_query_code
from myExpForPapers_nag import func_targetedAttack_nag



def get_query_avg_dis(query_code, code, multi_label):
    if len(query_code.shape) == 1:
        query_avg_dis = np.zeros([100])
        for i in range(100):
            hamming_dis = np.linalg.norm(query_code - code[multi_label == i], ord=0, axis=-1)
            query_avg_dis[i] = hamming_dis.mean()
        return query_avg_dis
    else:
        query_avg_dis = np.zeros([query_code.shape[0], 100])
        for i in range(100):
            index_i = multi_label==i
            for j in range(query_avg_dis.shape[0]):
                hamming_dis = np.linalg.norm(query_code[j] - code[index_i], ord=0, axis=-1)
                query_avg_dis[j, i] = hamming_dis.mean()
        return query_avg_dis

def cal_retrievable_rate(inputs_adv, model2, code2, multi_label, label_target_mat):
    #adv_code_black = np.zeros([inputs_adv.shape[0], inputs_adv.shape[1], 48])
    retrievable_rate = np.ones([inputs_adv.shape[0]])
    for i in range(inputs_adv.shape[0]):
        label_targeted = label_target_mat[i,0]
        target_class_size = (multi_label==label_targeted).sum()
        adv_code_black = get_query_code_batch(inputs_adv[i], model2)
        query_result_black = get_retrieval_result_by_query_code(adv_code_black, code2, threshold=5)

        all_result_black_reshape = np.concatenate([query_result_black[j] for j in range(query_result_black.shape[0])]).reshape([-1])
        query_result_white_unique = np.unique(all_result_black_reshape)
        query_result_white_unique = query_result_white_unique[multi_label[query_result_white_unique]==label_targeted]
        print("success:%d, close: %d, ratio:%f" % (query_result_white_unique.size, target_class_size, float(query_result_white_unique.size) / target_class_size))
        retrievable_rate[i] = float(query_result_white_unique.size) / target_class_size
    return retrievable_rate


def print_class_onehot():
    #for single
    #ori_indexes = np.array([0, 2, 4, 12, 20])
    #tar_indexes = np.array([0, 0, 0, 0, 0])
    ori_indexes = np.array([1, 1, 1, 6, 6, 6, 7, 10, 13, 13, 16, 17, 24, 24])
    tar_indexes = np.array([0, 2, 3, 1, 4, 15, 6, 7, 10, 13, 3 , 13, 6, 4])
    f = open("../data/imagenet/database.txt")
    lines = f.readlines()
    f.close()
    for i in range(ori_indexes.shape[0]):
        ori_index = ori_indexes[i]
        tar_index = tar_indexes[i]
        targetCode = target_code_mat[ori_index, tar_index]
        index_i_j = np.where(np.all(code == targetCode, axis=1))[0][0]
        class_onehot = dset_database[index_i_j][1]
        print(('ori: %d, target:%d'%(ori_index, tar_index)))
        print((lines[index_i_j]))
        #print class_onehot


def visualize_AD_imgs():
    selected_save_size = 8

    root_path = './save_for_load/ad_attack/%s/'%(targetType)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    plt.figure('ori')
    for i in range(inputs_AD.shape[0]):
        path_save_ori_img = root_path + 'ori_img_%d.jpg' % (i)
        plt.imsave(path_save_ori_img, np.moveaxis(inputs_AD[i].cpu().data.numpy(),  0, -1))
        #plt.subplot(1, ad_size, i+1)
        #plt.imshow(np.moveaxis(inputs_AD[i].cpu().data.numpy(),  0, -1))
    for i in range(adv_imgs.shape[0]):
        for j in range(adv_imgs.shape[1]):
            path_save_adv_img = root_path + 'adv_img_%d_%d.jpg'%(i, j)
            plt.imsave(path_save_adv_img, np.moveaxis(adv_imgs[i,j], 0, -1))
            adv_diff = (adv_imgs[i,j] -  inputs_AD[i].cpu().data.numpy())
            print('adv_diff:std:%f, avg: %f'%(adv_diff.std(), adv_diff.mean()))
            adv_diff_shift = adv_diff * 4.0 + 0.5
            path_save_pert_shift_img = root_path + 'pert_shift_%d_%d.jpg'%(i, j)
            plt.imsave(path_save_pert_shift_img, np.moveaxis(adv_diff_shift, 0, -1))
                #plt.imshow(np.moveaxis(adv_diff_shift, 0, -1))

        #plt.hist(adv_diff.reshape([-1]), bins=255)
    #target_code_mat
    for i in range(adv_imgs.shape[0]):
        for j in range(adv_imgs.shape[1]):
            targetCode = target_code_mat[i, j]
            index_i_j = np.where(np.all(code==targetCode, axis=1))[0][0]
            path_save_tar_img = root_path + 'tar_img_%d_%d.jpg'%(i, j)
            tar_img = dset_database[index_i_j][0].cpu().numpy()
            plt.imsave(path_save_tar_img, np.moveaxis(tar_img, 0, -1))
    return


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='MyExpForPapers')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dis_method', type=str, default='cW', help="distance method")
    parser.add_argument('--adv_method', type=str, default='miFGSMDI', help="adv method")
    parser.add_argument('--net1', type=str, default='ResNext101_32x4d', help="net1") # ResNext101_32x4d
    parser.add_argument('--net2', type=str, default='ResNet34', help="net2")
    parser.add_argument('--allowLoad', type=str, default='True', help="is Loading allowed")
    parser.add_argument('--linf', type=int, default=32, help="linf")

    parser.add_argument('--exp_target_type', type=str, default='multi', help="Which exp to do: multi or single")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    adv_method = args.adv_method
    dis_method = args.dis_method
    net1 = args.net1
    net2 = args.net2
    i_max, j_max = 64, 32
    step = 1.0
    linf = args.linf

    exp_target_type = args.exp_target_type
    job_dataset = 'imagenet'
    noiid = True



    step_size, step_max = step, linf
    '''
    _, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    model2, snapshot_path2, query_path2, database_path2 = load_net_params(net2)
    tmp = np.load(database_path2)
    _, code2, multi_label2 = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
    '''
    from publicVariables import iters_list
    hash_bit = 48
    snapshot_iter1 = iters_list[net1]
    network_settings1 = NetworkSettings(job_dataset, hash_bit, net1, snapshot_iter1, batch_size=16)
    model1 = network_settings1.get_model()
    dset_loaders = network_settings1.get_dset_loaders()
    _, code, multi_label = network_settings1.get_out_code_label(part='database')
    dset_database = dset_loaders['database'].dataset

    snapshot_iter2 = iters_list[net2]
    network_settings2 = NetworkSettings(job_dataset, hash_bit, net2, snapshot_iter2, batch_size=16)
    model2 = network_settings2.get_model()
    _, code2, multi_label2 = network_settings2.get_out_code_label(part='database')

    ad_datapath = '../data/ad_dataset/ads/0/'
    ad_sample_size = 256
    ad_size = 32 # advertised imgs used to generate the NAG ADV.
    target_size = 16

    bVisualize = False
    targetType = exp_target_type # 'multi'single
    inputs_AD_sample = load_net_inputs(ad_datapath, 0, batch_size=ad_sample_size)


    # get white closest class
    query_code = get_query_code_batch(inputs_AD_sample, model1, batch_size=8)
    query_avg_dis_white = get_query_avg_dis(query_code, code, multi_label)
    #close_white_class = np.where(query_avg_dis_white < 18)
    if targetType == 'single':
        index_close_white_class = np.argmin(query_avg_dis_white, axis=1) # also the label
        query_avg_dis_white_closest = np.array([query_avg_dis_white[i, index_close_white_class[i]] for i in range(ad_sample_size)])

        # see if the closet class samples are included in returns.
        # If none returns, called it 'safe'.
        # We choose K(ad_size) safe AD imgs with the smallest 'query_avg_dis_white_closest'
        # index_closest_safe is the index of size 32 in range(ad_sample_size) to select the ideal data
        img_num_by_class = get_query_result_num_by_class(query_code, code, multi_label, threshold=5)
        img_num_target_targeted = get_targeted_from_all_class(img_num_by_class, index_close_white_class)

        index_safe_AD = img_num_target_targeted == 0
        index_safe_AD_by_position = np.arange(ad_sample_size)[index_safe_AD]

        index_close_white_class_safe = index_close_white_class[index_safe_AD]
        query_avg_dis_white_closet_safe = query_avg_dis_white_closest[index_safe_AD]
        index_closest_safe = index_safe_AD_by_position[np.argsort(query_avg_dis_white_closet_safe, ad_size)[:ad_size]]
        #index_closest_safe_AD = [index_closest_safe]

        # get the inputs_AD and label_target
        inputs_AD = inputs_AD_sample[index_closest_safe]
        label_target = index_close_white_class[index_closest_safe]

        # retrieve to get the AD imgs with out any return.
        img_num_by_class_adv_white = get_img_num_by_class_from_img_batch(inputs_AD, model1, code, multi_label, threshold=5, batch_size=8)
        img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_white, label_target)
        #img_num_sum = img_num_by_class_adv_white.sum(axis=0)

        # select the target code
        path_target_code_mat = './save_for_load/ad_attack/target_code_single_mat.npy'
        if os.path.exists(path_target_code_mat):
            target_code_mat = np.load(path_target_code_mat)
        else:
            target_code_mat = np.zeros([ad_size, target_size, 48])
            for i in range(ad_size):
                index_i = multi_label==label_target[i]
                target_code_tmp = code[index_i][:128]
                img_num_by_class_tmp = get_query_result_num_by_class(target_code_tmp, code, multi_label, threshold=5)
                img_num_target_targeted_tmp = get_targeted_from_all_class(img_num_by_class_tmp, np.ones([128])*label_target[i])
                code_tmp = target_code_tmp[img_num_target_targeted_tmp>=100]
                target_code_mat[i] = code_tmp[:target_size]
            np.save(path_target_code_mat, target_code_mat)

        # generate the target adv of inputs AD
        path_adv_imgs = './save_for_load/ad_attack/adv_imgs_single.npy'

        if os.path.exists(path_adv_imgs):
            adv_imgs = np.load(path_adv_imgs)
        else:
            adv_imgs = np.zeros([ad_size, target_size, 3, 224, 224])
            var_lambda = 1.0
            for i in range(ad_size):
                img_t = inputs_AD[i].cpu().data
                for j in range(target_size):
                    print(i,j)
                    targetCode = target_code_mat[i, j]
                    adv__ = func_targetedAttack_nag(model1, img_t, targetCode, eps=step_size / 255,
                                                    l_inf_max=step_max, decay_factor=1.0, t_prob=0.5,
                                                    var_lambda=var_lambda, random_noise_level=float(32.0) / 255,
                                                    noise_distribution='uniform')
                    adv_imgs[i, j] = adv__.cpu().data.numpy()
            np.save(path_adv_imgs, adv_imgs)
        if not bVisualize:
            # retrieve with the adv_imgs to see the success rate
            label_target_mat = np.ones([ad_size, target_size])
            for i in range(ad_size):
                label_target_mat[i] = np.ones([target_size]) * label_target[i]
            inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())

            img_num_by_class_adv_white = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label, threshold=5, batch_size=8)
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_white, label_target_mat)
            print("White Box Targeted Result:", img_num_target_targeted.astype(int))
            print("White Box Success:", (img_num_target_targeted > 10).sum())

            img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2, threshold=5, batch_size=8)
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, label_target_mat)
            print("Black Box Targeted Result:", img_num_target_targeted.astype(int))
            print("Black Box Success:", (img_num_target_targeted>10).sum())

            print(img_num_by_class_adv_black.sum(axis=2).astype(int))

            from scipy import io
            io.savemat('./save_for_load/ad_attack/img_num_target_targeted_single_%s.mat'%(net2), {'img_num_target_targeted_single_%s'%(net2):img_num_target_targeted} )
            a = np.array([(multi_label == label_target_mat[ii, 0]).sum() for ii in range(inputs_adv.shape[0])])
            print(a)
            retrievable_rate = cal_retrievable_rate(inputs_adv, model2, code2, multi_label, label_target_mat)
            print(retrievable_rate.mean(), retrievable_rate.std())

            retrievable_rate_accu = np.zeros([ad_size, target_size])
            for i in range(target_size):
                retrievable_rate_i = cal_retrievable_rate(inputs_adv[:,0:i+1], model2, code2, multi_label, label_target_mat)
                retrievable_rate_accu[:,i] = retrievable_rate_i

            io.savemat('./save_for_load/ad_attack/retrievable_rate_single_%s.mat'%(net2), {'retrievable_rate_single': retrievable_rate, 'retrievable_rate_accu': retrievable_rate_accu})
            import matplotlib.pyplot as plt
            plt.hist(retrievable_rate)

            # check if the original AD transfers in the black-box
            img_num_by_class_ori_black = get_img_num_by_class_from_img_batch(inputs_AD, model2, code2, multi_label2, threshold=5, batch_size=8)
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_ori_black, label_target_mat[:,0])
            print("Black Box Targeted Result:", img_num_target_targeted.astype(int))
            print("Black Box Success:", (img_num_target_targeted>10).sum())
        else:
            visualize_AD_imgs() #this function only runs once by returning if a dir exists.

    elif targetType=='multi':
        img_num_by_class = get_query_result_num_by_class(query_code, code, multi_label, threshold=5)
        index_class_safe = img_num_by_class == 0
        # choose the target label
        label_target_mat = np.ones([ad_size, target_size])
        for i in range(ad_size):
            class_safe_i = np.arange(index_class_safe[i].shape[0])[index_class_safe[i]]
            index_closest_safe = np.argsort(query_avg_dis_white[i, class_safe_i])[:target_size]
            label_target_mat[i] = class_safe_i[index_closest_safe]

        label_target_mat = label_target_mat.astype(int)
        # choose the target code
        path_target_code_mat = './save_for_load/ad_attack/target_code_multi_mat.npy'
        if os.path.exists(path_target_code_mat):
            target_code_mat = np.load(path_target_code_mat)
        else:
            target_code_mat = np.zeros([ad_size, target_size, 48])
            for i in range(ad_size):
                for j in range(target_size):
                    index_i = multi_label == label_target_mat[i, j]
                    k = 0
                    while True:
                        if k==index_i.sum():
                            target_code_mat[i,j] = code[index_i][0]
                            break
                        target_code_tmp = code[index_i][k:k+1]
                        img_num_by_class_tmp = get_query_result_num_by_class(target_code_tmp, code, multi_label, threshold=5)
                        if img_num_by_class_tmp[0, label_target_mat[i, j]]>=100:
                            target_code_mat[i,j] = target_code_tmp
                            break
                        k = k + 1
            np.save(path_target_code_mat, target_code_mat)

        inputs_AD = inputs_AD_sample[:ad_size]
        # generate the target adv of inputs AD
        path_adv_imgs = './save_for_load/ad_attack/adv_imgs_multi.npy'
        if os.path.exists(path_adv_imgs):
            adv_imgs = np.load(path_adv_imgs)
        else:
            adv_imgs = np.zeros([ad_size, target_size, 3, 224, 224])
            var_lambda = 1.0
            for i in range(ad_size):
                img_t = inputs_AD[i].cpu().data
                for j in range(target_size):
                    print(i, j)
                    targetCode = target_code_mat[i, j]
                    adv__ = func_targetedAttack_nag(model1, img_t, targetCode, eps=step_size / 255,
                                                    l_inf_max=step_max, decay_factor=1.0, t_prob=0.5,
                                                    var_lambda=var_lambda, random_noise_level=float(32.0) / 255,
                                                    noise_distribution='uniform')
                    adv_imgs[i, j] = adv__.cpu().data.numpy()
            np.save(path_adv_imgs, adv_imgs)
        if not bVisualize:
            inputs_adv = Variable(torch.Tensor(adv_imgs).cuda())
            img_num_by_class_adv_white = get_img_num_by_class_from_img_batch(inputs_adv, model1, code, multi_label,
                                                                             threshold=5, batch_size=8)
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_white, label_target_mat)
            print("White Box Targeted Result:", img_num_target_targeted.astype(int))
            print("White Box Success:", (img_num_target_targeted > 10).sum())

            img_num_by_class_adv_black = get_img_num_by_class_from_img_batch(inputs_adv, model2, code2, multi_label2,
                                                                             threshold=5, batch_size=8)
            img_num_target_targeted = get_targeted_from_all_class(img_num_by_class_adv_black, label_target_mat)
            print("Black Box Targeted Result:", img_num_target_targeted.astype(int))
            print("Black Box Success:", (img_num_target_targeted > 10).sum())
            from scipy import io
            io.savemat('./save_for_load/ad_attack/img_num_target_targeted_multi_%s.mat'%(net2), {'img_num_target_targeted_multi_%s'%(net2):img_num_target_targeted} )
            print(img_num_by_class_adv_black.sum(axis=2).astype(int))
        else:
            visualize_AD_imgs() #this function is controlled by bVisualize
