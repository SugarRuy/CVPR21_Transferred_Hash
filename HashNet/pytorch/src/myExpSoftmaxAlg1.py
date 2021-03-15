import numpy as np
import os
from myTrainSoftmax import get_softmax_basenet, load_softmax_model
from myAttackSoftmax import  cal_success_rate

from mySoftmaxAlg1 import irrn_softmax

if __name__ == "__main__":

    step_size = 1.0
    step_max = 32
    imgs_size = 1024
    epoch = 9

    var_lambda = 100.1
    random_nosie_level = 32.0

    path_softmax_labels = './save_for_load/softmax/ori_labels_%d.npy'%(imgs_size)
    softmax_labels = np.load(path_softmax_labels)
    ori_labels, target_labels = softmax_labels[0], softmax_labels[1]

    path_softmax_ori_imgs = './save_for_load/softmax/ori_imgs_%d.npy'%(imgs_size)
    ori_imgs = np.load(path_softmax_ori_imgs)

    #net_values = ['ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'Inc_v3', 'DenseNet161']
    #net_values = ['ResNext101_32x4d', 'SEResNet50', 'Inc_v3', 'DenseNet161']
    net_values = ['ResNet34']
    adv_method_value = ['miFGSMDI', 'NAG']

    perturbation_level_arr = np.array([1.0])
    noise_level_arr = np.array([4, 8, 16, 32, 64, 128])

    big_irrn_mat = np.zeros([len(net_values), len(adv_method_value), len(perturbation_level_arr), len(noise_level_arr)])
    for i in range(len(net_values)):
        net = net_values[i]
        model = load_softmax_model(net, epoch=epoch).eval()
        for j in range(len(adv_method_value)):
            adv_method = adv_method_value[j]
            if adv_method == 'NAG':
                NAG_save_folder = './save_for_load/softmax/NAG_noise%s_lambda%s/' % (str(random_nosie_level), str(var_lambda))
                if not os.path.exists(NAG_save_folder):
                    os.makedirs(NAG_save_folder)
                path_softmax_adv_imgs = NAG_save_folder + '/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy' % (net, adv_method, step_size, step_max, imgs_size)
            else:
                path_softmax_adv_imgs = './save_for_load/softmax/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy' % (net, adv_method, step_size, step_max, imgs_size)
            adv_imgs = np.load(path_softmax_adv_imgs)
            irrn_mat = irrn_softmax(model, perturbation_level_arr, noise_level_arr, adv_imgs, ori_imgs, target_labels)
            big_irrn_mat[i, j] = irrn_mat
            print "ALG1: net:%s, adv_method:%s" % (net, adv_method)
            print irrn_mat
