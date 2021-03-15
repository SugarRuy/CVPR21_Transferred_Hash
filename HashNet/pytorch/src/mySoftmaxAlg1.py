import numpy as np
import torch
import torch.nn as nn
import os

from torch.autograd import Variable
import torchvision.transforms as transforms
from myTrainSoftmax import get_softmax_basenet, load_softmax_model
from myAttackSoftmax import  cal_success_rate

def irrn_softmax(model, perturbation_level_arr, noise_level_arr, adv_imgs, ori_imgs, target_labels, noise_distribution='uniform'):
    irrn_mat = np.zeros([len(perturbation_level_arr), len(noise_level_arr)])
    for i in range(len(perturbation_level_arr)):
        perturbation_ratio = perturbation_level_arr[i]
        for j in range(len(noise_level_arr)):
            noise_level = noise_level_arr[j]
            np.random.seed(0)
            if noise_distribution == 'uniform':
                random_noise = np.random.randint(-noise_level, noise_level + 1, adv_imgs.shape).astype(float)
            elif noise_distribution == 'Gaussian':
                # using 3-pi to define the max value of the noise
                random_noise = np.random.normal(0, noise_level / 3, size=adv_imgs.shape).astype(float)
                random_noise = np.clip(random_noise, -noise_level, noise_level)
            adv_imgs_noised = (adv_imgs - ori_imgs) * perturbation_ratio + ori_imgs + random_noise / 255

            success_rate_white, labels_predict_white = cal_success_rate(model, adv_imgs_noised, target_labels, batch_size=16)
            #print "alg1 %s, %s, noise_level:%d, success_rate:%f" % (net, adv_method, noise_level, success_rate_white)
            irrn_mat[i, j] = success_rate_white
    return irrn_mat

if __name__ == "__main__":
    # job_dataset = 'mnist'
    import argparse
    parser = argparse.ArgumentParser(description='Softmax Alg1')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="network")

    parser.add_argument('--imgs_size', type=int, default=1024, help="imgs_size")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--epoch', type=int, default=9, help="Using which epoch checkpoint model")
    parser.add_argument('--adv_method', type=str, default='NAG', help="adv method")
    parser.add_argument('--step_max', type=int, default=32, help="maximum step")
    parser.add_argument('--step_size', type=float, default=1.0, help="step size")

    parser.add_argument('--var_lambda', type=float, default=10.0, help="lbd to balance loss1 and loss2")
    parser.add_argument('--noise', type=str, default='uniform', help="noise distribution")
    parser.add_argument('--noise_level', type=float, default=32.0, help="random_noise_level")

    args = parser.parse_args()
    # NAG is not finished
    net = args.net

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    imgs_size = args.imgs_size
    batch_size = args.batch_size
    adv_method = args.adv_method
    epoch = args.epoch
    step_size = args.step_size
    step_max = args.step_max

    #NAG configuration
    var_lambda = args.var_lambda
    noise_distribution = args.noise
    random_nosie_level = args.noise_level

    # exp ori_imgs size
    job_dataset = 'imagenet'

    model = load_softmax_model(net, epoch=epoch).eval()

    path_softmax_labels = './save_for_load/softmax/ori_labels_%d.npy'%(imgs_size)
    softmax_labels = np.load(path_softmax_labels)
    ori_labels, target_labels = softmax_labels[0], softmax_labels[1]

    path_softmax_ori_imgs = './save_for_load/softmax/ori_imgs_%d.npy'%(imgs_size)
    ori_imgs = np.load(path_softmax_ori_imgs)

    if adv_method == 'NAG':
        NAG_save_folder = './save_for_load/softmax/NAG_noise%s_lambda%s/'%(str(random_nosie_level), str(var_lambda))
        if not os.path.exists(NAG_save_folder):
            os.makedirs(NAG_save_folder)
        path_softmax_adv_imgs = NAG_save_folder+'/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy' % (net, adv_method, step_size, step_max, imgs_size)
    else:
        path_softmax_adv_imgs = './save_for_load/softmax/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy'%(net, adv_method, step_size, step_max, imgs_size)

    adv_imgs = np.load(path_softmax_adv_imgs)

    # net candidates:
    net_values = ['ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'Inc_v3', 'DenseNet161']

    perturbation_level_arr = np.array([1.0])
    noise_level_arr = np.array([4, 8, 16, 32, 64])



    irrn_mat = irrn_softmax(model, perturbation_level_arr, noise_level_arr, adv_imgs, ori_imgs, target_labels)

    print("ALG1: net:%s, adv_method:%s"%(net, adv_method))
    print(irrn_mat)

