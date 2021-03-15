import numpy as np
import torch
import torch.nn as nn
import os

from torch.autograd import Variable
import torchvision.transforms as transforms
from myTrainSoftmax import get_softmax_basenet, load_softmax_model


def cal_success_rate(model, imgs, target_labels, batch_size=32):
    black_labels = np.zeros_like(target_labels)
    i = 0
    #print batch_size, imgs.shape[0]
    while batch_size*(i+1) < imgs.shape[0]:
        #print "Batch:%d"%(i)
        output = model(Variable(torch.Tensor(imgs[batch_size*i:batch_size*(i+1)]).cuda())).cpu().data.numpy()
        black_labels[batch_size*i:batch_size*(i+1)] = np.argmax(output, axis=1)
        i += 1
    output = model(Variable(torch.Tensor(imgs[batch_size*i:]).cuda())).cpu().data.numpy()
    black_labels[batch_size*i:] = np.argmax(output, axis=1)
    #print "Batch:%d, Finished" % (i)

    # compare target_labels and black_labels
    match_index = black_labels == target_labels
    success_rate = float(match_index.sum()) / black_labels.shape[0]
    return success_rate, black_labels


if __name__ == "__main__":
    # job_dataset = 'mnist'
    import argparse
    parser = argparse.ArgumentParser(description='Softmax')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net1', type=str, default='ResNet34', help="source network")
    parser.add_argument('--net2', type=str, default='ResNet152', help="target network")

    parser.add_argument('--imgs_size', type=int, default=1024, help="imgs_size")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--epoch1', type=int, default=9, help="Epoch for net1")
    parser.add_argument('--epoch2', type=int, default=9, help="Epoch for net2")
    parser.add_argument('--adv_method', type=str, default='miFGSMDI', help="adv method")
    parser.add_argument('--step_max', type=int, default=32, help="maximum step")
    parser.add_argument('--step_size', type=float, default=1.0, help="step size")

    # net candidates:
    net_values = ['ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'Inc_v3', 'DenseNet161']

    var_lambda = 100.1
    random_nosie_level = 32.0

    args = parser.parse_args()
    # NAG is not finished
    net1 = args.net1
    net2 = args.net2

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    imgs_size = args.imgs_size
    batch_size = args.batch_size
    adv_method = args.adv_method
    epoch1 = args.epoch1
    epoch2 = args.epoch2
    step_size = args.step_size
    step_max = args.step_max

    job_dataset = 'imagenet'

    model1 = load_softmax_model(net1, epoch=epoch1).eval()
    model2 = load_softmax_model(net2, epoch=epoch2).eval()

    path_softmax_labels = './save_for_load/softmax/ori_labels_%d.npy'%(imgs_size)
    softmax_labels = np.load(path_softmax_labels)
    ori_labels, target_labels = softmax_labels[0], softmax_labels[1]

    if adv_method == 'NAG':
        NAG_save_folder = './save_for_load/softmax/NAG_noise%s_lambda%s/'%(str(random_nosie_level), str(var_lambda))
        if not os.path.exists(NAG_save_folder):
            os.makedirs(NAG_save_folder)
        path_softmax_adv_imgs = NAG_save_folder+'/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy' % (net1, adv_method, step_size, step_max, imgs_size)
    else:
        path_softmax_adv_imgs = './save_for_load/softmax/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy'%(net1, adv_method, step_size, step_max, imgs_size)

    adv_imgs = np.load(path_softmax_adv_imgs)

    #print "label matched ori:", (ori_labels==target_labels).sum()


    path_softmax_ori_imgs = './save_for_load/softmax/ori_imgs_%d.npy'%(imgs_size)
    ori_imgs = np.load(path_softmax_ori_imgs)
    '''
    accuracy, ori_predict_labels = cal_success_rate(model1, ori_imgs, ori_labels, batch_size=32)
    print "accuracy(ori):", accuracy
    '''
    success_rate, labels_predict_black = cal_success_rate(model2, adv_imgs, target_labels, batch_size=32)
    print "attack from %s to %s: %s"%(net1, net2, adv_method)
    print "success_rate(adv_black):%f=(%d/%d)"%(success_rate, (labels_predict_black==target_labels).sum(), target_labels.shape[0])

    success_rate_ori_black, labels_predict_ori_black = cal_success_rate(model2, ori_imgs, target_labels, batch_size=32)
    print "attack from %s to %s: %s" % (net1, net2, adv_method)
    print "success_rate(ori_black):%f=(%d/%d)"%(success_rate_ori_black, (labels_predict_ori_black==target_labels).sum(), target_labels.shape[0])

    '''
    success_rate_white, labels_predict_white = cal_success_rate(model1, adv_imgs, target_labels, batch_size=32)
    print "attack from %s to %s: %s" % (net1, net1, adv_method)
    print "success_rate(adv_white):%f=(%d/%d)"%(success_rate_white, (labels_predict_white==target_labels).sum(), target_labels.shape[0])
    '''

    # validate data
    '''
    from myGetAdv import get_dsets
    dset = get_dsets(job_dataset)
    path_softmax_random_index = './save_for_load/softmax/ori_random_index.npy'
    random_index = np.load(path_softmax_random_index)
    index_label = np.argmax([dset['test'][random_index[i]][1] for i in range(random_index.shape[0])], axis=1)
    '''