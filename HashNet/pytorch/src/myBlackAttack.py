import math

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from myExpForPapers_nag import EXPSettings
from myRetrieval import HashHammingRetrievalSystem
from myWrapClassifier import WrapSimpleRetriSysClassifierThreshold
from publicFunctions import NetworkSettings
import argparse, os

from publicVariables import iters_list


class BaseAttackFlow:
    def __init__(self, retri_sys, aux_model, attack_method='square', count_method='count'):
        self.retri_sys = retri_sys
        self.aux_model = aux_model
        self.attack_method = attack_method
        self.count_method = count_method


class SimpleAttackFlow():
    # Simply simulate the attack flow:(Works on local computer)
    # Given a retri_sys that returns top k_aux index of the input, we use a auxiliary local classification model to get the label of its return.
    # To simplify the process in local computer, we use our local model to pre-calculate all the possible labels corresponding to the returned index.

    def __init__(self, wrapModel, attack_method='square'):
        self.attack_method = attack_method
        # wrap model is basically a wrapped classification model based on the retrieval system
        self.wrapModel = wrapModel

    def attack(self, inputs_test):
        if self.attack_method == 'square':
            self.square_attack_preset_args(inputs_test)
        return

    def square_attack_preset_args(self, inputs_test, labels_target, batch_size=32, eps=0.1, niters=10000, p_init=0.1, targeted=False, save_path='./exps/'):
        import attacks
        num_classes = self.wrapModel.num_classes

        attack = 'LinfSquareAttack'
        square_attack = attacks.square_attack_l2 if attack == 'L2SquareAttack' else attacks.square_attack_linf
        wrapModel = self.wrapModel
        model = attacks.ModelPT(model=wrapModel, batch_size=batch_size, gpu_memory=0.99)

        def dense_to_onehot(y_test, n_cls):
            y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
            y_test_onehot[np.arange(len(y_test)), y_test] = True
            return y_test_onehot

        labels_test = wrapModel(inputs_test).argmax(1)
        x_test = inputs_test.cpu().numpy()
        y_test = labels_test.cpu().numpy()

        logits_clean = model.predict(x_test)
        corr_classified = logits_clean.argmax(1) == y_test
        print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)))

        y_target = labels_target if targeted else y_test
        y_target_onehot = dense_to_onehot(y_target, n_cls=num_classes)

        metrics_path = save_path + '/just_for_save.metrics'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        loss_type = 'margin_loss' if not targeted else 'cross_entropy'
        n_queries, x_adv = square_attack(model, x_test, y_target_onehot, corr_classified, eps=eps, n_iters=niters,
                                         p_init=p_init, metrics_path=metrics_path, targeted=targeted,
                                         loss_type=loss_type)
        return n_queries, x_adv


def get_labels(model, dset_loader):
    # Given dataset loaders,
    # Outputs labels
    from tqdm import tqdm
    model = model.eval()
    iter_database = iter(dset_loader)
    batch_size = dset_loader.batch_size
    data_size = len(dset_loader.dataset)
    predicts = np.zeros([data_size])
    with torch.no_grad():
        for i in tqdm(range(math.ceil(data_size / batch_size))):
            data = iter_database.next()
            inputs_test = Variable(data[0]).cuda()
            outputs_test = model(inputs_test)
            predict_labels = torch.argmax(outputs_test, axis=-1)
            predicts[i * batch_size:(i + 1) * batch_size] = predict_labels.cpu().numpy()
    return predicts

'''
def get_aux_model(aux_net, dataset, aux_epochs=49):
    aux_model = get_softmax_basenet(aux_net)
    model_dict_path = '../snapshot/%s_%s_softmax/epoch_%d' % (dataset, aux_net, aux_epochs)
    aux_model.load_state_dict(torch.load(model_dict_path))
    aux_model = aux_model.eval().cuda()
    return aux_model


def get_aux_labels(aux_net, aux_dataset, aux_epochs, network_settings):
    black_dataset = network_settings.job_dataset

    aux_model = get_aux_model(aux_net, aux_dataset, aux_epochs=aux_epochs)
    path_save_labels_folder = './save_for_load/%s/softmax/aux_model/%s/%s/%s/' % (
    black_dataset, aux_net, aux_dataset, aux_epochs)
    if not os.path.exists(path_save_labels_folder):
        os.makedirs(path_save_labels_folder)
    path_save_labels = path_save_labels_folder + 'database_labels.npy'
    if os.path.exists(path_save_labels):
        return np.load(path_save_labels)
    else:
        dset_loaders = network_settings.get_dset_loaders()
        aux_labels = get_labels(aux_model, dset_loaders['database'])
        np.save(path_save_labels, aux_labels)
        return aux_labels
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='imagenet', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='ResNet101', help="base network type")

    # set default to -1 for accepting the pre-set snapshot_iter
    parser.add_argument('--snapshot_iter', type=int, default=-1, help="number of iterations the model has")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size to load data")

    # model, data and attack_method
    parser.add_argument('--attack', type=str, default='L2SquareAttack', help="attack algorithm type")
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--pert_level', type=float, default=0.125, help="Linf Level")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    job_dataset = args.dataset
    hash_bit = args.hash_bit
    net = args.net

    snapshot_iter = iters_list[net]

    batch_size = args.batch_size


    attack = args.attack

    network_settings = NetworkSettings(job_dataset, hash_bit, net, snapshot_iter, batch_size=128)
    model = network_settings.get_model()
    output, code, multi_label = network_settings.get_out_code_label('database')
    dset_loaders = network_settings.get_dset_loaders()

    _, code_test, _ = network_settings.get_out_code_label('test')

    # ---------------------------------------
    # exp:
    dis_method = 'cW'
    i_max, j_max = 64, 32
    step, linf = 1.0, 32
    exp_settings = EXPSettings(net, net, dis_method, i_max, j_max, step=step, linf=linf)
    #i_index_set, j_index_matrix = exp_settings.cal_index_set_matrix(multi_label_test, code_test2, code2, multi_label2, code_test, code, multi_label)
    i_index_set, j_index_matrix =  exp_settings.cal_index_set_matrix_white(code_test, code, multi_label)
    inputs_ori_tensor = exp_settings.cal_inputs_ori_tensor(dset_test=dset_loaders['test'].dataset)
    test_true_label_y = exp_settings.test_true_label_y
    label_targeted = np.zeros([i_max, j_max]).astype(int)
    for i in range(i_max):
        j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
        label_targeted_i = np.array([multi_label[j_index_set[j]] for j in range(j_max)])
        label_targeted[i] = label_targeted_i
    # ----------------------------------------

    aux_labels = multi_label
    retrieval_threshold = 5

    hash_retri_sys = HashHammingRetrievalSystem(model, code, dset_loaders['database'], retrieval_threshold, batch_size=16)

    wrapModel = WrapSimpleRetriSysClassifierThreshold(hash_retri_sys, aux_labels=aux_labels, threshold=retrieval_threshold)
    attackFlow = SimpleAttackFlow(wrapModel, attack_method='square')

    '''
    iter_test = iter(dset_loaders['test'])
    test_data = iter_test.next()
    inputs_test, labels_test = Variable(test_data[0]).cuda(), test_data[1]
    targeted = True
    num_classes = labels_test.shape[1]
    labels_test = np.argmax(labels_test, 1)
    if targeted:
        labels_target = (labels_test + 1) % num_classes
    else:
        labels_target = labels_test
    #n_queries, x_adv = attackFlow.square_attack_preset_args(inputs_test)
    n_queries, x_adv = attackFlow.square_attack_preset_args(inputs_test, labels_target=labels_target, niters=200, eps=0.1255, p_init=0.03, targeted=targeted)
    '''
    '''
    #for i in range(i_max):
    inputs_test = inputs_ori_tensor.cuda()
    targeted = False
    if targeted:
        labels_target = label_targeted[:, 0].astype(int)
    else:
        wrapModel_black = WrapSimpleRetriSysClassifierThreshold(hash_retri_sys, aux_labels=aux_labels, threshold=retrieval_threshold)
        ori_labels = wrapModel_black(inputs_test).argmax(-1)
        labels_target = ori_labels.cpu().numpy()
    '''
    targeted = args.targeted
    tmp_exp_path = './tmp/queried/'
    if not os.path.exists(tmp_exp_path):
        os.makedirs(tmp_exp_path)
    exp_size = 1024
    #pert_level = 0.062 #0.062 0.031 0.125
    pert_level = args.pert_level
    np.random.seed(0)
    index_random = np.random.randint(0, len(dset_loaders['database'].dataset), exp_size)
    path_ori_img = tmp_exp_path+'ori_img.npy'
    if os.path.exists(path_ori_img):
        ori_img = np.load(path_ori_img)
    else:
        from tqdm import tqdm
        ori_img = np.zeros([exp_size, 3, 224, 224])
        for i in tqdm(range(exp_size)):
            ori_img[i] = dset_loaders['database'].dataset[index_random[i]][0].numpy()
        np.save(path_ori_img, ori_img)
    inputs_test = torch.tensor(ori_img).cuda().float()
    from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class

    wrapModel_black = WrapSimpleRetriSysClassifierThreshold(hash_retri_sys, aux_labels=aux_labels, threshold=retrieval_threshold)
    ori_labels = wrapModel_black(inputs_test).argmax(-1)
    labels_target = ori_labels.cpu().numpy() if not targeted else (ori_labels.cpu().numpy()+1) % 100
    black_img_num_result = get_img_num_by_class_from_img_batch(inputs_test, model, code, multi_label, threshold=5, batch_size=32)
    adv_black_retrieval_result = get_targeted_from_all_class(black_img_num_result, labels_target).astype(int)
    index_valid = adv_black_retrieval_result>=10 if not targeted else adv_black_retrieval_result<10

    #n_queries, x_adv = attackFlow.square_attack_preset_args(inputs_test[index_valid], labels_target=labels_target[index_valid], niters=200, eps=0.031, p_init=0.031, targeted=targeted)
    path_x_adv = tmp_exp_path + 'x_adv_%s_%f_targeted_%s.npy'%(net, pert_level, str(targeted))
    if os.path.exists(path_x_adv):
        x_adv = np.load(path_x_adv)
    else:
        n_queries, x_adv = attackFlow.square_attack_preset_args(inputs_test[index_valid],
                                                            labels_target=labels_target[index_valid], niters=200,
                                                            eps=pert_level, p_init=pert_level, targeted=targeted, save_path=tmp_exp_path+'%s_%f_targeted_%s'%(net, pert_level, str(targeted)))

    inputs_adv = torch.as_tensor(x_adv.astype('float32')).cuda()
    black_img_num_result_adv = get_img_num_by_class_from_img_batch(inputs_adv, model, code, multi_label, threshold=5, batch_size=32)
    adv_black_retrieval_result_adv = get_targeted_from_all_class(black_img_num_result_adv, labels_target).astype(int)

    if not targeted:
        print('Attack Success Rate:%f=%d of %d'%((adv_black_retrieval_result_adv<10).sum()/(adv_black_retrieval_result_adv<10).shape[0],
                                             (adv_black_retrieval_result_adv<10).sum(), (adv_black_retrieval_result_adv<10).shape[0]))
    else:
        print('Attack Success Rate:%f=%d of %d' % (
        (adv_black_retrieval_result_adv >= 10).sum() / (adv_black_retrieval_result_adv >= 10).shape[0],
        (adv_black_retrieval_result_adv >= 10).sum(), (adv_black_retrieval_result_adv >= 10).shape[0]))
    '''

    # To see the adv attack success rate.
    # Temporary
    black_labels = wrapModel(inputs_test).argmax(-1)
    wrapModel_black = WrapSimpleRetriSysClassifierThreshold(hash_retri_sys, aux_labels=aux_labels, threshold=retrieval_threshold)
    ori_labels = wrapModel_black(inputs_test).argmax(-1)
    inputs_adv = torch.as_tensor(x_adv.astype('float32')).cuda()
    adv_labels = wrapModel_black(inputs_adv).argmax(-1)
    adv_labels == ori_labels
    adv_labels.cpu().numpy() == labels_target

    # evaluate the original success rate
    from myRetrieval import get_img_num_by_class_from_img_batch, get_targeted_from_all_class
    black_img_num_result = get_img_num_by_class_from_img_batch(inputs_test, model, code, multi_label, threshold=5, batch_size=32)
    adv_black_retrieval_result = get_targeted_from_all_class(black_img_num_result, labels_target).astype(int)

    # evaluate the black success rate
    black_img_num_result_adv = get_img_num_by_class_from_img_batch(inputs_adv, model, code, multi_label, threshold=5, batch_size=32)
    adv_black_retrieval_result_adv = get_targeted_from_all_class(black_img_num_result_adv, labels_target).astype(int)
    '''
    '''
    from torchvision.utils import save_image
    for i in range(i_max):
        save_image(inputs_ori_tensor[i], 'tmp/jpgs/ori_%s.jpg'%(i))
        save_image(torch.tensor(x_adv[i]), 'tmp/jpgs/adv_%s.jpg' % (i))
    '''
