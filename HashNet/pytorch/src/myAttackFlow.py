import math

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from myRetrieval import WrapHashRetrievalSystem
from mySoftmaxTrain import get_softmax_basenet
from myWrapClassifier import WrapSimpleRetriSysClassifier
from publicFunctions import NetworkSettings
import argparse, os

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

    def square_attack_preset_args(self, inputs_test, batch_size=32, eps=0.1, niters=10000, p_init=0.1, targeted=False):
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

        y_target = y_test
        y_target_onehot = dense_to_onehot(y_target, n_cls=num_classes)

        metrics_path = 'exps/just_for_save.metrics'
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
        for i in tqdm(range(math.ceil(data_size/batch_size))):
            data = iter_database.next()
            inputs_test = Variable(data[0]).cuda()
            outputs_test = model(inputs_test)
            predict_labels = torch.argmax(outputs_test, axis=-1)
            predicts[i * batch_size:(i + 1) * batch_size] = predict_labels.cpu().numpy()
    return predicts

def get_aux_model(aux_net, dataset, aux_epochs=49):

    aux_model = get_softmax_basenet(aux_net)
    model_dict_path = '../snapshot/%s_%s_softmax/epoch_%d'%(dataset, aux_net, aux_epochs)
    aux_model.load_state_dict(torch.load(model_dict_path))
    aux_model = aux_model.eval().cuda()
    return aux_model

def get_aux_labels(aux_net, aux_dataset, aux_epochs, network_settings):
    black_dataset = network_settings.job_dataset

    aux_model = get_aux_model(aux_net, aux_dataset, aux_epochs=aux_epochs)
    path_save_labels_folder = './save_for_load/%s/softmax/aux_model/%s/%s/%s/'%(black_dataset, aux_net, aux_dataset, aux_epochs)
    if not os.path.exists(path_save_labels_folder):
        os.makedirs(path_save_labels_folder)
    path_save_labels = path_save_labels_folder+'database_labels.npy'
    if os.path.exists(path_save_labels):
        return np.load(path_save_labels)
    else:
        dset_loaders = network_settings.get_dset_loaders()
        aux_labels = get_labels(aux_model, dset_loaders['database'])
        np.save(path_save_labels, aux_labels)
        return aux_labels



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='HashNet')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar10resize', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='ResNet50', help="base network type")

    # set default to -1 for accepting the pre-set snapshot_iter
    parser.add_argument('--snapshot_iter', type=int, default=15000, help="number of iterations the model has")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size to load data")

    # classifier parameters
    parser.add_argument('--K', type=int, default=5000, help="Retrieval topK")
    parser.add_argument('--k_aux', type=int, default=5000, help="wrapModel topK for calculate softmax, in [1,K]")

    # model, data and attack_method
    parser.add_argument('--is_hash', action='store_true') # default using softmax
    parser.add_argument('--is_test', action='store_true') # default using database set(for higher acc)
    parser.add_argument('--attack', type=str, default='L2SquareAttack', help="attack algorithm type")
    parser.add_argument('--is_wrap', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    job_dataset = args.dataset
    hash_bit = args.hash_bit
    net = args.net

    snapshot_iter = args.snapshot_iter
    K = args.K
    k_aux = args.k_aux
    batch_size = args.batch_size

    is_hash = args.is_hash
    is_test = args.is_test
    attack = args.attack

    network_settings = NetworkSettings(job_dataset, hash_bit, net, snapshot_iter, batch_size=16)
    model = network_settings.get_model()
    output, code, multi_label = network_settings.get_out_code_label('database')
    dset_loaders = network_settings.get_dset_loaders()

    #aux_model = get_aux_model(aux_net='ResNet50', dataset=job_dataset)
    #aux_labels = get_labels(aux_model, dset_loaders['database'])
    aux_net = 'ResNet50'
    aux_dataset = job_dataset
    aux_epochs = 49
    aux_labels = get_aux_labels(aux_net, aux_dataset, aux_epochs, network_settings)

    hash_retri_sys = WrapHashRetrievalSystem(model, code, database_loader=dset_loaders['database'], K=K, batch_size=batch_size)

    wrapModel = WrapSimpleRetriSysClassifier(hash_retri_sys, aux_labels=aux_labels, k_aux=k_aux)
    attackFlow = SimpleAttackFlow(wrapModel, attack_method='square')

    iter_test = iter(dset_loaders['test'])
    test_data = iter_test.next()
    inputs_test, labels_test = Variable(test_data[0]).cuda(), test_data[1]
    n_queries, x_adv = attackFlow.square_attack_preset_args(inputs_test)
    
    
    # To see the adv attack success rate. 
    # Temporary
    black_labels = wrapModel(inputs_test).argmax(-1)
    wrapModel_black = WrapSimpleRetriSysClassifier(hash_retri_sys, aux_labels=multi_label, k_aux=k_aux)
    ori_labels = wrapModel_black(inputs_test).argmax(-1)
    inputs_adv = torch.as_tensor(x_adv.astype('float32')).cuda()
    adv_labels = wrapModel_black(inputs_adv).argmax(-1)
    adv_labels == ori_labels
