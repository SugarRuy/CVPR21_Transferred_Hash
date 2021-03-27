import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from myRetrieval import HashHammingRetrievalSystem
from publicFunctions import get_dsets_loader_by_dataset, NetworkSettings


class WrapSoftmax(nn.Module):
    # Wrap a softmax model to perform like a kNN-voted Softmax model
    # Use the last layer features(softmax output)

    def __init__(self, model, database_code, database_labels):
        super(WrapSoftmax, self).__init__()
        self.model = model
        self.database_code = database_code
        self.database_labels = database_labels
        self.database_size = database_code.shape[0]
        self.num_classes = int(database_labels.max()+1)
        self.K = int(self.database_size / self.num_classes)

    def forward(self, x, K=-1):
        # TODO: threshold classification implementation.
        # TODO: abandon mechanism
        # TODO: pre-calculate the distribution of distance
        if K == -1:
            K = self.K
        out = self.model(x).cpu().detach().numpy()
        sim_score = np.matmul(out, self.database_code.transpose()) * (-0.5) + self.num_classes / 2
        topK_index = np.argpartition(sim_score, K, axis=-1)
        counts_by_class = np.zeros([x.shape[0], self.num_classes])
        for sample_i in range(x.shape[0]):
            counts_by_class[sample_i] = np.array([(self.database_labels[topK_index[sample_i]][:K]==i).sum() for i in range(self.num_classes)])
        probs = counts_by_class.astype(float) / K
        return torch.tensor(probs).cuda()


class WrapClassifier(nn.Module):
    # Wrap a HashNet model to perform like a kNN-voted Softmax model
    # Use the last layer features(softmax output)

    def __init__(self, hashmodel, database_code, database_labels):
        super(WrapClassifier, self).__init__()
        self.hashmodel = hashmodel
        self.database_code = database_code
        self.database_labels = database_labels
        self.database_size = database_code.shape[0]
        self.hash_bit = database_code.shape[1]
        self.num_classes = int(database_labels.max()+1)
        self.K = int(self.database_size / self.num_classes)

    def forward(self, x, K=-1):
        # TODO: threshold classification implementation.
        # TODO: abandon mechanism
        # TODO: pre-calculate the distribution of distance
        if K == -1:
            K = self.K
        hash_code = np.sign(self.hashmodel(x).cpu().detach().numpy())
        hamm_dist = np.matmul(hash_code, self.database_code.transpose()) * (-0.5) + self.hash_bit / 2
        topK_index = np.argpartition(hamm_dist, K, axis=-1)
        counts_by_class = np.zeros([x.shape[0], self.num_classes])
        for sample_i in range(x.shape[0]):
            counts_by_class[sample_i] = np.array([(self.database_labels[topK_index[sample_i]][:K]==i).sum() for i in range(self.num_classes)])
        probs = counts_by_class.astype(float) / K
        return torch.tensor(probs).cuda()

    def predict(self, x, K=-1):
        return self.forward(x, K=K)

class WrapSimpleRetriSysClassifier(nn.Module):
    # Wrap a retrieval system and auxiliary labels to perform like a kNN-voted(simple) Softmax model
    # Use the last layer features(softmax output)
    # The reason of 'simple' is because it use get_probs_by_topK_index(), which is basically counting the number of classes.
    def __init__(self, retri_sys, aux_labels, k_aux):
        # k_aux is little bit different than K in the usage. It sets the size we want to use to count the probs.
        super(WrapSimpleRetriSysClassifier, self).__init__()
        self.retri_sys = retri_sys
        self.aux_labels = aux_labels
        self.num_classes = int(aux_labels.max() + 1) # use the max number of labels as the num_classes
        self.k_aux = k_aux

    def forward(self, x):
        #import time
        #start_time = time.time()
        topK_index = self.retri_sys.retrieval(x)
        #print("--- retrieval %s seconds ---" % (time.time() - start_time))

        probs = get_probs_by_topK_index(topK_index, self.num_classes, self.k_aux, self.aux_labels)
        '''
        counts_by_class = np.zeros([x.shape[0], self.num_classes])
        for sample_i in range(x.shape[0]):
            counts_by_class[sample_i] = np.array([(self.aux_labels[topK_index[sample_i]][:self.k_aux]==i).sum() for i in range(self.num_classes)])
        probs = counts_by_class.astype(float) / self.k_aux
        '''
        return torch.tensor(probs).cuda()

    def predict(self, x):
        return self.forward(x)

def get_probs_by_topK_index(topK_index, num_classes, K, database_labels):
    # Given topK_index, simply counts the rate of recurrence of each class over the over indexes.
    # Then return a probability-like array(or matrix) by dividing the K number.
    input_size = topK_index.shape[0]
    counts_by_class = np.zeros([input_size, num_classes])
    for sample_i in range(input_size):
        counts_by_class[sample_i] = np.array(
            [(database_labels[topK_index[sample_i]][:K] == i).sum() for i in range(num_classes)])
    probs = counts_by_class.astype(float) / K
    return probs


class WrapDeepRankingClassifier(WrapClassifier):
    # Not in used. Not finished. Do not use it.
    def __init__(self, hashmodel, database_feat, database_labels):
        super(WrapDeepRankingClassifier, self).__init__(hashmodel, database_feat, database_labels)
        self.hashmodel = hashmodel
        self.database_feat = database_feat
        self.database_labels = database_labels
        self.database_size = database_feat.shape[0]
        self.hash_bit = database_feat.shape[1]
        self.num_classes = int(database_labels.max()+1)
        self.K = int(self.database_size / self.num_classes)
        raise NotImplementedError

    def forward(self, x, K=-1):
        # TODO: threshold classification implementation.
        # TODO: abandon mechanism
        # TODO: pre-calculate the distribution of distance
        if K == -1:
            K = self.K
        hash_code = self.hashmodel(x).cpu().detach().numpy()
        hamm_dist = np.matmul(hash_code, self.database_code.transpose()) * (-0.5) + self.hash_bit / 2
        topK_index = np.argpartition(hamm_dist, K, axis=-1)
        counts_by_class = np.zeros([x.shape[0], self.num_classes])
        for sample_i in range(x.shape[0]):
            counts_by_class[sample_i] = np.array([(self.database_labels[topK_index[sample_i]][:K]==i).sum() for i in range(self.num_classes)])
        probs = counts_by_class.astype(float) / K
        return torch.tensor(probs).cuda()

class WrapSimpleRetriSysClassifierThreshold(nn.Module):
    # Wrap a retrieval system and auxiliary labels to perform like a kNN-voted(simple) Softmax model
    # Use the last layer features(softmax output)
    # The reason of 'simple' is because it use get_probs_by_topK_index(), which is basically counting the number of classes.
    def __init__(self, retri_sys, aux_labels, threshold):
        # k_aux is little bit different than K in the usage. It sets the size we want to use to count the probs.
        super(WrapSimpleRetriSysClassifierThreshold, self).__init__()
        self.retri_sys = retri_sys
        self.aux_labels = aux_labels
        self.num_classes = int(aux_labels.max() + 1) # use the max number of labels as the num_classes
        self.threshold = threshold

    def forward(self, x):
        #import time
        #start_time = time.time()
        threshold_index = self.retri_sys.retrieval_hamming_ball(x)
        #print("--- retrieval %s seconds ---" % (time.time() - start_time))

        #probs = get_probs_by_topK_index(topK_index, self.num_classes, self.k_aux, self.aux_labels)
        probs = self.get_probs_by_threshold_index(threshold_index, num_classes=self.num_classes, database_labels=self.aux_labels)
        #probs = self.get_probs_by_threshold_index_divided(threshold_index, num_classes=self.num_classes, database_labels=self.aux_labels)
        return torch.tensor(probs).cuda()

    def predict(self, x):
        return self.forward(x)

    def get_probs_by_threshold_index(self, threshold_index, num_classes, database_labels):
        # Given threshold_index, simply counts the rate of recurrence of each class over the over indexes.
        # Then return a probability-like array(or matrix) by dividing the K number.
        input_size = threshold_index.shape[0]
        counts_by_class = np.zeros([input_size, num_classes])
        for sample_i in range(input_size):
            counts_by_class[sample_i] = np.array(
                [(database_labels[threshold_index[sample_i]] == i).sum() for i in range(num_classes)])
        probs = np.zeros([input_size, num_classes])
        for i in range(input_size):
            if counts_by_class[i].sum() == 0:
                probs[i] = 0
            else:
                probs[i] = counts_by_class[i].astype(float) / counts_by_class[i].sum()
            #probs = np.array([counts_by_class[i].astype(float) / counts_by_class[i].sum() for i in range(input_size)])
        return probs

    def get_probs_by_threshold_index_divided(self, threshold_index, num_classes, database_labels, targeted_num=10):
        input_size = threshold_index.shape[0]
        counts_by_class = np.zeros([input_size, num_classes])
        for sample_i in range(input_size):
            counts_by_class[sample_i] = np.array(
                [(database_labels[threshold_index[sample_i]] == i).sum() for i in range(num_classes)])
        probs = np.zeros([input_size, num_classes])
        for i in range(input_size):
            if counts_by_class[i].sum() == 0:
                probs[i] = 0
            else:
                probs[i] = counts_by_class[i].astype(float) / targeted_num
            # probs = np.array([counts_by_class[i].astype(float) / counts_by_class[i].sum() for i in range(input_size)])
        return probs

if __name__ == "__main__":

    import argparse, os
    parser = argparse.ArgumentParser(description='HashNet')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar10resize', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='ResNet50', help="base network type")

    # set default to -1 for accepting the pre-set snapshot_iter
    parser.add_argument('--snapshot_iter', type=int, default=15000, help="number of iterations the model has")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size to load data")

    # classifier parameters
    parser.add_argument('--threshold', type=int, default=5, help="Retrieval Threshold(conflit with k_aux)")
    parser.add_argument('--K', type=int, default=5000, help="Retrieval topK(conflit with threshold)")
    parser.add_argument('--k_aux', type=int, default=-1, help="Majority vote topK(conflit with threshold)")

    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    job_dataset = args.dataset
    hash_bit = args.hash_bit
    net = args.net

    snapshot_iter = args.snapshot_iter
    K = args.K
    threshold = args.threshold
    batch_size = args.batch_size

    network_settings = NetworkSettings(job_dataset, hash_bit, net, snapshot_iter)
    model = network_settings.get_model()
    output, code, multi_label = network_settings.get_out_code_label('database')
    dset_loaders = network_settings.get_dset_loaders()

    hash_retri_sys = WrapHashRetrievalSystem(model, code, database_loader=dset_loaders['database'], K=K, batch_size=batch_size)
    aux_labels = multi_label
    wrapModel = WrapSimpleRetriSysClassifier(hash_retri_sys, aux_labels=aux_labels, k_aux=K)

    iter_test = iter(dset_loaders['test'])
    iter_database = iter(dset_loaders['database'])

    test_size = 10000
    predicts = np.zeros([test_size])
    labels = np.zeros([test_size])
    test_len = len(dset_loaders['test'])
    for i in range(test_len):
        print("batch: ", i)
        data = iter_test.next()
        inputs_test, labels_test = Variable(data[0]).cuda(), data[1]
        #outputs_test = wrapModel(inputs_test).cpu().item().numpy()
        outputs_test = wrapModel(inputs_test)
        predict_labels = torch.argmax(outputs_test, axis=-1)
        if i == test_len - 1:
            predicts[i*batch_size:] = predict_labels.cpu().numpy()
            labels[i*batch_size:] = labels_test.numpy()
        else:
            predicts[i*batch_size:(i+1)*batch_size] = predict_labels.cpu().numpy()
            labels[i*batch_size:(i+1)*batch_size] = labels_test.numpy()


    acc = float((predicts==labels).sum()) / test_size
    print(acc)

