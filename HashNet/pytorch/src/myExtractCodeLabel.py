# -*- coding: utf-8 -*-


import os
import numpy as np
import torch

from torch.autograd import Variable

from publicFunctions import NetworkSettings

import torchvision.transforms as transforms

def trans_train_resize_mnist(resize = 224):
    return  transforms.Compose([
        transforms.Resize(resize),
        transforms.Grayscale(num_output_channels = 3),
        transforms.ToTensor()
        ])

def trans_train_resize_cifar10(resize = 224):
    return  transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
        ])

def trans_train_resize_imagenet(resize = 224):
    return  transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor()
        ])


def get_output_code_label_list_by_dset(dsets, dset_loaders, model, hash_bit, mode='test'):
    """
    Args:
        dsets: dict containing datasets of test and database;
        dset_loaders: dict containing dataset loaders of test and database;
        model: HashNet model;
        mode: 'test' or 'database'.

    Returns:
        Tuple(all_output, all_label)
        all_output: original output of the dataset of the model - (numpy.ndarray)
        all_label: one-hot labels of the dataset - (numpy.ndarray)
    """
    len_dset = len(dsets[mode])
    len_loader = len(dset_loaders[mode])

    print('Total number of %s batches(images): %d' % (mode, len_loader))

    all_output = np.zeros([len_dset, hash_bit])
    all_label = np.zeros([len_dset])
     #STILL CONSTRUCTING!!!
    iter_loader = iter(dset_loaders[mode])
    batch_size = dset_loaders[mode].batch_size
    for i in range(len_loader):
        print("batch:",i)
        data = iter_loader.next()
        inputs = data[0]
        labels = data[1]
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)
        offset_size = labels.shape[0] # how many images are loaded in current batch
        all_output[i*batch_size:i*batch_size+offset_size] = outputs.cpu().data.float()
        if len(labels.shape) == 2:
            all_label[i*batch_size:i*batch_size+offset_size] = torch.argmax(labels, axis=1)
        elif len(labels.shape) == 1:
            all_label[i*batch_size:i*batch_size+offset_size] = labels.numpy()

    all_output = torch.Tensor(all_output)
    all_label = torch.Tensor(all_label)

    return all_output, all_label

def extract_all_output_labels(model, dset_loader):
    # Given dataset loaders,
    # Outputs labels
    import math
    from tqdm import tqdm
    model = model.eval()
    iter_database = iter(dset_loader)
    batch_size = dset_loader.batch_size
    data_size = len(dset_loader.dataset)
    output_list = []
    label_list = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(data_size/batch_size))):
            data = iter_database.next()
            inputs_test = Variable(data[0]).cuda()
            outputs_test = model(inputs_test)
            output_list.append(outputs_test.cpu().numpy())
            label_list.append(data[1].cpu().numpy())
    all_output = np.vstack(output_list)
    if len(data[1].shape)==2:
        # when one-hot labels used, get the argmax
        label_list = np.vstack(label_list)
        label_list = label_list.argmax(axis=-1)
    elif len(data[1].shape)==1:
        # when single labels used, concatenate the label list
        label_list = np.concatenate(label_list)
    return all_output, label_list

def extract_all_code_output_labels(model, dset_loader):
    all_output, label_list_argmax = extract_all_output_labels(model, dset_loader)
    all_code = np.sign(all_output)
    return all_code, all_output, label_list_argmax

if __name__ == "__main__":
    # Step 2: Extract Code and One-hot Label
    # Extract the Hashcode of different job_datasets by different hashnet models
    # Then save them into ./save_for_load/blackbox/
    import argparse
    parser = argparse.ArgumentParser(description='HashNet')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    # for hashnet
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--snapshot_iter', type=int, default=-1, help="number of iterations the model has")

    # for all
    parser.add_argument('--dataset', type=str, default='imagenet', help="dataset name")
    parser.add_argument('--net', type=str, default='SEResNet50', help="base network type")


    # set default to -1 for accepting the pre-set snapshot_iter
    parser.add_argument('--batch_size', type=int, default=64, help="batch size to load data")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    job_dataset = args.dataset
    hash_bit = args.hash_bit
    net = args.net

    batch_size = args.batch_size

    from publicVariables import iters_list
    snapshot_iter = iters_list[net] if args.snapshot_iter == -1 else args.snapshot_iter

    print(args)

    network_settings = NetworkSettings(job_dataset, hash_bit, net, snapshot_iter, batch_size=batch_size)
    model = network_settings.get_model()
    dset_loaders = network_settings.get_dset_loaders()
    save_path = network_settings.save_path
    model_dict_path, test_save_path, database_save_path = \
        network_settings.model_dict_path, network_settings.test_save_path, network_settings.database_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    MODE = 'test'
    output_code, output, label = extract_all_code_output_labels(model, dset_loaders[MODE])
    np.savez(test_save_path, output, output_code, label)

    MODE = 'database'
    output_code, output, label = extract_all_code_output_labels(model, dset_loaders[MODE])
    np.savez(database_save_path, output, output_code, label)
