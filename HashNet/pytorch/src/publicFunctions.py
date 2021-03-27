# -*- coding: utf-8 -*-
from torch.utils import data as util_data
from torchvision import transforms as transforms, datasets as dset

import network
from data_list import ImageList
import numpy as np
import torch
from torch.autograd import Variable
from data_list import default_loader
import os

def get_dsets(job_dataset):
    # get the dsets for extract code
    # or can be used to softmax training with the dsets['train']

    import torchvision.datasets as dset
    dsets = {}
    trans_test = get_trans(job_dataset=job_dataset, mode='test')
    trans_train = get_trans(job_dataset=job_dataset, mode='train')
    if job_dataset == 'mnist':
        root = '../data/mnist'

        dsets['test'] = dset.MNIST(root=root, train=False, transform=trans_test, download=True)
        dsets['database'] = dset.MNIST(root=root, train=True, transform=trans_test, download=True)
        dsets['train'] = dset.MNIST(root=root, train=True, transform=trans_train, download=True)
    if 'cifar10' in job_dataset:
        if 'cifar100' in job_dataset:
            root = '../data/cifar100'
            dsets['test'] = dset.CIFAR100(root=root, train=False, transform=trans_test, download=True)
            dsets['database'] = dset.CIFAR100(root=root, train=True, transform=trans_test, download=True)
            dsets['train'] = dset.CIFAR100(root=root, train=True, transform=trans_train, download=True)
        else:
            root = '../data/cifar10'
            dsets['test'] = dset.CIFAR10(root=root, train=False, transform=trans_test, download=True)
            dsets['database'] = dset.CIFAR10(root=root, train=True, transform=trans_test, download=True)
            dsets['train'] = dset.CIFAR10(root=root, train=True, transform=trans_train, download=True)

    if job_dataset == 'fashion_mnist':
        root = '../data/fashion_mnist'
        dsets['test'] = dset.FashionMNIST(root=root, train=False, transform=trans_test, download=True)
        dsets['database'] = dset.FashionMNIST(root=root, train=True, transform=trans_test, download=True)
        dsets['train'] = dset.FashionMNIST(root=root, train=True, transform=trans_train, download=True)

    if 'imagenet' in job_dataset or 'nus_wide' in job_dataset or 'places365' in job_dataset:
        # load data file from path
        from publicVariables import data_list_path
        dsets["test"] = ImageList(open(data_list_path[job_dataset]["test"]).readlines(), \
                                  transform=trans_test)
        dsets["database"] = ImageList(open(data_list_path[job_dataset]["database"]).readlines(), \
                                      transform=trans_test)
        dsets["train"] = ImageList(open(data_list_path[job_dataset]["database"]).readlines(), \
                                      transform=trans_train)

    return dsets


def get_dsets_loader_by_dataset(job_dataset, batch_size=1, num_workers=16):
    # Parameters:
    #   job_dataset (str) -  name of dataset
    # Returns:
    #   Tuple(dsets, dsets_loaders).
    #   dsets is a dict containing the train and the test datasets
    #   dsets_loader is a dict containing the train and the test dataset loaders
    # Return type:
    #   tuple
    dset_loaders = {}
    dsets = get_dsets(job_dataset=job_dataset)
    dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                batch_size=batch_size, \
                                                shuffle=False, num_workers=num_workers)

    dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                                    batch_size=batch_size, \
                                                    shuffle=False, num_workers=num_workers)
    return dsets, dset_loaders


def get_network_config(net, hash_bit):
    # Set the configuration to load hashnet model
    import myNetwork
    config = {"hash_bit": hash_bit, "network": {}}

    if "ResNet" in net and "SERes" not in net:
        config["network"]["type"] = network.ResNetFc
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "VGG" in net:
        config["network"]["type"] = network.VGGFc
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "AlexNet" in net:
        config["network"]["type"] = network.AlexNetFc
        config["network"]["params"] = {"hash_bit":config["hash_bit"]}
    elif "ResNext" in net:
        config["network"]["type"] = myNetwork.ResNext
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "Inc_v3" in net:
        config["network"]["type"] = myNetwork.Inception
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "Inc_v4" in net:
        config["network"]["type"] = myNetwork.Inc_v4
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "IncRes_v2" in net:
        config["network"]["type"] = myNetwork.IncRes_v2
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "DenseNet" in net:
        config["network"]["type"] = myNetwork.DenseNetFc
        config["network"]["params"] = {"name":net, "hash_bit":config["hash_bit"]}
    elif "SENet" in net:
        config["network"]["type"] = myNetwork.SENet
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    elif "SERes" in net:
        config["network"]["type"] = myNetwork.SENet
        config["network"]["params"] = {"name": net, "hash_bit": config["hash_bit"]}
    return config

def load_model_class(net, hash_bit=48):
    config = get_network_config(net, hash_bit)
    net_config = config["network"]
    base_network = net_config["type"](**net_config["params"])
    return base_network


def make_one_hot(labels, C=10):
    import numpy as np
    one_hot = np.zeros([labels.shape[0], C])
    for i in range(labels.shape[0]):
        one_hot[i, labels[i].astype("uint16")] = 1
    return one_hot



def get_train_dsets(config):
    # prepare dataset for hashnet training
    import os
    import torchvision.datasets as dset
    from data_list import ImageList
    ## set pre-process
    prep_dict = {}
    job_dataset = config["dataset"]
    prep_dict["train_set1"] = get_trans(job_dataset=job_dataset, mode='test')
    prep_dict["train_set2"] = get_trans(job_dataset=job_dataset, mode='test')

    dsets = {}
    data_config = config["data"]

    if 'cifar10' in config["dataset"]:
        # cifar10 for 32*32, cifar10resize for 224*224
        if 'cifar100' in job_dataset:
            root = '../data/cifar100'
            if not os.path.exists(root):
                os.mkdir(root)
            dsets['train_set1'] = dset.CIFAR100(root=root, train=True, transform=prep_dict["train_set1"], download=True)
            dsets['train_set2'] = dset.CIFAR100(root=root, train=True, transform=prep_dict["train_set2"], download=True)
        else:
            root = '../data/cifar10'
            if not os.path.exists(root):
                os.mkdir(root)
            dsets['train_set1'] = dset.CIFAR10(root=root, train=True, transform=prep_dict["train_set1"], download=True)
            dsets['train_set2'] = dset.CIFAR10(root=root, train=True, transform=prep_dict["train_set2"], download=True)
    elif config["dataset"] == 'mnist':
        root = '../data/mnist'
        if not os.path.exists(root):
            os.mkdir(root)
        dsets["train_set1"] = dset.MNIST(root=root, train=True, transform=prep_dict["train_set1"], download=True)
        dsets["train_set2"] = dset.MNIST(root=root, train=True, transform=prep_dict["train_set2"], download=True)
    elif config["dataset"] == 'fashion_mnist':
        root = '../data/fashion_mnist'
        if not os.path.exists(root):
            os.mkdir(root)
        dsets["train_set1"] = dset.FashionMNIST(root=root, train=True, transform=prep_dict["train_set1"],
                                                download=True)
        dsets["train_set2"] = dset.FashionMNIST(root=root, train=True, transform=prep_dict["train_set2"],
                                                download=True)
    else:
        dsets['train_set1'] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(), \
                                        transform=prep_dict["train_set1"])
        dsets['train_set2'] = ImageList(open(data_config["train_set2"]["list_path"]).readlines(), \
                                        transform=prep_dict["train_set2"])
    return dsets


def get_trans(resize=256, crop_size=224, job_dataset='mnist', mode='test'):
    """

    Args:
        resize: Not used, placesholder for future use
        crop_size: Not used, placesholder for future use
        job_dataset (str) - dataset

    Returns:
        torchvision.transforms: transforms object in following process(train, test, query, etc..).
    """
    if job_dataset == 'mnist' or job_dataset == 'fashion_mnist':
        if mode == 'test' or mode =='train': # easy to train, same settings
            return transforms.Compose([
                # transforms.Resize(resize),
                # transforms.RandomResizedCrop(crop_size),
                transforms.Resize(size=(32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

    elif 'cifar' in job_dataset:
        if 'resize' in job_dataset:
            # 224*224 cifar10
            if mode == 'train':
                return transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            if mode == 'test':
                return transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor()
                ])

        else:
            # 32*32 cifar10
            if mode == 'train':
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            if mode == 'test':
                return transforms.Compose([
                    transforms.ToTensor()
                ])

    else:
        return transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])

def trans_train_resize_imagenet(resize = 224):
    return  transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor()
        ])


def load_net_inputs(ad_datapath, index, batch_size=1):
    if batch_size == 1:
        ad_imagelist = os.listdir(ad_datapath)
        ad_imagepath = ad_datapath + ad_imagelist[index]
        img = default_loader(ad_imagepath)
        t = trans_train_resize_imagenet()
        img_t = t(img)
        X = np.array(img_t.unsqueeze(0))
        inputs = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    else:
        inputs_np = np.zeros([batch_size, 3, 224, 224])
        ad_imagelist = os.listdir(ad_datapath)
        t = trans_train_resize_imagenet()
        for i in range(batch_size):
            ad_imagepath = ad_datapath + ad_imagelist[index+i]
            img = default_loader(ad_imagepath)
            img_t = t(img)
            inputs_np[i] = np.array(img_t)
        inputs = Variable(torch.Tensor(inputs_np).cuda(), requires_grad=True)
    return inputs


def load_model(net, model_path, model_dict_path, hash_bit=48):
    # ADD The new network names if saved as state_dict
    from publicVariables import new_networks
    import torch
    if net in new_networks:
        base_network = load_model_class(net, hash_bit=hash_bit)
        base_network.load_state_dict(torch.load(model_dict_path))
        return base_network.cuda().eval()
    else:
        model = torch.load(model_path)
        return model[0].cuda().eval()

def create_loading_path(job_dataset, net, iters):
    path_list = []
    snapshot_path = '../snapshot/'+job_dataset+'_48bit_'+ net +'_hashnet/'
    model_path = snapshot_path + 'iter_'+str(iters)+'_model.pth.tar'
    model_dict_path =  snapshot_path + 'iter_'+str(iters)+'_model_dict.pth.tar'
    query_path = './save_for_load/'+net+'/'+job_dataset+'_test_output_code_label.npz'
    database_path = './save_for_load/'+net+'/'+job_dataset+'_database_output_code_label.npz'
    path_list.append(snapshot_path)
    path_list.append(model_path)
    path_list.append(model_dict_path)
    path_list.append(query_path)
    path_list.append(database_path)
    return path_list

def load_net_params(net, job_dataset='imagenet'):
    from publicVariables import iters_list
    iters = iters_list[net]

    from publicFunctions import create_loading_path
    path_list = create_loading_path(job_dataset, net, iters)
    snapshot_path = path_list[0]
    model_path = path_list[1]
    model_dict_path = path_list[2]
    query_path = path_list[3]
    database_path = path_list[4]

    from publicFunctions import load_model
    model = load_model(net, model_path, model_dict_path)
    model = model.cuda().eval()

    return model, snapshot_path, query_path, database_path


class BaseNetworkSettings():
    def __init__(self):
        pass

    def get_model(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

    def get_out_code_label(self):
        raise NotImplementedError

    def get_dset_loaders(self):
        raise NotImplementedError

class NetworkSettings(BaseNetworkSettings):
    def __init__(self, job_dataset, hash_bit, net, snapshot_iter, batch_size=16):
        #super(NetworkSettings, self).__init__()
        snapshot_path = '../snapshot/%s_%dbit_%s_hashnet/' % (job_dataset, hash_bit, net)
        model_path = snapshot_path + 'iter_%05d_model.pth.tar' % (snapshot_iter)
        model_dict_path = snapshot_path + 'iter_%05d_model_dict.pth.tar' % (snapshot_iter)
        save_path = './save_for_load/%s/%dbits/%s/' % (job_dataset, hash_bit, net)
        test_save_path = save_path + 'test_output_code_label_%05d.npz' % (snapshot_iter)
        database_save_path = save_path + 'database_output_code_label_%05d.npz' % (snapshot_iter)
        self.job_dataset = job_dataset
        self.hash_bit = hash_bit
        self.net = net
        self.snapshot_iter = snapshot_iter
        self.batch_size = batch_size

        self.snapshot_path = snapshot_path
        self.model_path = model_path
        self.model_dict_path = model_dict_path
        self.save_path = save_path
        self.test_save_path = test_save_path
        self.database_save_path = database_save_path

    def get_model(self):
        '''
        model = load_model(self.net, self.model_path, self.model_dict_path)
        model = model.cuda().eval()
        '''
        model = load_model_class(self.net, hash_bit=self.hash_bit)
        model.load_state_dict(torch.load(self.model_dict_path))
        model = model.cuda().eval()
        self.model = model
        return model

    def get_out_code_label(self, part):
        save_path = self.database_save_path if part=='database' else self.test_save_path
        tmp = np.load(save_path)
        output, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']
        return output, code, multi_label

    def get_dset_loaders(self):
        import os
        num_workers = 0 if os.name == 'nt' else 16  # here is a bug for windows... only 0 works on win10, IDK why.
        dsets, dset_loaders = get_dsets_loader_by_dataset(self.job_dataset, batch_size=self.batch_size, num_workers=num_workers)
        return dset_loaders

def load_dset_params(job_dataset):
    from myGetAdv import get_dsets
    dsets = get_dsets(job_dataset)
    dset_test = dsets['test']
    dset_database = dsets['database']
    return dset_test, dset_database


def model_np_batch(model, inputs, batch_size=8):
    # 48 should be replaced by the shape of the output
    # print inputs.shape
    if len(inputs.shape) == 4:
        oCodeValue = np.zeros([inputs.shape[0], 48])
        i = 0
        while batch_size * i + batch_size < inputs.shape[0]:
            oCodeValue[batch_size * i:batch_size * i + batch_size] = model(
                inputs[batch_size * i:batch_size * i + batch_size])
            i += 1
        oCodeValue[batch_size * i:] = model(inputs[batch_size * i:])

    elif len(inputs.shape) == 5:
        oCodeValue = np.zeros([inputs.shape[0], inputs.shape[1], 48])
        i, j = 0, 0
        for j in range(inputs.shape[1]):
            i = 0
            while batch_size * i + batch_size < inputs.shape[0]:
                oCodeValue[batch_size * i:batch_size * i + batch_size, j] = model(
                    inputs[batch_size * i:batch_size * i + batch_size, j])
                i += 1
            oCodeValue[batch_size * i:, j] = model(inputs[batch_size * i:, j])

    return oCodeValue