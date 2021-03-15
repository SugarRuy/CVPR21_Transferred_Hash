import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

import myNetwork

optim_dict = {"SGD": optim.SGD, 'Adam': optim.Adam}

def trans_train(resize = 256, crop_size = 224, job_dataset = 'mnist'):
    if job_dataset == 'mnist' or job_dataset == 'fashion_mnist':
        return  transforms.Compose([
            #transforms.Resize(resize),
            #transforms.RandomResizedCrop(crop_size),
            transforms.Resize(size=(224, 224)),
            transforms.Grayscale(num_output_channels = 3), 
            transforms.ToTensor()
            ])
    if job_dataset == 'cifar10':
        return  transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
            ])
    if job_dataset == 'imagenet':
        return  transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
            ])
def make_one_hot(labels, C=10):
    one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
    for i in range(labels.size(0)):
        one_hot[i, labels[i]] = 1
    return one_hot
    
def train(config):
    ## set pre-process
    prep_dict = {}
    job_dataset = config["dataset"]
    prep_dict["train_set1"] = trans_train(job_dataset = job_dataset)
    prep_dict["train_set2"] = trans_train(job_dataset = job_dataset)
    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    if config["dataset"] == 'cifar10':

        
        dsets['train_set1'] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set1"])
        dsets['train_set2'] = ImageList(open(data_config["train_set2"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set2"])
        #root = '../../../../data/cifar10/pytorch_path'
        #if not os.path.exists(root):
        #    os.mkdir(root)
        #dsets["train_set1"] = dset.CIFAR10(root=root, train=True, transform=prep_dict["train_set1"], download=True)
        #dsets["train_set2"] = dset.CIFAR10(root=root, train=True, transform=prep_dict["train_set2"], download=True)
        
    if config["dataset"] == 'mnist':
        root = '../../../../data/mnist'

        if not os.path.exists(root):
            os.mkdir(root)
        dsets["train_set1"] = dset.MNIST(root=root, train=True, transform=prep_dict["train_set1"], download=True)
        dsets["train_set2"] = dset.MNIST(root=root, train=True, transform=prep_dict["train_set2"], download=True)
   
    if config["dataset"] == 'imagenet':
        
        dsets['train_set1'] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set1"])
        dsets['train_set2'] = ImageList(open(data_config["train_set2"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set2"])
        
    if config["dataset"] == 'fashion_mnist':
        root = '../../../../data/fashion_mnist'

        if not os.path.exists(root):
            os.mkdir(root)
        dsets["train_set1"] = dset.FashionMNIST(root=root, train=True, transform=prep_dict["train_set1"], download=True)
        dsets["train_set2"] = dset.FashionMNIST(root=root, train=True, transform=prep_dict["train_set2"], download=True)
  
    
    # The reason of the randomness in training process is
    # "shuffle=True"
    dset_loaders["train_set1"] = util_data.DataLoader(dsets["train_set1"], \
            batch_size=data_config["train_set1"]["batch_size"], \
            shuffle=True, num_workers=4)


    dset_loaders["train_set2"] = util_data.DataLoader(dsets["train_set2"], \
            batch_size=data_config["train_set2"]["batch_size"], \
            shuffle=True, num_workers=4)


    ## set base network
    net_config = config["network"]
    base_network = net_config["type"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                      {"params":base_network.hash_layer.parameters(), "lr":10}]
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train   
    len_train1 = len(dset_loaders["train_set1"])
    len_train2 = len(dset_loaders["train_set2"])

    for i in range(config["num_iterations"]):
        if i % config["snapshot_interval"] == 0:
            #torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
            #    "iter_{:05d}_model.pth.tar".format(i)))
            torch.save(base_network.state_dict(), osp.join(config["output_path"], \
                                                           "iter_{:05d}_model_dict.pth.tar".format(i)))

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train1 == 0:
            iter1 = iter(dset_loaders["train_set1"])
        if i % len_train2 == 0:
            iter2 = iter(dset_loaders["train_set2"])
        inputs1, labels1 = iter1.next()
        inputs2, labels2 = iter2.next()
        if job_dataset == 'mnist' or job_dataset == 'fashion_mnist':
            # convert labels1 & s2 to one hot array
            labels1, labels2 = make_one_hot(labels1), make_one_hot(labels2)
        
        
        if use_gpu:
            inputs1, inputs2, labels1, labels2 = \
                Variable(inputs1).cuda(), Variable(inputs2).cuda(), \
                Variable(labels1).cuda(), Variable(labels2).cuda()
        else:
            inputs1, inputs2, labels1, labels2 = Variable(inputs1), \
                Variable(inputs2), Variable(labels1), Variable(labels2)
           
        inputs = torch.cat((inputs1, inputs2), dim=0)
        outputs = base_network(inputs)
        similarity_loss = loss.pairwise_loss(outputs.narrow(0,0,inputs1.size(0)), \
                                 outputs.narrow(0,inputs1.size(0),inputs2.size(0)), \
                                 labels1, labels2, \
                                 sigmoid_param=config["loss"]["sigmoid_param"], \
                                 l_threshold=config["loss"]["l_threshold"], \
                                 class_num=config["loss"]["class_num"])

        similarity_loss.backward()

        print(("Iter: {:05d}, loss: {:.3f}".format(i, similarity_loss.item())))
        config["out_file"].write("Iter: {:05d}, loss: {:.3f}\n".format(i, similarity_loss.item()))
        #print("Iter: {:05d}, loss: {:.3f}".format(i, similarity_loss.float().data[0]))
        #config["out_file"].write("Iter: {:05d}, loss: {:.3f}\n".format(i, \
        #    similarity_loss.float().data[0]))
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')

    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='imagenet', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='SENet154', help="base network type")
    parser.add_argument('--prefix', type=str, default='', help="save path prefix")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--class_num', type=float, default=1.0, help="positive negative pairs balance weight")
    args = parser.parse_args()

    #args.gpu_id = '0'
    #args.dataset = 'imagenet'
    #args.hash_bit = args.hash_bit
    #args.net = 'Inc_v3'
    args.prefix = args.net+'_hashnet'
    #args.prefix = 'ResNet34_hashnet'
    
    #args.lr = args.lr
    #args.class_num = args.class_num
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    # train config  
    config = {}
    config["num_iterations"] = 100001
    config["snapshot_interval"] = 1000
    config["dataset"] = args.dataset
    config["hash_bit"] = args.hash_bit
    config["output_path"] = "../snapshot/"+config["dataset"]+"_"+ \
                            str(config["hash_bit"])+"bit_"+args.prefix
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["network"] = {}
    if "ResNet" in args.net and "SERes" not in args.net:
        config["network"]["type"] = network.ResNetFc
        config["network"]["params"] = {"name":args.net, "hash_bit":config["hash_bit"]}
    elif "VGG" in args.net:
        config["network"]["type"] = network.VGGFc
        config["network"]["params"] = {"name":args.net, "hash_bit":config["hash_bit"]}
    elif "AlexNet" in args.net:
        config["network"]["type"] = network.AlexNetFc
        config["network"]["params"] = {"hash_bit":config["hash_bit"]}
    elif "Inc_v3" in args.net:
        config["network"]["type"] = myNetwork.InceptionV3
        config["network"]["params"] = {"name":args.net, "hash_bit":config["hash_bit"]}
    elif "Inc_v4" in args.net:
        config["network"]["type"] = myNetwork.InceptionV4
        config["network"]["params"] = {"name":args.net, "hash_bit":config["hash_bit"]}    
    elif "IncRes_v2" in args.net:
        config["network"]["type"] = myNetwork.IncRes_v2
        config["network"]["params"] = {"hash_bit":config["hash_bit"]}
    elif "DenseNet" in args.net:
        config["network"]["type"] = myNetwork.DenseNetFc
        config["network"]["params"] = {"name":args.net, "hash_bit":config["hash_bit"]}
    elif "SENet" in args.net:
        config["network"]["type"] = myNetwork.SENet
        config["network"]["params"] = {"name": args.net, "hash_bit": config["hash_bit"]}
    elif "SERes" in args.net:
        config["network"]["type"] = myNetwork.SENet
        config["network"]["params"] = {"name": args.net, "hash_bit": config["hash_bit"]}


    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}

    optimizer = "Adam"
    if optimizer == 'SGD':
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay":0.0005, "nesterov":True}, "lr_type":"step", \
                               "lr_param":{"init_lr":args.lr, "gamma":0.5, "step":10000} }

    elif optimizer == 'Adam':
        config["optimizer"] = {"type":"Adam",  "optim_params":{"lr":1.0, \
                               "weight_decay":0.0005}, "lr_type":"step",\
                                "lr_param":{"init_lr":args.lr, "gamma":0.5, "step":10000}}


    config["loss"] = {"l_weight":1.0, "q_weight":0, "l_threshold":15.0, "sigmoid_param":10./config["hash_bit"], "class_num":args.class_num}

    if config["dataset"] == "imagenet":
        '''
        config["data"] = {"train_set1":{"list_path":"../data/imagenet/train.txt", "batch_size":16}, \
                          "train_set2":{"list_path":"../data/imagenet/train.txt", "batch_size":16}}
        '''
        config["data"] = {"train_set1": {"list_path": "../data/imagenet/database.txt", "batch_size": 4}, \
                          "train_set2": {"list_path": "../data/imagenet/database.txt", "batch_size": 4}}

    elif config["dataset"] == "nus_wide":
        config["data"] = {"train_set1":{"list_path":"../data/nus_wide/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../data/nus_wide/train.txt", "batch_size":36}}
    elif config["dataset"] == "coco":
        config["data"] = {"train_set1":{"list_path":"../data/coco/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../data/coco/train.txt", "batch_size":36}}
    elif config["dataset"] == 'mnist':
        config["data"] = {"train_set1":{"list_path":"", "batch_size":16}, \
                          "train_set2":{"list_path":"", "batch_size":16}}
    elif config["dataset"] == 'cifar10':
        config["data"] = {"train_set1":{"list_path":"../data/cifar10/train.txt", "batch_size":16}, \
                          "train_set2":{"list_path":"../data/cifar10/train.txt", "batch_size":16}}
    elif config["dataset"] == 'fashion_mnist':
        # no file path needed
        config["data"] = {"train_set1":{"list_path":"", "batch_size":16}, \
                          "train_set2":{"list_path":"", "batch_size":16}}
    print(config["loss"])
    train(config)
