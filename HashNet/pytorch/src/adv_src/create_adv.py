# -*- coding: utf-8 -*-
# This code should be ran with PyTorch 1.0
import foolbox
import torch
import numpy as np
import pickle
import torch.utils.data as util_data
import pre_process as prep
import os
import network
import torchvision.transforms as T
from data_list import ImageList
from PIL import Image
from torch.autograd import Variable

def get_config(mode='train'):
    # This config is the same as 
    config = {}
    
    if mode == 'train':
        config["num_iterations"] = 10000
        config["snapshot_interval"] = 3000
        config["hash_bit"] = 48
        config["dataset"] = "coco"
        config["network"] = {}
        config["network"]["type"] = network.ResNetFc
        config["network"]["params"] = {"name":'ResNet50', "hash_bit":config["hash_bit"]}
        config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay":0.0005, "nesterov":True}, "lr_type":"step", \
                               "lr_param":{"init_lr":0.0003, "gamma":0.5, "step":2000} }
        config["loss"] = {"l_weight":1.0, "q_weight":0, "l_threshold":15.0, "sigmoid_param":10./config["hash_bit"], "class_num":1}
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config["data"] = {"train_set1":{"list_path":"../../data/coco/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../../data/coco/train.txt", "batch_size":36}}
    elif mode == 'test' or mode == 'database':
        config["dataset"] = "coco"
        config["snapshot_path"] = "../../snapshot/coco_48bit_resnet50_hashnet/iter_16000_model.pth.tar"
        config["output_path"] = "../../snapshot/coco_48bit_resnet50_hashnet"
        config["data"] = {"database":{"list_path":"../../data/coco/database.txt", "batch_size":32}, \
                          "test":{"list_path":"../../data/coco/test.txt", "batch_size":32}}
        config["R"] = 5000
        config["prep"] = {"test_10crop":False, "resize_size":256, "crop_size":224}
        config["network"] = {}
    return config

def get_dsets_loader(mode='test'):
    # return a dsets and dsets_loader which contains data after crop and other pre-processing.
    prep_dict = {}
    config = get_config(mode)
    dsets = {}
    prep_config = config["prep"]
    dset_loaders = {}
    if mode=='train':
        
        prep_dict["train_set1"] = prep.image_train( \
                                resize_size=prep_config["resize_size"], \
                                crop_size=prep_config["crop_size"])

        data_config = config["data"]
        dsets["train_set1"] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(), \
                                    transform=prep_dict["train_set1"])
        dset_loaders["train_set1"] = util_data.DataLoader(dsets["train_set1"], \
                batch_size=data_config["train_set1"]["batch_size"], \
                shuffle=False, num_workers=4)
        
    if mode=='test' or mode=='database':
        # put test and dababase together
        
        prep_dict["database"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])

        data_config = config["data"]
        
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=16)   
                
        dsets["database"] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                transform=prep_dict["database"])
        dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                batch_size=data_config["database"]["batch_size"], \
                                shuffle=False, num_workers=16)
        

    return dsets, dset_loaders


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module




if __name__ == "__main__":
    snapshot_path = '../../snapshot/coco_48bit_resnet50_hashnet/'
    model_path = snapshot_path + 'iter_16000_model.pth.tar'
    model = torch.load(model_path)
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    # Intentionly, now we don't use eval(). This is because the original HashNet code doesn't use eval() model.
    # 
    # model = model.eval()
    
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0,255), num_classes = 48)
    attack = foolbox.attacks.FGSM(fmodel) 

    output_path = '../save_for_load/coco2017_test_output_code_label.npz'
    tmp = np.load(output_path)
    query_output, query_code, query_multi_label, query_img_list = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']

    dset_loaders = pickle.load(open('../save_for_load/test_database_loader.p', 'rb'))

    
    label = query_code[0]
    img = Image.open(query_img_list[0])
    
    trans = T.Compose([T.Scale([256, 256]), T.RandomResizedCrop(224)])
    dataCrop = trans(img)
    
    image =  np.moveaxis(np.array(dataCrop), -1, 0).astype('float32')
    adversarial = attack(image, label)
    
    a = fmodel.predictions(image)
    
    inputs = torch.Tensor(np.expand_dims(image, 0)).cuda()
    b = model(inputs)