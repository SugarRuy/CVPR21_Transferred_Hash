# -*- coding: utf-8 -*-
import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as util_data

import torchvision.transforms as transforms

from torchvision import models
import pretrainedmodels

from torch.autograd import Variable

from myGetAdv import get_dsets

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}


def get_softmax_basenet(net):
    # get the pytorch model for specific net input
    if net == 'ResNet152':
        model = models.resnet152(pretrained=True)
    elif net == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif net == 'ResNet34':
        model = models.resnet34(pretrained=True)
    elif net == 'ResNet101':
        model = models.resnet101(pretrained=True)
    elif net == 'ResNext101_32x4d':
        model = pretrainedmodels.resnext101_32x4d(pretrained='imagenet')
    elif net == 'IncRes_v2':
        # model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        print 'Fuck! Stop! Not Finished!'
    elif net == 'Inc_v3':
        from myNetwork import get_customInceptionV3
        # model = pretrainedmodels.inceptionresnetv2()
        model = get_customInceptionV3(100)

        # model = models.inception_v3()
    elif net == 'DenseNet161':
        model = models.densenet161(pretrained=True)
    elif net == 'SEResNet50':
        model = pretrainedmodels.se_resnet50(pretrained='imagenet')

    return model

def load_softmax_model(net, job_dataset='imagenet', epoch=9):
    # load a trained pytorch model, with specific model epoch
    model_dict_path =  '../snapshot/%s_%s_softmax/epoch_%s.dict' %(job_dataset, net, str(epoch))
    try:
        model_path = '../snapshot/%s_%s_softmax/epoch_%s' %(job_dataset, net, str(epoch))
        model = torch.load(model_path)
    except:
        model = get_softmax_basenet(net)
        model_dict = torch.load(model_dict_path)
        model.load_state_dict(model_dict)

    return model.cuda().eval()

if __name__ == "__main__":
    # job_dataset = 'mnist'
    parser = argparse.ArgumentParser(description='Softmax')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='SEResNet50', help="network")
    parser.add_argument('--suffix', type=str, default='', help="save path suffix")
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer")
    parser.add_argument('--batch', type=int, default=32, help="batch size")

    args = parser.parse_args()


    net = args.net
    net_with_suffix = net if args.suffix == '' else net + '_' + args.suffix

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    optim_type = args.optim
    batch_size = args.batch
    # job_dataset_value = ['cifar10', 'fashion_mnist']
    # job_dataset_value = ['imagenet']
    job_dataset_value = ['imagenet']

    for job_dataset in job_dataset_value:
        print "Softmax Training Start: %s on %s" % (net, job_dataset)
        output_path = '../snapshot/' + job_dataset + '_%s_softmax/' %(net_with_suffix)
        if not osp.exists(output_path):
            os.mkdir(output_path)

        print job_dataset

        model = get_softmax_basenet(net)

        model.train(mode=True)
        model = model.cuda()

        dsets = get_dsets(job_dataset)
        data_loader = {}
        data_loader['train'] = util_data.DataLoader(dsets['database'], batch_size=batch_size, shuffle=True, num_workers=12)
        data_loader['test'] = util_data.DataLoader(dsets['test'], batch_size=batch_size, shuffle=True, num_workers=12)

        train_loader = data_loader['train']
        test_loader = data_loader['test']

        if optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

        criterion = nn.CrossEntropyLoss()

        log = open('../snapshot/' + job_dataset + '_%s_softmax/log.txt' % (net_with_suffix), "w")

        for epoch in range(10):
            # trainning
            ave_loss = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                optimizer.zero_grad()

                x, target = x.cuda(), target.cuda()
                if len(target.shape) > 1:
                    _, target = torch.max(target, 1)
                x, target = Variable(x), Variable(target)
                out = model(x)
                #print out
                if  net != 'Inc_v3' and net != 'IncRes_v2':
                    loss = criterion(out, target)
                else:
                    loss1 = criterion(out[0], target)
                    loss2 = criterion(out[1], target)
                    loss = loss1 + 0.4 * loss2
                ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
                loss.backward()
                optimizer.step()
                print 'training batch_idx:', batch_idx
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                    print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                        epoch, batch_idx + 1, ave_loss)
                    log.write('==>>> epoch: {}, batch index: {}, train loss: {:.6f}\n'.format(
                        epoch, batch_idx + 1, ave_loss))

            # testing
            correct_cnt, ave_loss = 0, 0
            total_cnt = 0
            for batch_idx, (x, target) in enumerate(test_loader):

                x, target = x.cuda(), target.cuda()
                if len(target.shape) > 1:
                    _, target = torch.max(target, 1)
                x, target = Variable(x, volatile=True), Variable(target, volatile=True)
                out = model(x)
                if net != 'Inc_v3' and net != 'IncRes_v2':
                    loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                else:
                    loss1 = criterion(out[0], target)
                    loss2 = criterion(out[1], target)
                    loss = loss1 + 0.4 * loss2
                    _, pred_label = torch.max(out[0].data, 1)

                total_cnt += x.data.size()[0]
                correct_cnt += (pred_label == target.data).sum()
                # smooth average
                ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                    print '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                        epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt)
                    log.writelines('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}\n'.format(
                        epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))
            #torch.save(nn.Sequential(model), '../snapshot/' + job_dataset + '_%s_softmax/epoch_' % (net_with_suffix) + str(epoch))
            torch.save(model.state_dict(),
                       '../snapshot/' + job_dataset + '_%s_softmax/epoch_%s.dict' % (net_with_suffix, str(epoch)))
        log.close()