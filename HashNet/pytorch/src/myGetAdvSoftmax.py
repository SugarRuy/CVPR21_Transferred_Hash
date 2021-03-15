import numpy as np
import torch
import torch.nn as nn
import os

from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method
from myGetAdvVulnerable import get_unique_index
from torch.autograd import Variable
import torchvision.transforms as transforms
from myTrainSoftmax import get_softmax_basenet, load_softmax_model
from myGetAdv import get_dsets

def softmax_targetedAttack(model, img_t, targetLabel, eps=1.0 / 255, l_inf_max=32, loss=nn.CrossEntropyLoss(),
                   decay_factor=1.0, t_prob=0.5, bShowProcess=False, bEarlyStop=False, adv_method='iFGSM'):
    #if not isinstance(targetLabel_var, torch.cuda.FloatTensor):
    targetLabelOneHot = np.zeros([1])
    targetLabelOneHot[0] = int(targetLabel)
    targetLabelOneHot = Variable(torch.Tensor(targetLabelOneHot).cuda()).long()


    def ndarray2PILImage(img_np):
        t = transforms.ToPILImage()
        return t(np.moveaxis(np.uint8(img_np * 255), 0, -1))

    def randomly_input_diversity(adv_np, p=0.5):
        # internal function from myGetAdv where it's a global func.
        def randomly_resize_transform():

            import random
            resize_size = random.randint(199 / 2, 224 / 2) * 2
            return transforms.Compose([
                transforms.Resize(size=(resize_size, resize_size)),
                transforms.Pad(112 - resize_size / 2),
                transforms.ToTensor()
            ])
        import random
        t_prob = random.random()
        if t_prob > p:
            # print 'original adv'
            return Variable(torch.Tensor(adv_np).unsqueeze_(0).cuda(), requires_grad=True)
        else:
            t = randomly_resize_transform()

            adv_pil = ndarray2PILImage(adv_np)
            adv_tensor = t(adv_pil).unsqueeze_(0)
            # print adv_tensor.size()
            return Variable(adv_tensor.cuda(), requires_grad=True)

    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    #output_softmax = nn.functional.softmax(output)
    # print torch.max(output)
    #print output_softmax

    print '...targeted %s begin....' % (adv_method)
    print 'Source Label:%d, Target Label:%d' % (np.argmax(output.cpu().data.numpy()), targetLabel)
    i = 0
    grad_var = 0
    while i < l_inf_max:
        adv_np = adv.cpu().data.numpy()
        adv_np = np.clip(adv_np, 0.0, 1.0)

        # inputs_adv is used as the input of model;
        # adv is used as the adv result of iter i.
        # They are different when using Diversity Inputs method

        if adv_method == 'iFGSM' or adv_method == 'miFGSM':
            inputs_adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
        elif adv_method == 'iFGSMDI' or adv_method == 'miFGSMDI':
            inputs_adv = randomly_input_diversity(adv_np[0], p=t_prob)

        output = model(inputs_adv)
        celoss = loss(output, targetLabelOneHot.detach())
        celoss.backward(retain_graph=True)

        if adv_method == 'iFGSM':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSM':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'iFGSMDI':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSMDI':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'PGD':
            grad_var = inputs_adv.grad

        adv = adv - eps * torch.sign(grad_var)

        if bShowProcess:
            output = model(adv)
            output_label = np.argmax(output.data.cpu().numpy())
            print "adv label: %d, target label: %d" % (output_label, targetLabel)
            print "adv mse loss:%f" % (celoss)
            if bEarlyStop:
                if output_label == targetLabel:
                    break
        elif bEarlyStop:
            output = model(adv)
            output_label = np.argmax(output.cpu().data.numpy())
            if output_label == targetLabel:
                break
        i = i + 1

    adv_np = adv.cpu().data.numpy()
    adv_np[adv_np < 0] = 0
    adv_np[adv_np > 1] = 1
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    output = model(adv)
    output_label = np.argmax(output.cpu().data.numpy())
    print '...Final Label : ', output_label
    return adv


def softmax_targetedAttack_NAG(model, img_t, targetLabel, eps=1.0 / 255, l_inf_max=32, loss=nn.CrossEntropyLoss(),
                               decay_factor=1.0, t_prob=0.5, bShowProcess=False, bEarlyStop=False, random_noise_level=32.0, var_lambda = 1.0,
                               noise_distribution='uniform', adv_method='iFGSM'):
    #if not isinstance(targetLabel_var, torch.cuda.FloatTensor):
    #targetLabel_var =
    targetLabel_var = Variable(torch.Tensor(np.array([int(targetLabel)])).cuda()).long()
    targetLabelOneHot = np.zeros([1000])
    targetLabelOneHot[int(targetLabel)] = 1
    targetLabelOneHot = Variable(torch.Tensor(targetLabelOneHot).cuda()).long()

    def ndarray2PILImage(img_np):
        t = transforms.ToPILImage()
        return t(np.moveaxis(np.uint8(img_np * 255), 0, -1))

    def randomly_input_diversity(adv_np, p=0.5):
        # internal function from myGetAdv where it's a global func.
        def randomly_resize_transform():

            import random
            resize_size = random.randint(199 / 2, 224 / 2) * 2
            return transforms.Compose([
                transforms.Resize(size=(resize_size, resize_size)),
                transforms.Pad(112 - resize_size / 2),
                transforms.ToTensor()
            ])
        import random
        t_prob = random.random()
        if t_prob > p:
            # print 'original adv'
            return Variable(torch.Tensor(adv_np).unsqueeze_(0).cuda(), requires_grad=True)
        else:
            t = randomly_resize_transform()

            adv_pil = ndarray2PILImage(adv_np)
            adv_tensor = t(adv_pil).unsqueeze_(0)
            # print adv_tensor.size()
            return Variable(adv_tensor.cuda(), requires_grad=True)
    #print "RandomNoiseLevel:", random_noise_level
    X = np.array(img_t.unsqueeze(0))
    adv = Variable(torch.Tensor(X).cuda(), requires_grad=True)
    output = model(adv)
    #output_softmax = nn.functional.softmax(output)
    # print torch.max(output)
    #print output_softmax

    print '...targeted %s begin....' % (adv_method)
    print 'Source Label:%d, Target Label:%d' % (np.argmax(output.cpu().data.numpy()), targetLabel_var)
    i = 0
    grad_var = 0
    while i < l_inf_max:
        adv_np = adv.cpu().data.numpy()
        adv_np = np.clip(adv_np, 0.0, 1.0)

        # inputs_adv is used as the input of model;
        # adv is used as the adv result of iter i.
        # They are different when using Diversity Inputs method

        if adv_method == 'iFGSM' or adv_method == 'miFGSM':
            inputs_adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
        elif adv_method == 'iFGSMDI' or adv_method == 'miFGSMDI':
            inputs_adv = randomly_input_diversity(adv_np[0], p=t_prob)

        output = model(inputs_adv)
        loss1 = loss(output, targetLabel_var.detach())

        noise_shape = [5, 3, 224, 224]
        if noise_distribution == 'uniform':
            random_noise = np.random.uniform(-random_noise_level, random_noise_level, noise_shape)
        elif noise_distribution == 'normal' or noise_distribution == 'Gaussian':
            random_noise = np.random.normal(loc=0.0, scale=random_noise_level / 3, size=noise_shape)
            random_noise = np.clip(random_noise, -random_noise_level, random_noise_level)

        adv_intermediate_noised = inputs_adv + Variable(torch.Tensor(random_noise).cuda())
        output_noised = model(adv_intermediate_noised)

        #loss2 = loss(output_noised, targetLabel_var.detach())
        #loss2 = nn.L1Loss()(output_noised, output.detach())
        loss2 = torch.mean(torch.norm(output_noised - output.detach(), p=1, dim=1) / output.shape[-1], dim=0)
        #loss2 = nn.L1Loss()(output_noised, targetLabelOneHot.detach().float().cuda())
        #loss2 = torch.mean(torch.norm(output_noised - targetLabel_var.detach().float().cuda(), p=1, dim=1) / output.shape[-1], dim=0)

        total_celoss = loss1 + var_lambda * loss2
        #print "loss1, loss2, total:", loss1.cpu().data.numpy()[0], loss2.cpu().data.numpy()[0], total_celoss.cpu().data.numpy()[0]
        total_celoss.backward(retain_graph=True)

        if adv_method == 'iFGSM':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSM':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'iFGSMDI':
            grad_var = inputs_adv.grad
        elif adv_method == 'miFGSMDI':
            grad_var = grad_var * decay_factor + inputs_adv.grad / torch.norm(inputs_adv.grad, p=1)
        elif adv_method == 'PGD':
            grad_var = inputs_adv.grad

        adv = adv - eps * torch.sign(grad_var)

        if bShowProcess:
            output = model(adv)
            output_label = np.argmax(output.data.cpu().numpy())
            print "adv label: %d, target label: %d" % (output_label, targetLabel)
            print "adv total loss:%f" % (total_celoss)
            if bEarlyStop:
                if output_label == targetLabel:
                    break
        elif bEarlyStop:
            output = model(adv)
            output_label = np.argmax(output.cpu().data.numpy())
            if output_label == targetLabel:
                break
        i = i + 1

    adv_np = adv.cpu().data.numpy()
    adv_np = np.clip(adv_np, 0.0, 1.0)
    adv = Variable(torch.Tensor(adv_np).cuda(), requires_grad=True)
    output = model(adv)
    output_label = np.argmax(output.cpu().data.numpy())
    print '...Final Label : ', output_label
    return adv


def get_ori_imgs_and_labels(dset, imgs_size=1024):
    ori_imgs_shape = np.array([imgs_size, 3, 224, 224])
    dset_size = len(dset['test'])
    random_index = np.random.choice(dset_size, imgs_size, replace=False)
    ori_imgs = np.zeros(ori_imgs_shape)
    softmax_labels = np.zeros([2, imgs_size])

    for i in range(imgs_size):
        index = random_index[i]
        img_t = dset['test'][index][0]
        label = int(np.argmax(dset['test'][index][1]))
        target_label = np.random.randint(100)
        while target_label == label:
            print "target/original confilt at:%d, redo."%(i)
            target_label = np.random.randint(100)
        #print label, target_label
        ori_imgs[i] = img_t.numpy()
        #softmax_labels[:,i] = np.array([label, target_label])
        softmax_labels[0, i] = label
        softmax_labels[1, i] = target_label
    return ori_imgs, softmax_labels, random_index



def get_softmax_adv_by_method(model, img_t, targetLabel, step_size, step_max, adv_method):

    if adv_method == 'FGSM':
        adv__ = softmax_targetedAttack(model, img_t, targetLabel, eps=step_size / 255, l_inf_max=1.0,
                                       loss=nn.CrossEntropyLoss(),
                                       bShowProcess=False, bEarlyStop=False, adv_method='iFGSM')
    elif adv_method == 'NAG':
        adv__ = softmax_targetedAttack_NAG(model, img_t, targetLabel, eps=step_size / 255, l_inf_max=step_max, loss=nn.CrossEntropyLoss(),
                                   decay_factor=1.0, t_prob=0.5, bShowProcess=False, bEarlyStop=False,
                                   random_noise_level=random_nosie_level/255, var_lambda=var_lambda,
                                   noise_distribution=noise_distribution, adv_method='iFGSM')
    else:
        adv__ = softmax_targetedAttack(model, img_t, targetLabel, eps=step_size / 255, l_inf_max=step_max,
                                       loss=nn.CrossEntropyLoss(),
                                       bShowProcess=False, bEarlyStop=False, adv_method=adv_method)

    return adv__


if __name__ == "__main__":
    # job_dataset = 'mnist'
    import argparse
    parser = argparse.ArgumentParser(description='Softmax')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet34', help="network")

    parser.add_argument('--imgs_size', type=int, default=1024, help="imgs_size")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--epoch', type=int, default=9, help="Using which epoch checkpoint model")
    parser.add_argument('--adv_method', type=str, default='NAG', help="adv method")
    parser.add_argument('--step_max', type=int, default=32, help="maximum step")
    parser.add_argument('--step_size', type=float, default=1.0, help="step size")

    parser.add_argument('--var_lambda', type=float, default=99.9, help="lbd to balance loss1 and loss2")
    parser.add_argument('--noise', type=str, default='uniform', help="noise distribution")
    parser.add_argument('--noise_level', type=float, default=32.0, help="random_noise_level")

    args = parser.parse_args()
    # NAG is not finished
    net = args.net

    # net candidates:
    net_values = ['ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNext101_32x4d', 'SEResNet50', 'Inc_v3']

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

    dsets = get_dsets(job_dataset)


    if not os.path.exists('./save_for_load/softmax/'):
        os.makedirs('./save_for_load/softmax/')

    # generate or load ori_imgs and it labels and target labels
    path_softmax_ori_imgs = './save_for_load/softmax/ori_imgs_%d.npy'%(imgs_size)
    path_softmax_labels = './save_for_load/softmax/ori_labels_%d.npy'%(imgs_size)
    path_softmax_random_index = './save_for_load/softmax/ori_random_index_%d.npy'%(imgs_size)
    if os.path.exists(path_softmax_ori_imgs) and os.path.exists(path_softmax_labels):
        ori_imgs = np.load(path_softmax_ori_imgs)
        softmax_labels = np.load(path_softmax_labels)
        ori_labels, target_labels = softmax_labels[0], softmax_labels[1]
        random_index = np.load(path_softmax_random_index)
    else:
        ori_imgs,  softmax_labels, random_index = get_ori_imgs_and_labels(dsets, imgs_size=imgs_size)
        ori_labels, target_labels = softmax_labels[0], softmax_labels[1]
        np.save(path_softmax_ori_imgs, ori_imgs)
        np.save(path_softmax_labels, softmax_labels)
        np.save(path_softmax_random_index, random_index)

    # generate or load adv_imgs
    if adv_method == 'NAG':
        NAG_save_folder = './save_for_load/softmax/NAG_noise%s_lambda%s/'%(str(random_nosie_level), str(var_lambda))
        if not os.path.exists(NAG_save_folder):
            os.makedirs(NAG_save_folder)
        path_softmax_adv_imgs = NAG_save_folder+'/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy' % (net, adv_method, step_size, step_max, imgs_size)
    else:
        path_softmax_adv_imgs = './save_for_load/softmax/%s_%s_adv_imgs_stepsize%s_stepmax%s_%d.npy'%(net, adv_method, step_size, step_max, imgs_size)
    if os.path.exists(path_softmax_adv_imgs):
        adv_imgs = np.load(path_softmax_adv_imgs)
    else:
        adv_imgs = np.zeros([imgs_size, 3, 224, 224])
        for i in range(imgs_size):
            print "sample i:", i
            img_t = torch.Tensor(ori_imgs[i])
            targetLabel = target_labels[i]
            adv__ = get_softmax_adv_by_method(model, img_t, targetLabel, step_size, step_max, adv_method)
            adv_imgs[i] = adv__.cpu().data.numpy()
        np.save(path_softmax_adv_imgs, adv_imgs)

