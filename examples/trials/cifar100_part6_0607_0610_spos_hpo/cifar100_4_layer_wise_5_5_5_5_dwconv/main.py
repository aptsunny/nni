'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import argparse
from typing import List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import progress_bar
import nni
import numpy as np


_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def fast_19_lr_parameters(model, lr_group, arch_search=None):
    base_conv = []

    rest_name = []
    for name, param in model.named_parameters():
        if name.find('layer'):
            base_conv.append(param)
            # base_conv.append(param)
            # print("requires_grad: True ", name)
        else:
            rest_name.append(param)
            # rest_name.append(param)
            # print("requires_grad: False ", name)
    # rest_name.sort()

    # 18 + 1 choices
    choice = []
    if len(rest_name)==108: # 3 * 3 *6 *2
        for i in range(0, len(rest_name), 6):
            a = rest_name[i:i+6]
            choice.append(a)

    choice.append(base_conv)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def fast_4_lr_parameters(model, lr_group, arch_search=None):
    base_conv = []

    rest_name = []
    for name, param in model.named_parameters():
        if name.find('layer'):
            base_conv.append(param)
            # base_conv.append(param)
            # print("requires_grad: True ", name)
        else:
            """
            layer1_features.0.conbine.0.weight
            layer1_features.0.conbine.1.weight
            layer1_features.0.conbine.1.bias
            
            layer1_features.0.conbine.3.weight
            layer1_features.0.conbine.4.weight
            layer1_features.0.conbine.4.bias
            
            layer3_features.0.conbine.0.weight
            layer3_features.0.conbine.1.weight
            layer3_features.0.conbine.1.bias
            
            layer3_features.0.conbine.3.weight
            layer3_features.0.conbine.4.weight
            layer3_features.0.conbine.4.bias
            """
            rest_name.append(param)
            # rest_name.append(param)
            # print("requires_grad: False ", name)
    # rest_name.sort()

    choice = []
    if len(rest_name)==12:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i+3]
            choice.append(a)

    choice.append(base_conv)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def forfor(a):
    return [item for sublist in a for item in sublist]

def fast_17_lr_parameters_before(model, lr_group, arch_search=None):
    # 0,1,2,3,4,5,6,7,8
    # (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)->(3,5,7)
    # (3,3),(3,5),(3,7),(5,3),(5,5),(5,7),(7,3),(7,5),(7,7)
    # (1+1+2*9 + 1+1+2*9 )*3+1=121
    base_conv = []
    rest_name = []
    for name, param in model.named_parameters():
        # rest_name.append(name)
        rest_name.append(param)
        """
        if name.find('layer'):
            # base_conv.append(param)
            base_conv.append(name)
            # print("requires_grad: True ", name)
        else:
            # rest_name.append(param)
            rest_name.append(name)
            # print("requires_grad: False ", name)
        """
    # rest_name.sort()
    conv_level = []
    if len(rest_name) == 121:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            conv_level.append(a)
    # choice 41,包括17个(1:0/2:1/9:20/10:21/17:40)5个+
    # 一共12个 每个layer 6个，(3:02,04,06 /4:08,10,12 /5:14,16,18 /6:03,09,15 /7:05,11,17 /8:07,13,19)
    # (11:22,24,26/ 12:28,30,32 / 13:34,36,38 / 14:23,29,35 /15:25,31,37 /16:27,33,39)
    layer1_1_3=[]
    layer1_1_5=[]
    layer1_1_7=[]
    layer1_2_3=[]
    layer1_2_5=[]
    layer1_2_7=[]

    layer3_1_3=[]
    layer3_1_5=[]
    layer3_1_7=[]
    layer3_2_3 = []
    layer3_2_5 = []
    layer3_2_7 = []

    for i in range(0, len(conv_level)):
        if i in [2,4,6]:
            layer1_1_3.append(conv_level[i])
        elif i in [8,10,12]:
            layer1_1_5.append(conv_level[i])
        elif i in [14,16,18]:
            layer1_1_7.append(conv_level[i])
        elif i in [3,9,15]:
            layer1_2_3.append(conv_level[i])
        elif i in [5,11,17]:
            layer1_2_5.append(conv_level[i])
        elif i in [7,13,19]:
            layer1_2_7.append(conv_level[i])

        elif i in [22,24,26]:
            layer3_1_3.append(conv_level[i])
        elif i in [28,30,32]:
            layer3_1_5.append(conv_level[i])
        elif i in [34,36,38]:
            layer3_1_7.append(conv_level[i])
        elif i in [23,29,35]:
            layer3_2_3.append(conv_level[i])
        elif i in [25,31,37]:
            layer3_2_5.append(conv_level[i])
        elif i in [27,33,39]:
            layer3_2_7.append(conv_level[i])

    choice = []
    choice.append(conv_level[0])
    choice.append(conv_level[1])
    choice.append(conv_level[20])
    choice.append(conv_level[21])
    choice.append(conv_level[40])


    # choice.append(layer1_1_3)
    # choice.append(layer1_1_5)
    # choice.append(layer1_1_7)
    # choice.append(layer1_2_3)
    # choice.append(layer1_2_5)
    # choice.append(layer1_2_7)
    #
    # choice.append(layer3_1_3)
    # choice.append(layer3_1_5)
    # choice.append(layer3_1_7)
    # choice.append(layer3_2_3)
    # choice.append(layer3_2_5)
    # choice.append(layer3_2_7)

    choice.append(forfor(layer1_1_3))
    choice.append(forfor(layer1_1_5))
    choice.append(forfor(layer1_1_7))
    choice.append(forfor(layer1_2_3))
    choice.append(forfor(layer1_2_5))
    choice.append(forfor(layer1_2_7))

    choice.append(forfor(layer3_1_3))
    choice.append(forfor(layer3_1_5))
    choice.append(forfor(layer3_1_7))
    choice.append(forfor(layer3_2_3))
    choice.append(forfor(layer3_2_5))
    choice.append(forfor(layer3_2_7))

    # choice.append(base_conv)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def fast_17_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    for name, param in model.named_parameters():
        rest_name.append(param)
    choice = []
    if len(rest_name) == 50:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def fast_9_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 26:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            b = figure_[i:i+3]
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def fast_ofa_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice = []
    choice = []
    if len(rest_name) == 129:
        loc = 0
        # 15
        split_list = [3, 6, 9, 9, 13, 13, 9, 9, 13, 13, 13, 13, 3, 1, 2]
        for i in range(0, len(split_list), 1):
            st = loc
            en = loc + split_list[i]
            # print(loc, loc + split_list[i])
            b = figure_[st:en]
            a = rest_name[st:en]
            loc = en
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups


def fast_hpo_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []

    if len(rest_name) == 44:
        loc = 0
        # 15
        split_list = [1, 6, 6, 6, 6, 6, 6, 6, 1]
        for i in range(0, len(split_list), 1):
            st = loc
            en = loc + split_list[i]
            # print(loc, loc + split_list[i])
            b = figure_[st:en]
            a = rest_name[st:en]
            loc = en
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups


def prepare(args, epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    # net = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0:[3,3],2:[3,3]})
    # net = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0:[3,3],2:[5,5]})
    # net = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0:[3,3],2:[7,5]})

    from network_origin import cifar_fast
    net = cifar_fast(input_size=32, n_class=100)
    # print(net)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    """"""

    lr_group = [args['prep'],
                args['layer1_dwconv0_5_5'],
                args['layer1_dwconv1_5_5'],
                args['layer1_dwconv2_5_5'],
                args['layer2_dwconv0_5_5'],
                args['layer3_dwconv0_5_5'],
                args['layer3_dwconv1_5_5'],
                args['layer3_dwconv2_5_5'],
                args['rest']]


    optimizer = torch.optim.SGD(fast_hpo_lr_parameters(net, lr_group),
                                momentum=0.9,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-8, last_epoch=-1)

def random_choice(path_num, m, layers):
    # choice = {}
    import random
    import collections
    import numpy as np
    choice = collections.OrderedDict()
    # choice = {
    #     0: {'conv_0': [0], 'conv_1': [1, 2], 'conv_2': [0,2], 'rate': 0},
    #     2: {'conv_0': [0], 'conv_1': [0, 1], 'conv_2': [1,2], 'rate': 0}}

    # for i in range(len(layers)):
    for i in layers:
        # expansion rate 固定为1
        rate = np.random.randint(low=0, high=1, size=1)[0]
        # conv
        m_ = np.random.randint(low=1, high=(m+1), size=1)[0]
        rand_conv = random.sample(range(path_num), m_)

        m_2 = np.random.randint(low=1, high=(m + 1), size=1)[0]
        rand_conv_2 = random.sample(range(path_num), m_2)

        choice[i] = {'conv_0': [0], 'conv_1': rand_conv, 'conv_2': rand_conv_2,'rate': rate}
    return choice

# Training
def train(epoch, batches=-1):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    scheduler.step()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # naive
        # import numpy as np
        # architecture = [np.random.randint(9) for i in range(2)] # 4+1=5
        # architecture = [np.random.randint(9) for i in range(2)] # 19 wrong 应该是12+1=13
        # outputs = net(inputs, architecture)

        # 其实由于可以multi-path，所以不止81种，但是设置成唯一的single path
        # architecture = {
        #     0: {'conv_0': [0], 'conv_1': [1, 2], 'conv_2': [0, 2], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0, 1], 'conv_2': [1, 2], 'rate': 0}}

        # architecture = {
        #     0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}
        # 暂时用固定 0,1,2,
        # architecture = random_choice(path_num=1, m=1, layers=[0,2])

        get_random_cand = lambda: tuple(np.random.randint(1) for i in range(7))  # 4
        architecture = get_random_cand()

        outputs = net(inputs, architecture)

        # outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if batches > 0 and (batch_idx+1) >= batches:
            return

    return architecture

def test(epoch, architecture):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # naive
            import numpy as np
            # architecture = [np.random.randint(1) for i in range(2)]
            # architecture = [np.random.randint(9) for i in range(2)]
            outputs = net(inputs, architecture)

            # outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100) # 20
    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument("--batches", type=int, default=-1)
    # parser.add_argument("--batchsize", type=int, default=256)
    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        """
        RCV_CONFIG = {
            "prep": 0.00087,
            "layer1_dwconv0_5_5": 0.06077,
            "layer1_dwconv1_5_5": 0.09482,
            "layer1_dwconv2_5_5": 0.01448,
            "layer2_dwconv0_5_5": 0.05309,
            "layer3_dwconv0_5_5": 0.03843,
            "layer3_dwconv1_5_5": 0.00401,
            "layer3_dwconv2_5_5": 0.09642,
            "rest": 0.00063}
        """
        _logger.debug(RCV_CONFIG)
        prepare(RCV_CONFIG, args.epochs)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            architecture = train(epoch, args.batches)
            acc, best_acc = test(epoch, architecture)
            # print(acc, best_acc)
            nni.report_intermediate_result(acc)
        nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
