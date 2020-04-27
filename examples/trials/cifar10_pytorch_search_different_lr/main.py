'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import argparse
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

def fast_9_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    for name, param in model.named_parameters():
        if name.find('layer') > 0:
            if name.find('layer1_features.0') >0 or name.find('layer3_features.0') >0:
                # print('1:',name)
                rest_name.append(param)
        else:
            rest_name.append(param)
            # print('2:', name)

    choice = []
    if len(rest_name)==25:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i+3]
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

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    # print('==> Building model..')
    # if args['model'] == 'vgg':
    #     net = VGG('VGG19')
    # if args['model'] == 'resnet18':
    #     net = ResNet18()
    # if args['model'] == 'googlenet':
    #     net = GoogLeNet()
    # if args['model'] == 'densenet121':
    #     net = DenseNet121()
    # if args['model'] == 'mobilenet':
    #     net = MobileNet()
    # if args['model'] == 'dpn92':
    #     net = DPN92()
    # if args['model'] == 'shufflenetg2':
    #     net = ShuffleNetG2()
    # if args['model'] == 'senet18':
    #     net = SENet18()
    # if args['model'] == 'naive_cifar':
    #     net = Network_cifar(num_classes=100)

    net = Network_cifar(num_classes=100)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

    # part1
    # "lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    lr_group = [args['lr_01'],
                args['lr_02'],
                args['lr_03'],
                args['lr_04'],
                args['lr_05'],
                args['lr_06'],
                args['lr_07'],
                args['lr_08'],
                args['lr_09'],]

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(fast_9_lr_parameters(net, lr_group), momentum=0.9, weight_decay=5e-4)
    if args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(fast_9_lr_parameters(net, lr_group))
    if args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(fast_9_lr_parameters(net, lr_group))
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(fast_9_lr_parameters(net, lr_group))
    if args['optimizer'] == 'Adamax':
        optimizer = optim.Adam(fast_9_lr_parameters(net, lr_group))


    if args['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-8, last_epoch=-1)
    if args['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    if args['scheduler'] == 'LambdaLR':
        import numpy as np
        lambda1 = lambda epoch: np.sin(epoch) / epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # optimizer = optim.SGD(net.parameters(), lr=args['base_lr'], momentum=0.9, weight_decay=5e-4)

    # part2
    # import numpy as np
    # nums_lr_group = 19  # 9 * 2 + 1
    # lr_l, lr_r = float(args['lr_l']), float(args['lr_r'])
    # lr_group = list(np.random.uniform(lr_l, lr_r) for i in range(nums_lr_group))
    # "lr_l":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_r":{"_type": "loguniform", "_value": [0.01, 0.1]},

    # # part3
    # "lr_01":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_02":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_03":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_04":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_05":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_06":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_07":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_08":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_09":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_10":{"_type": "loguniform", "_value": [0.001, 0.01]},
    #
    # "lr_11":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_12":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_13":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_14":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_15":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_16":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_17":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_18":{"_type": "loguniform", "_value": [0.001, 0.01]},
    # "lr_19":{"_type": "loguniform", "_value": [0.001, 0.01]},

    # "lr_01":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_02":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_03":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_04":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_05":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_06":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_07":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_08":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_09":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_10":{"_type": "uniform", "_value": [0.0001, 0.1],
    #
    # "lr_11":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_12":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_13":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_14":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_15":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_16":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_17":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_18":{"_type": "uniform", "_value": [0.0001, 0.1],
    # "lr_19":{"_type": "uniform", "_value": [0.0001, 0.1],

    # "lr_01":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_02":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_03":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_04":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_05":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_06":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_07":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_08":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_09":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_10":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_11":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_12":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_13":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_14":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_15":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_16":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_17":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_18":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "lr_19":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "model":{"_type":"choice", "_value":["naive_cifar"]}


    # lr_group = [args['lr_01'],
    #             args['lr_02'],
    #             args['lr_03'],
    #             args['lr_04'],
    #             args['lr_05'],
    #             args['lr_06'],
    #             args['lr_07'],
    #             args['lr_08'],
    #             args['lr_09'],
    #             args['lr_10'],
    #
    #             args['lr_11'],
    #             args['lr_12'],
    #             args['lr_13'],
    #             args['lr_14'],
    #             args['lr_15'],
    #             args['lr_16'],
    #             args['lr_17'],
    #             args['lr_18'],
    #             args['lr_19'],]

    # optimizer = torch.optim.SGD(fast_19_lr_parameters(net, lr_group),
    #                             momentum=0.9,
    #                             weight_decay=5e-4)

    # part4

    # "layer1_conv1_3_3":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "layer1_conv2_3_3":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "layer3_conv1_3_3":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "layer3_conv2_3_3":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    # "base_lr":{"_type": "loguniform", "_value": [0.001, 0.01]},
    #     "base_lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},


    # lr_group = [args['layer1_conv1_3_3'],
    #             args['layer1_conv2_3_3'],
    #             args['layer3_conv1_3_3'],
    #             args['layer3_conv2_3_3'],
    #             args['base_lr']]

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
        import numpy as np
        # 3*3
        architecture = [np.random.randint(1) for i in range(2)] # 4+1=5
        # architecture = [np.random.randint(9) for i in range(2)] # 19 wrong 应该是12+1=13

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

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | LR = %.4f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), scheduler.get_lr()[0], 100.*correct/total, correct, total))

        if batches > 0 and (batch_idx+1) >= batches:
            return

def test(epoch):
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
            architecture = [np.random.randint(1) for i in range(2)]
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
        #RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}

        # RCV_CONFIG = {'lr_01': 0.1, 'lr_02': 0.1, 'lr_03': 0.1, 'lr_04': 0.1, 'lr_05': 0.1, 'lr_06': 0.1, 'lr_07': 0.1, 'lr_08': 0.1, 'lr_09': 0.1, 'optimizer': 'SGD', 'scheduler':'CosineAnnealingLR'}
        _logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG, args.epochs)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(epoch, args.batches)
            acc, best_acc = test(epoch)

            # print(acc)
            nni.report_intermediate_result(acc)

        nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
