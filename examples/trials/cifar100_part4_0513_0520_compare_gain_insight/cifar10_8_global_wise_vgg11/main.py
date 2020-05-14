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


def layer_wise_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 83:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            b = figure_[i:i+3]
            figure_choice.append(b)
            choice.append(a)

    elif len(rest_name) == 66:
        for i in range(0, len(rest_name), 2):
            a = rest_name[i:i + 2]
            b = figure_[i:i+2]
            figure_choice.append(b)
            choice.append(a)

    elif len(rest_name) == 34:# vgg11
        for i in range(0, len(rest_name), 4):
            a = rest_name[i:i + 4]
            b = figure_[i:i+4]
            figure_choice.append(b)
            choice.append(a)

    elif len(rest_name) == 62:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            b = figure_[i:i+3]
            figure_choice.append(b)
            choice.append(a)

    elif len(rest_name) == 149:
        for i in range(0, len(rest_name), 3):
            a = rest_name[i:i + 3]
            b = figure_[i:i+3]
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups


def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

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

    # net = MobileNet(num_classes=100)

    # "model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"],

    if args['model'] == 'mobilenet':
        net = MobileNet(num_classes=100)  # 83 83-2/3= 27 28   layers: (3)+(13)*6+(2)=15
        # lr_group = [0.01] * 7 + [0.001] * 7 + [0.0001] * 7 + [0.00001] * 7
        lr_group = [args['lr_01']] * 1 + [args['lr_02']] * 2 + [args['lr_03']] * 2 + [args['lr_04']] * 2 + [
            args['lr_05']] * 2 + [args['lr_06']] * 2 + [args['lr_07']] * 2 + [args['lr_08']] * 2 + [
                       args['lr_09']] * 2 + [args['lr_10']] * 2 + [args['lr_11']] * 2 + [args['lr_12']] * 2 + [
                       args['lr_13']] * 2 + [args['lr_14']] * 2 + [args['lr_15']] * 1

    if args['model'] == 'vgg':
        # net = VGG('VGG19')  # 66 66/2=33
        # 16 expect M 16*4=64 + 2 =66  layers:2,2,4,4,4, + 2 = 6
        # lr_group = [0.01] * 7 + [0.001] * 7 + [0.0001] * 7 + [0.00001] * 12
        # lr_group = [args['lr_01']] * 4 + [args['lr_02']] * 4 + [args['lr_03']] * 8 + [args['lr_04']] * 8 + [args['lr_05']] * 8 + [args['lr_06']] * 1

        net = VGG('VGG11')  #  34 34/2=17
        # 8 expect M 8*4=32 + 2 =34  layers:1,1,2,2,2 + 2 = 6
        # lr_group = [0.01] * 7 + [0.001] * 7 + [0.0001] * 7 + [0.00001] * 12
        # lr_group = [args['lr_00']] * 1 + [args['lr_01']] * 1 + [args['lr_02']] * 1 + [args['lr_03']] * 1 + [args['lr_04']] * 1 + [args['lr_05']] * 1 + [args['lr_06']] * 1 + [args['lr_07']] * 1 + [args['lr_08']] * 1

    if args['model'] == 'resnet18':
        net = ResNet18(num_classes=100)  # 62 21   layers:3+(12,15,15,15)+2
        # lr_group = [0.01] * 7 + [0.001] * 7 + [0.0001] * 7
        lr_group = [args['lr_01']] * 1 + [args['lr_02']] * 4 + [args['lr_03']] * 5 + [args['lr_04']] * 5 + [
            args['lr_05']] * 5 + [args['lr_06']] * 1

    if args['model'] == 'shufflenetg2':
        net = ShuffleNetG2(num_classes=100)  # 149 147//3+1=49+1=50 # layers:3+(9+9+9+9)+(9*8)+(36)+2 = 5
        # lr_group = [0.01] * 12 + [0.001] * 12 + [0.0001] * 12 + [0.00001] * 14
        lr_group = [args['lr_01']] * 1 + [args['lr_02']] * 12 + [args['lr_03']] * 24 + [args['lr_04']] * 12 + [
            args['lr_05']] * 1

    if args['model'] == 'googlenet':
        net = GoogLeNet(num_classes=100)
    if args['model'] == 'densenet121':
        net = DenseNet121(num_classes=100)
    if args['model'] == 'dpn92':
        net = DPN92(num_classes=100)
    if args['model'] == 'senet18':
        net = SENet18(num_classes=100)


    print(net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()


    # lr_group = [args['prep'],
    #             args['layer1_conv0_3_3'],
    #             args['layer1_conv1_7_7'],
    #             args['layer1_conv2_3_3'],
    #             args['layer2_conv0_3_3'],
    #             args['layer3_conv0_3_3'],
    #             # args['layer3_conv1_3_3'],
    #             args['layer3_conv2_5_5'],
    #             args['rest']]

    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

    # optimizer = torch.optim.SGD(layer_wise_parameters(net, lr_group),
    #                             momentum=0.9,
    #                             weight_decay=5e-4)

    # if args['optimizer'] == 'SGD':
    #     optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    # if args['optimizer'] == 'Adadelta':
    #     optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adagrad':
    #     optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adam':
    #     optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adamax':
    #     optimizer = optim.Adam(net.parameters(), lr=args['lr'])


# Training
def train(epoch, batches=-1):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.size(), targets.size())
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
            outputs = net(inputs)
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
    parser.add_argument("--epochs", type=int, default=100)

    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument("--batches", type=int, default=-1)

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()

        RCV_CONFIG = {'lr': 0.001, 'model': 'vgg'}
        # RCV_CONFIG = {
        #     'model': 'vgg',
        #     'lr_00': 0.002,
        #     'lr_01': 0.1,
        #     'lr_02': 0.05,
        #     'lr_03': 0.025,
        #     'lr_04': 0.0125,
        #     'lr_05': 0.00675,
        #     'lr_06': 0.002,
        #     'lr_07': 0.002,
        #     'lr_08': 0.002}

        # RCV_CONFIG = {'lr': 0.001, 'optimizer': 'SGD', 'model':'mobilenet'}
        # RCV_CONFIG = {'lr': 0.001, 'optimizer': 'SGD', 'model':'vgg'}
        # RCV_CONFIG = {'lr': 0.001, 'optimizer': 'SGD', 'model':'resnet18'}
        # RCV_CONFIG = {'lr': 0.001, 'optimizer': 'SGD', 'model':'shufflenetg2'}
        _logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(epoch, args.batches)
            acc, best_acc = test(epoch)
            print(acc, best_acc)

            # nni.report_intermediate_result(acc)
        # nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
