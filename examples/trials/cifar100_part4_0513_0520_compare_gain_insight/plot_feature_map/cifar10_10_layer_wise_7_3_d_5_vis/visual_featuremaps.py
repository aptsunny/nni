# coding: utf-8
import os
import torch
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from torch.utils.data import DataLoader
import torchvision
from models import *
# from utils.utils import MyDataset, Net, normalize_invert
from torch.utils.data import Sampler
# import matplotlib
from PIL import Image


hook_list = []
def hook(module, input, output):
    hook_list.append(input[0])
    hook_list.append(output)
    print(len(hook_list), input[0].size(), output.size())

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

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

def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def load_parameters(epoch):
    # pretrained_path = os.path.join("./checkpoint/18_net_params.pkl")
    pretrained_path = os.path.join('/home/ubuntu/workspace/nni_sy/examples/trials/pytorch_tensorboard_Analysis/cifar10_10_layer_wise_7_3_d_5_vis/checkpoint/{}_net_params.pkl'.format(epoch))
    net = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [7, 3], 2: ['d', 5]})
    # print(net)
    net = net.to(device)
    pretrained_dict = torch.load(pretrained_path)
    for key in list(pretrained_dict):
        pretrained_dict[key[7:]] = pretrained_dict[key]
        del pretrained_dict[key]
    net.load_state_dict(pretrained_dict)
    return net

def load_image(normMean, normStd):
    normTransform = transforms.Normalize(normMean, normStd)
    testTransform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normTransform
    ])

    # testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=testTransform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    # return testloader

    from torch.utils.data import SubsetRandomSampler
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

    # from dataset_cifar import DataIterator, SubsetSampler
    split = 0.0002 # 50000*0.002= 100
    split_idx = 0
    # train_sampler = None
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=testTransform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=testTransform)

    if split > 0.0:
        # 分层至少100
        # XXX = StratifiedShuffleSplit(n_splits=10, test_size=split, random_state=0)
        XXX = ShuffleSplit(n_splits=1, test_size=0.0002)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)train_test_split()
        a = list(range(len(trainset)))
        sss = XXX.split(list(range(len(trainset))), trainset.targets)

        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
        # train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
        sampler=valid_sampler, drop_last=False)
    return testloader

def record_feature(m):
    # for name, param in m.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    print('>>', m)
    return

def counter(net):
    from flops_counter_feature import get_model_complexity_info
    # inputs = torch.randn(3, 32, 32).unsqueeze(0)
    model = net
    with torch.cuda.device(0):
        # put choice into {forward}
        # choice = {
        #     0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    Flops_counter = False # True
    Record_feature = False # True # # some problem
    Write_forward = False # forward
    Write_layer = False


    device = 'cpu'
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    testloader = load_image(normMean, normStd)
    architecture = random_choice(path_num=1, m=1, layers=[0, 2])

    # if not os.path.isdir('./feature_map/'):
    #     os.mkdir('./feature_map/')
    if not os.path.isdir('./feature_map/images_folder/'):
        os.mkdir('./feature_map/')
        os.mkdir('./feature_map/images_folder/')

    writer = SummaryWriter(log_dir='./feature_map/')
    vis_layer = 'avgpool'
    for i in range(1,2):
        epoch=str(i)
        net = load_parameters(epoch)
        if Flops_counter:
            counter(net)
        elif Record_feature:
            net.apply(record_feature)
        elif Write_forward:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                img, label = inputs.to(device), targets.to(device)
                # According to forward
                # Block[0] [1,64,32,32] -> [1,128,16,16]
                # Block[1] [1,128,16,16] -> [1,256,8,8]
                # print(x.size())# input
                x = img
                for name, layer in net._modules.items():
                    # print(name)
                    if name == 'stem':
                        x = layer(x)
                        # 由于__init__()相较于forward()缺少relu操作，需要手动增加
                        # x = F.relu(x) if 'conv' in name else x
                        # 依据选择的层，进行记录feature maps
                        vis_layer = name
                        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                        print('stem:', x1.size())
                        # img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=8)  # B，C, H, W
                        # writer.add_image('epoch{}_images{}_layers_{}_feature_maps'.format(epoch, batch_idx, vis_layer), img_grid, global_step=666)
                    elif name == 'Block':
                        x = layer[0](x, architecture[0])
                        # print(x.size()) # [1, 128, 16, 16]
                        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                        print('Block[0]:', x1.size())

                        x = layer[1](x)
                        # print(x.size()) # [1, 256, 8, 8]
                        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                        print('Block[1]:', x1.size())

                        x = layer[2](x, architecture[2])
                        # print(x.size()) #  [1, 512, 4, 4]
                        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                        print('Block[2]:', x1.size())
                        vis_layer = name
                    # elif name == 'avgpool':
                    #     x = layer(x)
                    #     x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                    #     print('avgpool:', x1.size())
                    #     vis_layer = name

                    # else:
                    #     # x = x.view(x.size(0), -1) if "fc" in name else x # 为fc层预处理x
                    #     x = x.view(x.size(0), -1) if "classifier" in name else x
                    #     print(x.size())
                    #     x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                    #     print('classifier:', x1.size())
                    #     vis_layer = name

                    # img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=8)  # B，C, H, W
                    img_grid = vutils.make_grid(x1, normalize=True, scale_each=True)  # B，C, H, W
                    writer.add_image('epoch{}_images{}_layers_{}_feature_maps'.format(epoch, batch_idx, vis_layer), img_grid, global_step=666)
        elif Write_layer: # According to specific layer
            # cifar100: 100images
            for batch_idx, (inputs, targets) in enumerate(testloader):
                img, label = inputs.to(device), targets.to(device)
                x = img
                if i==1:
                    # write img_raw
                    img_raw = normalize_invert(img, normMean, normStd)
                    img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
                    writer.add_image('0_images{}_input_raw_img'.format(batch_idx), img_raw, global_step=666)

                    img_raw = img_raw.transpose(2,1,0)
                    # import cv2
                    # import numpy as np
                    # cv2.imwrite('./feature_map/images_folder/images_{}.png'.format(batch_idx), img_raw)
                    # img_raw_2 = np.array(img_raw * 255).clip(0, 255).squeeze()
                    im = Image.fromarray(img_raw)

                    im.save('/home/ubuntu/workspace/nni_sy/examples/trials/pytorch_tensorboard_Analysis/cifar10_10_layer_wise_7_3_d_5_vis/feature_map/images_folder/images_{}.png'.format(batch_idx))
                    # im.save('./feature_map/images_folder/images_{}.png'.format(batch_idx))
                    # matplotlib.image.imsave('./feature_map/images_folder/images_{}.png'.format(batch_idx), img_raw)

                # record
                # layer1 bottleneck
                # handle = net.Block[0].conv_first[0][0].register_forward_hook(hook) # conv2d [1,64,32,32] -> [1,128,32,32]
                # handle = net.Block[0].conv_first[0][1].register_forward_hook(hook) # BatchNorm2d [1,64,32,32] -> [1,128,32,32]
                # handle = net.Block[0].conv_first[0][2].register_forward_hook(hook) # ReLU [1,64,32,32] -> [1,128,32,32]
                # handle = net.Block[0].pool.register_forward_hook(hook) # [1, 128, 32, 32] -> [1, 128, 16, 16])
                # handle = net.Block[0].mix_conv[0][0].register_forward_hook(hook) # [1,128,16,16] -> [1,128,16,16]
                # handle = net.Block[0].mix_conv_2[0][0].register_forward_hook(hook) # [1,128,16,16] -> [1,128,16,16]

                # layer2 Block[1] Sequential
                # handle = net.Block[1][0].register_forward_hook(hook) # conv2d
                # handle = net.Block[1][1].register_forward_hook(hook) # BatchNorm2d
                # handle = net.Block[1][2].register_forward_hook(hook) ReLU [1, 256, 16, 16]
                # handle = net.Block[1][3].register_forward_hook(hook) # MaxPool2d

                # layer3
                # handle = net.Block[2].conv_first[0][0].register_forward_hook(hook)  # [1, 256, 8, 8]) [1, 512, 8, 8])
                # handle = net.Block[2].pool.register_forward_hook(hook)
                # handle = net.Block[2].mix_conv[0][0].register_forward_hook(hook)  # 这个位置是 relu [1, 512, 4, 4] [1, 512, 4, 4])
                # handle = net.Block[2].mix_conv_2[0][0].register_forward_hook(hook)

                # avgpool
                # handle = net.avgpool.register_forward_hook(hook)  # conv的input维度
                # output = net(img)
                # print(output.size())
                # vvv = hook_list
                # print(hook_list)
                # handle.remove()

                print('images{}_epoch{}_before_{}_feature_maps'.format(batch_idx, epoch, vis_layer))

                if vis_layer == 'stem':
                    handle = net.stem.register_forward_hook(hook)
                    output = net(img) # print(output.size())
                    handle.remove()

                elif vis_layer == 'layer1_bottleneck': # ([1, 64, 32, 32]) torch.Size([1, 128, 16, 16])
                    handle = net.Block[0].register_forward_hook(hook)
                    output = net(img) # print(output.size())
                    handle.remove()

                elif vis_layer == 'layer2_bottleneck':  # [1, 128, 16, 16]) torch.Size([1, 256, 8, 8]
                    handle = net.Block[1].register_forward_hook(hook)
                    output = net(img)  # print(output.size())
                    handle.remove()

                elif vis_layer == 'layer3_bottleneck':  # [1, 256, 8, 8]) torch.Size([1, 512, 4, 4]
                    handle = net.Block[2].register_forward_hook(hook)
                    output = net(img)  # print(output.size())
                    handle.remove()

                elif vis_layer == 'avgpool':  # [1, 512, 4, 4]) torch.Size([1, 512, 1, 1]
                    handle = net.avgpool.register_forward_hook(hook)
                    output = net(img)  # print(output.size())
                    handle.remove()

                elif vis_layer == 'layer2_MaxPool2d':
                    handle = net.Block[1][3].register_forward_hook(hook)
                    output = net(img) # print(output.size())
                    handle.remove()

                hook_list[batch_idx * 2] = torch.mean(hook_list[batch_idx*2], dim=1, keepdim=False, out=None)
                hook_list[batch_idx * 2 + 1] = torch.mean(hook_list[batch_idx*2+1], dim=1, keepdim=False, out=None)

                hook_list[batch_idx * 2] = hook_list[batch_idx * 2].unsqueeze(0)
                hook_list[batch_idx * 2 + 1] = hook_list[batch_idx * 2 + 1].unsqueeze(0)

                # hook_list[batch_idx * 2] = torch.cat((hook_list[batch_idx * 2], hook_list[batch_idx * 2]), 0)
                # hook_list[batch_idx * 2 + 1] = torch.cat((hook_list[batch_idx * 2 + 1], hook_list[batch_idx * 2 + 1]), 0)
                # hook_list[batch_idx * 2] = torch.stack([hook_list[batch_idx * 2], hook_list[batch_idx * 2]], 1)

                # print(i) # 4epoch ,100images # 0-399
                # conv_before vs conv_after
                img_grid_before = vutils.make_grid(hook_list[batch_idx*2], normalize=True, scale_each=True)  # B，C, H, W
                writer.add_image('images{}_epoch{}_before_{}_feature_maps'.format(batch_idx, epoch, vis_layer), img_grid_before, global_step=666)

                # img_grid_after = vutils.make_grid(hook_list[batch_idx*2+1], normalize=True, scale_each=True)  # B，C, H, W
                # writer.add_image('images{}_epoch{}_after_{}_feature_maps'.format(batch_idx, epoch, vis_layer), img_grid_after, global_step=666)

        for batch_idx, (inputs, targets) in enumerate(testloader):
            print(batch_idx, inputs.size(), targets)


    writer.close()

# img_grid_before = vutils.make_grid(hook_list[batch_idx*2].transpose(0, 1), normalize=True, scale_each=True)  # B，C, H, W
# writer.add_image('epoch{}_images{}_before_{}_feature_maps'.format(epoch, batch_idx, vis_layer), img_grid_before, global_step=666)

# img_grid_after = vutils.make_grid(hook_list[batch_idx*2+1].transpose(0, 1), normalize=True, scale_each=True)  # B，C, H, W
# writer.add_image('epoch{}_images{}_after_{}_feature_maps'.format(epoch, batch_idx, vis_layer), img_grid_after, global_step=666)


# x = self.stem(x)
# x = self.Block[0](x, choice[0]) # layer1
# x = self.Block[1](x)
# x = self.Block[2](x, choice[2]) # layer3
# x = self.avgpool(x)
# x = x.view(-1, last_channel)
# x = self.classifier(x)

# architecture = random_choice(path_num=1, m=1, layers=[0,2])
# outputs = net(inputs, architecture)

# 对x执行单层运算


# 获得模型的键值
# keys = []
# for k, v in desnet.state_dict().items():
#     if v.shape:
#         keys.append(k)
#     print(k, v.shape)
#
# # 从预训练文件中加载权重
# state = {}
# pretrained_dict = torch.load('/home/lulu/pytorch/Paper_Code/weights/densenet121-a639ec97.pth')
# for i, (k, v) in enumerate(pretrained_dict.items()):
#     if 'classifier' not in k:
#         state[keys[i]] = v