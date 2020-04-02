"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture

from torchvision import transforms
from torchvision.datasets import CIFAR10

logger = logging.getLogger('mnist_AutoML')

class Net_cifar_2(nn.Module):
    def __init__(self):
        super(Net_cifar_2, self).__init__()
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1),
                                  nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1),
                                  nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        bs = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x0 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x0))

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x).view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_cifar(nn.Module):
    def __init__(self, hidden_size):
        super(Net_cifar, self).__init__()
        # two options of conv1   -> 3 or 5
        self.conv1 = LayerChoice([nn.Conv2d(1, 20, 5, 3),
                                  nn.Conv2d(1, 20, 3, 3)],
                                 key='first_conv')
        # two options of mid_conv
        self.mid_conv = LayerChoice([nn.Conv2d(20, 20, 3, 3, padding=1),
                                     nn.Conv2d(20, 20, 5, 3, padding=2)],
                                    key='mid_conv')
        self.conv2 = nn.Conv2d(20, 50, 5, 3)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        # skip connection over mid_conv -> 设置跳层连接
        self.input_switch = InputChoice(n_candidates=2,
                                        n_chosen=1,
                                        key='skip')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        old_x = x
        x = F.relu(self.mid_conv(x))
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = torch.add(x, skip_x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_dataset(cls):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid

def oneshot_test():
    # this is exactly same as traditional model training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = Net_cifar_2().to(device)


    # model_summary
    sample = torch.zeros((1, 3, 32, 32))
    model_summary(model, sample)
    # dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    # dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

    dataset_train, dataset_valid = get_dataset("cifar10")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

    # use NAS here
    # def top1_accuracy(output, target):
    #     # this is the function that computes the reward, as required by ENAS algorithm
    #     batch_size = target.size(0)
    #     _, predicted = torch.max(output.data, 1)
    #     return (predicted == target).sum().item() / batch_size
    #
    # def metrics_fn(output, target):
    #     # metrics function receives output and target and computes a dict of metrics
    #     from utils import accuracy, reward_accuracy
    #     return {"acc1": reward_accuracy(output, target)}

    def accuracy(output, target):
        batch_size = target.size(0)
        _, predicted = torch.max(output.data, 1)
        return {"acc1": (predicted == target).sum().item() / batch_size}

    # from nni.nas.pytorch import enas
    # trainer = enas.EnasTrainer(model,
    #                            loss=criterion,
    #                            metrics=metrics_fn,
    #                            reward_function=top1_accuracy,
    #                            optimizer=optimizer,
    #                            batch_size=128,
    #                            num_epochs=10,  # 10 epochs
    #                            dataset_train=dataset_train,
    #                            dataset_valid=dataset_valid,
    #                            log_frequency=10)  # print log every 10 steps
    from nni.nas.pytorch.darts import DartsTrainer
    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=accuracy,
                           optimizer=optimizer,
                           num_epochs=2,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=64,
                           log_frequency=10)


    trainer.train()  # training
    trainer.export(file="final_architecture.json")  # export the final architecture to file


###

class Net_mnist(nn.Module):
    def __init__(self, hidden_size):
        super(Net_mnist, self).__init__()
        # two options of conv1   -> 3 or 5
        self.conv1 = LayerChoice([nn.Conv2d(1, 20, 5, 1),
                                  nn.Conv2d(1, 20, 3, 1)],
                                 key='first_conv')
        # two options of mid_conv
        self.mid_conv = LayerChoice([nn.Conv2d(20, 20, 3, 1, padding=1),
                                     nn.Conv2d(20, 20, 5, 1, padding=2)],
                                    key='mid_conv')
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        # skip connection over mid_conv -> 设置跳层连接
        self.input_switch = InputChoice(n_candidates=2,
                                        n_chosen=1,
                                        key='skip')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        old_x = x
        x = F.relu(self.mid_conv(x))
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = torch.add(x, skip_x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #data_dir = os.path.join(args['data_dir'], nni.get_trial_id())
    data_dir = os.path.join(args['data_dir'], 'data')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    hidden_size = args['hidden_size']
    # 定义有自由度的网络结构
    model = Net_mnist(hidden_size=hidden_size).to(device)

    get_and_apply_next_architecture(model)

    # model_summary
    sample = torch.zeros((1, 1, 28, 28))
    model_summary(model, sample)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        if epoch < args['epochs']: # epochs是搜索次数？
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')
        else:
            # report final result
            nni.report_final_result(test_acc)
            logger.debug('Final result is %g', test_acc)
            logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument("--data_dir", type=str,
    #                     default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument("--data_dir", type=str,
                        default='/home/ubuntu/workspace/nni/examples/nas', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')


    args, _ = parser.parse_known_args()
    return args

def download(path):
    from torchvision import datasets
    import torchvision.transforms as transforms
    import urllib

    num_workers = 0
    batch_size = 20
    basepath = path
    transform = transforms.ToTensor()

    def set_header_for(url, filename):
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')
        opener.retrieve(
        url, f'{basepath}/{filename}')

    set_header_for('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
    set_header_for('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    set_header_for('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    set_header_for('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    train_data = datasets.MNIST(root='data', train=True,
                                       download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                                      download=False, transform=transform)




def random_api():
    nums = range(5,10)
    gen_index = torch.randint(high=5,size=(1,))
    print(1, gen_index)
    w = F.one_hot(gen_index, num_classes=5).view(-1).bool()
    print(2, w)

    gen_index = torch.randint(high=2, size=(8,)).view(-1)
    print(3, gen_index)

    gen_index = torch.randint(high=2, size=(8,)).view(-1).bool()
    print(4, gen_index)

    perm = torch.randperm(12)
    print(5, perm)
    mask = [i in perm[:7] for i in range(12)]
    print(6, torch.tensor(mask, dtype=torch.bool))
    return

def model_summary(model, sample):
    from torchsummaryX import summary
    # get_and_apply_next_architecture(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # model_Net1 = model.to(device)
    # use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # sample = torch.zeros((1, 1, 28, 28))
    # test_data = torch.zeros((1, 1, 28, 28))
    test_data = sample.to(device)
    # test_data = test_data.to(device)
    summary(model, test_data)
    # summary(model, torch.zeros((1, 1, 28, 28)))
    return

def pytorch_tensor_info():
    import numpy as np
    # torch.tensor default float32
    my_tensor = torch.tensor([[0.0, 1.0],[0.1, 0.2]])
    print(my_tensor)
    # new_tensor = my_tensor.int().float()
    # numpy default float64
    np_tensor = np.array([[0.0, 1.0],[0.1, 0.2]])
    tensor_from_np = torch.tensor(np_tensor)
    to_numpy = tensor_from_np.numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_tensor = my_tensor.to(device=device)
    print(my_tensor)

    single_number = torch.tensor([0])
    print(single_number.item())

    tensor_with_gradient = torch.tensor([[0.0, 1.0],[0.1, 0.2]], requires_grad=True)
    result = tensor_with_gradient.pow(2).sum()
    result.backward()
    print(tensor_with_gradient.grad)

    # make sure the tensor never need gradient
    tensor_with_gradient.detach_() # inplace function
    tensor_with_gradient = tensor_with_gradient.detach() #


    # operation broadcast
    y = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
    y = y + 1
    print(y)

    y = y.unsqueeze(0) # 维度扩增
    print(y, y.shape)

    y = y.squeeze() # 维度为1 缩减
    print(y)

    x = torch.randn(5)
    y = torch.randn(5)
    print(torch.einsum('i,j->ij', x, y))

    A = torch.randn(3, 5, 4)
    l = torch.randn(2, 5)
    r = torch.randn(2, 4)
    print(torch.einsum('bn,anm,bm->ba', l, A, r))

    return

def pytorch_dataloader():
    from torch.utils.data import Dataset,DataLoader

    class MyDataset(Dataset):
        def __init__(self):
            super(MyDataset, self).__init__()
            self.data = torch.randn([1024, 10, 10])
        def __len__(self):
            return 1024
        def __getitem__(self, idx):
            return self.data[idx, :, :]
    mydata = MyDataset()
    mydata_loader = DataLoader(mydata, batch_size=64, shuffle=True)
    # for i in mydata_loader:
    #     print(i)

    class MyDictDataset(Dataset):
        def __init__(self):
            super(MyDictDataset, self).__init__()
            self.x = torch.randn([32, 10])
            self.y = torch.randn([32])
        def __len__(self):
            return 32
        def __getitem__(self, idx):
            return {'x':self.x[idx, :],'y':self.y[idx]}

    mydictdata = MyDictDataset()
    mydictdata_loader = DataLoader(mydictdata, batch_size=8, shuffle=True)

    # for batch in mydictdata_loader:
    #     print(batch['x'])
    #     print(batch['y'])

    from torch.utils.data import TensorDataset
    x = torch.randn(10,100)
    y = torch.randn(10)
    tensor_dataset = TensorDataset(x, y)

    # for i in tensor_dataset:
    #     print(i)

    return

def pytorch_network():
    x = torch.rand([8, 100, 10]).detach()
    print(x)
    y = torch.rand(8)
    y = (y>0.5).int()
    print(y)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.first_layer = nn.Linear(1000, 50)
            self.second_layer = nn.Linear(50,1)
        def forward(self, x):
            x = torch.flatten(x, start_dim=1, end_dim=2) # 100*10 拉直
            x = nn.functional.relu(self.first_layer(x))
            x = self.second_layer(x)
            return x
    mlp = MLP()
    output = mlp(x)
    print(output)

    class Embedding(nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.embedding = nn.Embedding(4,100)
        def forward(self, x):
            return self.embedding(x)
    embedding = Embedding()
    embedding_input = torch.tensor([[0, 1, 0],[2, 3, 3]])
    embedding_output = embedding(embedding_input)

    print(embedding_output.shape)

    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10,
                                15,
                                num_layers=2,
                                bidirectional=True, # 文本方向指定
                                dropout=0.1)
        def forward(self, x):
            output, (hidden, cell) = self.lstm(x)
            return output, hidden, cell

    permute_x = x.permute([1, 0, 2]) # 第一维和第二维交换
    lstm = LSTM()
    output_lstm1, output_lstm2, output_lstm3 = lstm(permute_x)
    print(output_lstm2.shape) # 100, 8(batchsize), 30(15*2)


    class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()
            self.convld = nn.Conv1d(100, 50, 2)
        def forward(self, x):
            return self.convld(x)
    conv = Conv()
    output = conv(x)
    print(output.shape)

    return

if __name__ == '__main__':
    # part0 torch
    # pytorch_tensor_info()
    # pytorch_dataloader()
    pytorch_network()

    # part1 mnist example
    # path = './data/'
    # download(path)

    # try:
    #     params = vars(get_params())
    #     main(params)
    # except Exception as exception:
    #     logger.exception(exception)
    #     raise

    # params = vars(get_params())
    # main(params)

    # part2
    # test_api
    # random_api()

    # part3 test_trainer
    # oneshot_test()


