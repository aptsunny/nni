import torch.nn as nn
import torch.nn.functional as F

class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3)
        # self.conv2 = nn.Conv2d(64, 128, 3)
        # output_size=6, 6*6*128=4608
        # self.fc1 = nn.Linear(4608, 100)
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # output_size=8, 8*8*128=8192
        self.fc1 = nn.Linear(8192, 100)

        """
        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(3, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # output_size=16, 16*16*64=16384
        self.fc1 = nn.Linear(16384, 100)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        
        # out = F.relu(self.conv1_dw_pw(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2_dw_pw(out))
        # out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc1(out)
        return out


class B_Net(nn.Module):
    def __init__(self):
        super(B_Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3)
        # self.conv2 = nn.Conv2d(64, 128, 3)
        # output_size=6, 6*6*128=4608
        # self.fc1 = nn.Linear(4608, 100)
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # output_size=8, 8*8*128=8192
        self.fc1 = nn.Linear(8192, 100)

        """
        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(3, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # output_size=16, 16*16*64=
        self.conv2 = nn.Sequential(
            # dw
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # output_size=8, 8*8*128=8192

        self.fc1 = nn.Linear(8192, 100)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))

        # out = F.relu(self.conv1_dw_pw(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2_dw_pw(out))

        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc1(out)
        return out
if __name__ == '__main__':
    # ------
    import torch
    from torchvision import models
    from torchsummary import summary
    from ptflops import get_model_complexity_info

    # inputs = torch.randn(3, 32, 32).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model = VGG('VGG11').to(device)
    # model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [3, 3], 2: [3, 3]})
    # model = models.vgg11(num_classes=100).to(device)
    # model = models.resnet50(num_classes=100).to(device)

    # ResneSt
    # get list of models
    # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    # load pretrained models, using ResNeSt-50 as an example
    # model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

    # from resnet import ResNet50, ResNet50_3_4_1_8, ResNet50_6_1_6_3
    # model = ResNet50(num_classes=100)
    # model = ResNet50_3_4_1_8(num_classes=100)
    # model = ResNet50_6_1_6_3(num_classes=100)

    model = A_Net()
    print(model)
    model = model.to(device)

    input_size = 32 # 224 224 #

    # part1
    with  torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=False, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # part2
    summary(model, (3, input_size, input_size))
