import math
import torch
import torch.nn as nn

# Bottleneck -> SuperNetwork_3
class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, shadow_bn, kernel=[3,3], activation=nn.ReLU):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn
        # self.stride = stride

        # multi-path
        # self.kernel_list_1 = [3, 5, 7, 9]
        self.kernel_list_0 = [3]
        # self.kernel_list_1 = [3, 5, 7]
        # self.kernel_list_2 = [3, 5, 7]
        self.kernel_list_1 = [kernel[0]]
        self.kernel_list_2 = [kernel[1]]

        # channel expansion # default=1
        # self.expansion_rate = [3, 6]
        self.expansion_rate = [1]

        self.activation = activation(inplace=True)
        self.conv_first = nn.ModuleList([])

        # multi-path
        self.mix_conv = nn.ModuleList([])
        self.mix_conv_2 = nn.ModuleList([])

        self.pool = nn.MaxPool2d(2)

        # build the layer
        for t in self.expansion_rate:
            # conv_first
            for j in self.kernel_list_0:
                self.conv_first.append(nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, kernel_size=j, padding=j // 2, bias=False),
                    nn.BatchNorm2d(outplanes),
                    activation(inplace=True)
                ))

            # mix_conv
            conv_list = nn.ModuleList([])
            for j in self.kernel_list_1:
                conv_list.append(nn.Sequential(
                    nn.Conv2d(outplanes, outplanes, kernel_size=j, padding=j // 2, bias=False),
                    nn.BatchNorm2d(outplanes),
                    activation(inplace=True)
                ))
            self.mix_conv.append(conv_list)
            del conv_list

            # conv_end
            conv_end = nn.ModuleList([])
            for j in self.kernel_list_2:
                conv_end.append(nn.Sequential(
                    nn.Conv2d(outplanes, outplanes, kernel_size=j, padding=j // 2, bias=False),
                    nn.BatchNorm2d(outplanes),
                    activation(inplace=True)
                ))
            self.mix_conv_2.append(conv_end)
            del conv_end


    def forward(self, x, choice):
        # choice: {'conv', 'rate'} sample path
        conv_ids_0 = choice['conv_0']# conv_ids, e.g. [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]
        conv_ids_1 = choice['conv_1']
        conv_ids_2 = choice['conv_2']
        m_ = len(conv_ids_1)  # num of selected paths
        m_2 = len(conv_ids_2)  # num of selected paths
        rate_id = choice['rate']  # rate_ids, e.g. 0, 1
        assert m_ in [1, 2, 3] # [1, 2, 3, 4]
        assert m_2 in [1, 2, 3] # [1, 2, 3, 4]
        assert rate_id in [0, 1]

        # conv_first
        x = self.conv_first[rate_id][conv_ids_0[0]](x)
        x = self.pool(x)

        residual = x
        # mix_conv
        if m_ == 1: # single path
            out = self.mix_conv[rate_id][conv_ids_1[0]](x)
        else: # multi path
            temp = []
            for id in conv_ids_1:
                temp.append(self.mix_conv[rate_id][id](x))
            out = sum(temp)

        # mix_conv_2
        if m_2 == 1: # single path
            out = self.mix_conv_2[rate_id][conv_ids_2[0]](out)
        else: # multi path
            temp = []
            for id in conv_ids_1:
                temp.append(self.mix_conv_2[rate_id][id](out))
            out = sum(temp)

        # out = self.conv_end[rate_id][conv_ids_2[0]](out)

        # residual
        # if self.stride == 1 and self.inplanes == self.outplanes:
        out = out + residual
        return out

# origin
class Inverted_Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, shadow_bn, stride, activation=nn.ReLU6):
        super(Inverted_Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn
        self.stride = stride

        # self.kernel_list = [3, 5, 7, 9]
        self.kernel_list = [9]

        self.expansion_rate = [3]
        # self.expansion_rate = [3, 6]

        self.activation = activation(inplace=True)

        #
        self.pw = nn.ModuleList([])
        self.mix_conv = nn.ModuleList([])
        self.mix_bn = nn.ModuleList([])
        self.pw_linear = nn.ModuleList([])

        for t in self.expansion_rate:
            # pw
            self.pw.append(nn.Sequential(
                nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False),
                nn.BatchNorm2d(inplanes * t),
                activation(inplace=True)
            ))
            # dw
            conv_list = nn.ModuleList([])
            for j in self.kernel_list:
                conv_list.append(nn.Sequential(
                    nn.Conv2d(inplanes * t, inplanes * t, kernel_size=j, stride=stride, padding=j // 2,
                              bias=False, groups=inplanes * t),
                    nn.BatchNorm2d(inplanes * t),
                    activation(inplace=True)
                ))

            self.mix_conv.append(conv_list)
            del conv_list
            # pw
            self.pw_linear.append(nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False))

            # bn
            bn_list = nn.ModuleList([])
            if self.shadow_bn:
                for j in range(len(self.kernel_list)):
                    bn_list.append(nn.BatchNorm2d(outplanes))
                self.mix_bn.append(bn_list)
            else:
                self.mix_bn.append(nn.BatchNorm2d(outplanes))
            del bn_list

    def forward(self, x, choice):
        # choice: {'conv', 'rate'} sample path
        conv_ids = choice['conv']  # conv_ids, e.g. [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]
        m_ = len(conv_ids)  # num of selected paths
        rate_id = choice['rate']  # rate_ids, e.g. 0, 1
        assert m_ in [1, 2, 3, 4]
        assert rate_id in [0, 1]
        residual = x
        # pw
        out = self.pw[rate_id](x)
        # dw
        if m_ == 1: # single path
            out = self.mix_conv[rate_id][conv_ids[0]](out)
        else: # multi path
            temp = []
            for id in conv_ids:
                temp.append(self.mix_conv[rate_id][id](out))
            out = sum(temp) # sum
        # pw
        out = self.pw_linear[rate_id](out)

        if self.shadow_bn:
            out = self.mix_bn[rate_id][m_ - 1](out)
        else:
            out = self.mix_bn[rate_id](out)

        # residual
        if self.stride == 1 and self.inplanes == self.outplanes:
            out = out + residual
        return out

# channel = [32, 48, 48, 96, 96, 96, 192, 192, 192, 256, 256, 320, 320]
# last_channel = 1280

# channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
channel = [64, 128, 256, 512]
last_channel = 512

class SuperNetwork_3(nn.Module):
    def __init__(self, shadow_bn, layers=12, classes=10, sample={}):
        super(SuperNetwork_3, self).__init__()
        self.layers = layers

        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )

        # 这里的layer是stage
        self.Block = nn.ModuleList([])
        for i in range(self.layers):
            if i in [0, 2]:  # layer1, layer3
                # print(sample[i])
                self.Block.append(Bottleneck(channel[i], channel[i + 1], shadow_bn, sample[i]))
            else:
                self.Block.append(nn.Sequential(
                    nn.Conv2d(channel[i], channel[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channel[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                ))

        # self.avgpool = nn.MaxPool2d(4) # avg
        self.avgpool = nn.AvgPool2d(4)

        self.classifier = nn.Linear(last_channel, classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, choice=None):
        choice = {
            0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
            2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}

        x = self.stem(x)
        x = self.Block[0](x, choice[0]) # layer1
        x = self.Block[1](x)
        x = self.Block[2](x, choice[2]) # layer3

        x = self.avgpool(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

class SuperNetwork(nn.Module):
    def __init__(self, shadow_bn, layers=12, classes=10):
        super(SuperNetwork, self).__init__()
        self.layers = layers
        # prep
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels['prep'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels['prep']),
            nn.ReLU6(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.Inverted_Block = nn.ModuleList([])
        # for i in range(self.layers):
            # if i in [2, 5]: # layer3, layer6
            #     self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=2))
            # else:
        self.Inverted_Block.append(Inverted_Bottleneck(channels['prep'], channels['layer1'], shadow_bn, stride=1))
        self.Inverted_Block.append(Inverted_Bottleneck(channels['layer2'], channels['layer3'], shadow_bn, stride=1))



        self.last_conv = nn.Sequential(
            nn.Conv2d(channels['layer2'], channels['layer3'], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels['layer3']),
            nn.ReLU6(inplace=True)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels['layer3'], classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def forward(self, x, choice=None):
        # choice = {
        #     0: {'conv': [0, 0], 'rate': 1},
        #     1: {'conv': [0, 0], 'rate': 1},
        #     2: {'conv': [0, 0], 'rate': 1},
        #     3: {'conv': [0, 0], 'rate': 1},
        #     4: {'conv': [0, 0], 'rate': 1},
        #     5: {'conv': [0, 0], 'rate': 1},
        #     6: {'conv': [0, 0], 'rate': 1},
        #     7: {'conv': [0, 0], 'rate': 1},
        #     8: {'conv': [0, 0], 'rate': 1},
        #     9: {'conv': [0, 0], 'rate': 1},
        #     10: {'conv': [1, 2], 'rate': 1},
        #     11: {'conv': [1, 2], 'rate': 0}}
        x = self.stem(x)
        x = self.pool2(x)
        # for i in range(self.layers):
        #     x = self.Inverted_Block[i](x, choice[i])
        x = self.Inverted_Block[0](x, choice[0])
        x = self.pool3(x)

        x = self.last_conv(x)
        x = self.pool4(x)

        x = self.Inverted_Block[1](x, choice[1])

        x = self.global_pooling(x)
        x = x.view(-1, channels['layer3']) #
        x = self.classifier(x)
        return x

class SuperNetwork_2(nn.Module):
    def __init__(self, shadow_bn, layers=12, classes=10):
        super(SuperNetwork_2, self).__init__()
        self.layers = layers

        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )

        self.Inverted_Block = nn.ModuleList([])
        for i in range(self.layers):
            if i in [2, 5]: # layer3, layer6
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=2))
            else:
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=1))

        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def forward(self, x, choice=None):

        # choice = {
        #     0: {'conv': [0, 0], 'rate': 1},
        #     1: {'conv': [0, 0], 'rate': 1},
        #     2: {'conv': [0, 0], 'rate': 1},
        #     3: {'conv': [0, 0], 'rate': 1},
        #     4: {'conv': [0, 0], 'rate': 1},
        #     5: {'conv': [0, 0], 'rate': 1},
        #     6: {'conv': [0, 0], 'rate': 1},
        #     7: {'conv': [0, 0], 'rate': 1},
        #     8: {'conv': [0, 0], 'rate': 1},
        #     9: {'conv': [0, 0], 'rate': 1},
        #     10: {'conv': [1, 2], 'rate': 1},
        #     11: {'conv': [1, 2], 'rate': 0}}

        x = self.stem(x)

        for i in range(self.layers):
            x = self.Inverted_Block[i](x, choice[i])

        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel) #
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # choice = {
    #     0: {'conv': [0, 0], 'rate': 1},
    #     1: {'conv': [0, 0], 'rate': 1},
    #     2: {'conv': [0, 0], 'rate': 1},
    #     3: {'conv': [0, 0], 'rate': 1},
    #     4: {'conv': [0, 0], 'rate': 1},
    #     5: {'conv': [0, 0], 'rate': 1},
    #     6: {'conv': [0, 0], 'rate': 1},
    #     7: {'conv': [0, 0], 'rate': 1},
    #     8: {'conv': [0, 0], 'rate': 1},
    #     9: {'conv': [0, 0], 'rate': 1},
    #     10: {'conv': [1, 2], 'rate': 1},
    #     11: {'conv': [1, 2], 'rate': 0}}

    # model = SuperNetwork(shadow_bn=False, layers=12, classes=10)
    # model = SuperNetwork_2(shadow_bn=True, layers=3, classes=100)
    # model = SuperNetwork_2(shadow_bn=True, layers=1, classes=100)
    # model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100)

    # model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [3, 3], 2: [3, 5]})
    # model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [3, 3], 2: [7, 5]})
    # print(model)
    # input = torch.randn(3, 32, 32).unsqueeze(0)
    # print(model(input, choice).size())

    # 1: torch.Size([1, 64, 32, 32])
    # 2: torch.Size([1, 128, 16, 16])
    # 3: torch.Size([1, 256, 8, 8])
    # torch.Size([1, 512, 4, 4])
    # torch.Size([1, 512, 1, 1])
    # torch.Size([1, 512])
    # torch.Size([1, 100])

    """
    from ptflops import get_model_complexity_info
    inputs = torch.randn(3, 32, 32).unsqueeze(0)
    model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [3, 3], 2: [5, 5]})
    with torch.cuda.device(0):
        # 把choice拿到forward里
        #
        # choice = {
        #     0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    """

    # ------
    import torch
    from torchvision import models
    from torchsummary import summary
    from ptflops import get_model_complexity_info

    # inputs = torch.randn(3, 32, 32).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model = VGG('VGG11').to(device)
    model = SuperNetwork_3(shadow_bn=True, layers=3, classes=100, sample={0: [3, 3], 2: [3, 3]})
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

    # print(model)
    model = model.to(device)

    input_size = 32 # 224 224 #

    # part1
    with  torch.cuda.device(0):
        # 把choice拿到forward里
        # choice = {
        #     0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}

        # flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        flops, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # part2
    summary(model, (3, input_size, input_size))
