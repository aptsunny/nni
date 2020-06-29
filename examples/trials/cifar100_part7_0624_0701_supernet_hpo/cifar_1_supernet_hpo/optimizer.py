# import torch
import torch.optim as optim
# from utils import get_parameters, fast_hpo_lr_parameters

# weight-decay
def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []

    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)

    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)

    groups = [dict(params=group_weight_decay),
              dict(params=group_no_weight_decay, weight_decay=0.)]

    return groups

def fast_hpo_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 128:
        loc = 0
        # 1+(3+3+6+6)*7+1
        # conv3/conv5/dwconv3/dwconv5
        split_list = [1,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      1]
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

def combined(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 128:
        loc = 0
        # 1+ (3+3+6+6)*7+1
        split_list = [1,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      1]
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


def select_optim(args, lr_hpo):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(lr_hpo,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    if args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(lr_hpo,
                              lr=args.learning_rate)
    if args.optimizer  == 'Adagrad':
        optimizer = optim.Adagrad(lr_hpo,
                              lr=args.learning_rate)
    if args.optimizer  == 'Adam':
        optimizer = optim.Adam(lr_hpo,
                              lr=args.learning_rate)
    return optimizer

def get_optim(args, model):
    if args.layerwise_lr:
        # 128
        # lr_group = [args['prep'],
        #             args['layer1_conv0_3_3'],
        #             args['layer1_conv1_3_3'],
        #             args['layer1_conv2_3_3'],
        #             args['layer2_conv0_3_3'],
        #             args['layer3_conv0_3_3'],
        #             args['layer3_conv1_3_3'],
        #             args['layer3_conv2_3_3'],
        #             args['rest']]
        #
        # RCV_CONFIG = {
        #     "prep": 0.00087,
        #     "layer1_conv0_3_3": 0.06077,
        #     "layer1_conv1_3_3": 0.09482,
        #     "layer1_conv2_3_3": 0.01448,
        #     "layer2_conv0_3_3": 0.05309,
        #     "layer3_conv0_3_3": 0.03843,
        #     "layer3_conv1_3_3": 0.00401,
        #     "layer3_conv2_3_3": 0.09642,
        #     "rest": 0.00063}

        # lr_group = [0.1]* 30

        lr_3 = [
            0.04542,
            0.04214,
            0.03514,
            0.06933,
            0.09605,
            0.09912,
            0.00016
        ]
        lr_5 = [
            0.08994,
            0.00182,
            0.06746,
            0.08332,
            0.00077,
            0.07645,
            0.01914
        ]
        lr_d3 = [
            0.09917,
            0.00239,
            0.03508,
            0.09429,
            0.09718,
            0.00043,
            0.00014,
        ]
        lr_d5 = [
            0.05762,
            0.0991,
            0.06728,
            0.08458,
            0.03604,
            0.00139,
            0.00022,
        ]
        lr_group = []
        lr_group.append(0.00003)  #
        for i in range(7):
            lr_group.append(lr_3[i])
            lr_group.append(lr_5[i])
            lr_group.append(lr_d3[i])
            lr_group.append(lr_d5[i])
        lr_group.append(0.00178)  #
        # print(len(lr_group))

        optimizer = select_optim(args, fast_hpo_lr_parameters(model, lr_group))

        # optimizer = torch.optim.SGD(fast_hpo_lr_parameters(model, lr_group),
        #                             momentum=args.momentum,
        #                             weight_decay = args.weight_decay)
        #                             weight_decay=5e-4)  # nni

    elif args.global_lr:
        optimizer = select_optim(args, get_parameters(model))
        # optimizer = torch.optim.SGD(get_parameters(model),
        #                             lr=args.learning_rate,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

    else:
        optimizer = select_optim(args, get_parameters(model))
        # optimizer = torch.optim.SGD(get_parameters(model),
        #                             lr=args.learning_rate,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

    return optimizer