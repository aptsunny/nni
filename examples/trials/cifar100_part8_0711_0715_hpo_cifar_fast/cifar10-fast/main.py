from core import *
from torch_backend import *
import nni
import logging
_logger = logging.getLogger("cifar10_pytorch_automl")

def res_block(c_in, c_out, stride, **kw):
    block = {
        'bn1': batch_norm(c_in, **kw),
        'relu1': nn.ReLU(True),
        'branch': {
            'conv1': nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            'bn2': batch_norm(c_out, **kw),
            'relu2': nn.ReLU(True),
            'conv2': nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        }
    }
    projection = (stride != 1) or (c_in != c_out)
    if projection:
        block['conv3'] = (nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False), ['relu1'])
    block['add'] = (Add(), [('conv3' if projection else 'relu1'), 'branch/conv2'])
    return block

def DAWN_net(c=64, block=res_block, prep_bn_relu=False, concat_pool=True, **kw):
    if isinstance(c, int):
        c = [c, 2 * c, 4 * c, 4 * c]

    classifier_pool = {
        'in': Identity(),
        'maxpool': nn.MaxPool2d(4),
        'avgpool': (nn.AvgPool2d(4), ['in']),
        'concat': (Concat(), ['maxpool', 'avgpool']),
    } if concat_pool else {'pool': nn.MaxPool2d(4)}

    return {
        'input': (None, []),
        'prep': union({'conv': nn.Conv2d(3, c[0], kernel_size=3, stride=1, padding=1, bias=False)},
                      {'bn': batch_norm(c[0], **kw), 'relu': nn.ReLU(True)} if prep_bn_relu else {}),
        'layer1': {
            'block0': block(c[0], c[0], 1, **kw),
            'block1': block(c[0], c[0], 1, **kw),
        },
        'layer2': {
            'block0': block(c[0], c[1], 2, **kw),
            'block1': block(c[1], c[1], 1, **kw),
        },
        'layer3': {
            'block0': block(c[1], c[2], 2, **kw),
            'block1': block(c[2], c[2], 1, **kw),
        },
        'layer4': {
            'block0': block(c[2], c[3], 2, **kw),
            'block1': block(c[3], c[3], 1, **kw),
        },
        'final': union(classifier_pool, {
            'flatten': Flatten(),
            'linear': nn.Linear(2 * c[3] if concat_pool else c[3], 10, bias=True),
        }),
        'logits': Identity(),
    }

def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        # 'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'bn': batch_norm(c_out, **kw),
        'relu': nn.ReLU(True)
    }

def basic_net(channels, weight, pool, **kw):
    return {
        'input': (None, []),
        'prep': conv_bn(3, channels['prep'], bn_weight_init=1.0, **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'logits': Mul(weight),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    residual = lambda c, **kw: {'in': Identity(), 'res1': conv_bn(c, c, **kw), 'res2': conv_bn(c, c, **kw),
                                'add': (Add(), ['in', 'res2/relu'])}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n

def train(model, lr_schedule, train_set, test_set, batch_size, num_workers=0):
# def train(model, lr_schedule, train_batches, test_batches, batch_size, num_workers=0):
#     num_workers = 0
    train_batches = DataLoader(train_set, batch_size, shuffle=True, set_random_choices=True, num_workers=num_workers)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    lr = lambda step: lr_schedule(step / len(train_batches)) / batch_size
    opts = [SGD(trainable_params(model).values(),
                {'lr': lr, 'weight_decay': Const(5e-4 * batch_size), 'momentum': Const(0.9)})]
    logs, state = Table(), {MODEL: model, LOSS: x_ent_loss, OPTS: opts}

    best_acc=0.0
    for epoch in range(lr_schedule.knots[-1]):

        result = train_epoch(state, Timer(torch.cuda.synchronize), train_batches, test_batches)
        # print(result['valid']['acc'])
        logs.append(union({'epoch': epoch + 1,
                           'lr': lr_schedule(epoch + 1)},
                           result))

        nni.report_intermediate_result(result['valid']['acc'])

        if result['valid']['acc'] > best_acc:
            best_acc = result['valid']['acc']

        # logs.append(union({'epoch': epoch + 1,
        #                    'lr': lr_schedule(epoch + 1)},
        #                   train_epoch(state, Timer(torch.cuda.synchronize), train_batches, test_batches)))



    return logs, best_acc



if __name__ == '__main__':
    RCV_CONFIG = nni.get_next_parameter()


    batch_norm = partial(BatchNorm, weight_init=None, bias_init=None)
    remove_identity_nodes = lambda net: remove_by_type(net, Identity)

    DATA_DIR = './data'
    dataset = cifar10(DATA_DIR)
    timer = Timer()
    print('Preprocessing training data')
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'),
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')


    lr_schedule = PiecewiseLinear([0, 5, 24], [0, 0.4, 0])
    batch_size = 512

    n = net()
    # draw(build_graph(n))
    model = Network(n).to(device).half()
    train_set_x = Transform(train_set, [Crop(32, 32), FlipLR(), Cutout(8,8)])

    #
    summary, best_acc = train(model, lr_schedule, train_set_x, test_set, batch_size=batch_size, num_workers=0)
    # summary = train(model, train_batches, test_batches, test_set, batch_size=batch_size, num_workers=0)


    nni.report_final_result(best_acc)

    # 单gpu下
    # 改完结构之后 3.15*24=75.6
    # nni状况下 run1/per gpu 1min34=94s
    # Concurrency4 wait2, run2/per gpu 一共3min18s=198,
    # Concurrency4 run4,  6min14s=374s

    #

    """
    try:
        # RCV_CONFIG = nni.get_next_parameter()
        RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}
        _logger.debug(RCV_CONFIG)



        # prepare(RCV_CONFIG)
        # acc = 0.0
        # best_acc = 0.0
        # for epoch in range(start_epoch, start_epoch+args.epochs):
        #     train(epoch, args.batches)
        #     acc, best_acc = test(epoch)
        #     print(acc, best_acc)
            # nni.report_intermediate_result(acc)

        # nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
    
    """