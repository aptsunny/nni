```markdown
SuperNetwork_2(
  (stem): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU6(inplace=True)
  )
  (Inverted_Block): ModuleList(
    (0): Inverted_Bottleneck(
      (activation): ReLU6(inplace=True)
      (pw): ModuleList(
        (0): Sequential(
          (0): Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (mix_conv): ModuleList(
        (0): ModuleList(
          (0): Sequential(
            (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(192, 192, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
        (1): ModuleList(
          (0): Sequential(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(384, 384, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
      )
      (mix_bn): ModuleList(
        (0): ModuleList(
          (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ModuleList(
          (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (pw_linear): ModuleList(
        (0): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
    (1): Inverted_Bottleneck(
      (activation): ReLU6(inplace=True)
      (pw): ModuleList(
        (0): Sequential(
          (0): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (mix_conv): ModuleList(
        (0): ModuleList(
          (0): Sequential(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(384, 384, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
        (1): ModuleList(
          (0): Sequential(
            (0): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(768, 768, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(768, 768, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
      )
      (mix_bn): ModuleList(
        (0): ModuleList(
          (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ModuleList(
          (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (pw_linear): ModuleList(
        (0): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
    (2): Inverted_Bottleneck(
      (activation): ReLU6(inplace=True)
      (pw): ModuleList(
        (0): Sequential(
          (0): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (mix_conv): ModuleList(
        (0): ModuleList(
          (0): Sequential(
            (0): Conv2d(768, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(768, 768, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(768, 768, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(768, 768, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4), groups=768, bias=False)
            (1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
        (1): ModuleList(
          (0): Sequential(
            (0): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1536, bias=False)
            (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Sequential(
            (0): Conv2d(1536, 1536, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1536, bias=False)
            (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=1536, bias=False)
            (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (3): Sequential(
            (0): Conv2d(1536, 1536, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4), groups=1536, bias=False)
            (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
        )
      )
      (mix_bn): ModuleList(
        (0): ModuleList(
          (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ModuleList(
          (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (pw_linear): ModuleList(
        (0): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
  )
  (last_conv): Sequential(
    (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU6(inplace=True)
  )
  (global_pooling): AdaptiveAvgPool2d(output_size=1)
  (classifier): Linear(in_features=512, out_features=100, bias=True)
)



```

```markdown
Network_cifar(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (relu1): ReLU(inplace=True)
  
  (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu2): ReLU(inplace=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (layer1_features): ModuleList(
    (0): BasicBlock(
      (conbine): Sequential(
        (0): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (conv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu3): ReLU(inplace=True)
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (conv4): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu4): ReLU(inplace=True)
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (layer3_features): ModuleList(
    (0): BasicBlock(
      (conbine): Sequential(
        (0): Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (avgpool): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=512, out_features=100, bias=False)
)

```

```markdown
module.stem.0.weight
module.stem.1.weight
module.stem.1.bias

module.Block.0.conv_first.0.0.weight
module.Block.0.conv_first.0.1.weight
module.Block.0.conv_first.0.1.bias

module.Block.0.mix_conv.0.0.0.weight
module.Block.0.mix_conv.0.0.1.weight
module.Block.0.mix_conv.0.0.1.bias

module.Block.0.mix_conv.0.1.0.weight
module.Block.0.mix_conv.0.1.1.weight
module.Block.0.mix_conv.0.1.1.bias

module.Block.0.mix_conv.0.2.0.weight
module.Block.0.mix_conv.0.2.1.weight
module.Block.0.mix_conv.0.2.1.bias

module.Block.0.mix_conv_2.0.0.0.weight
module.Block.0.mix_conv_2.0.0.1.weight
module.Block.0.mix_conv_2.0.0.1.bias

module.Block.0.mix_conv_2.0.1.0.weight
module.Block.0.mix_conv_2.0.1.1.weight
module.Block.0.mix_conv_2.0.1.1.bias

module.Block.0.mix_conv_2.0.2.0.weight
module.Block.0.mix_conv_2.0.2.1.weight
module.Block.0.mix_conv_2.0.2.1.bias

module.Block.1.0.weight
module.Block.1.1.weight
module.Block.1.1.bias

module.Block.2.conv_first.0.0.weight
module.Block.2.conv_first.0.1.weight
module.Block.2.conv_first.0.1.bias

module.Block.2.mix_conv.0.0.0.weight
module.Block.2.mix_conv.0.0.1.weight
module.Block.2.mix_conv.0.0.1.bias

module.Block.2.mix_conv.0.1.0.weight
module.Block.2.mix_conv.0.1.1.weight
module.Block.2.mix_conv.0.1.1.bias

module.Block.2.mix_conv.0.2.0.weight
module.Block.2.mix_conv.0.2.1.weight
module.Block.2.mix_conv.0.2.1.bias

module.Block.2.mix_conv_2.0.0.0.weight
module.Block.2.mix_conv_2.0.0.1.weight
module.Block.2.mix_conv_2.0.0.1.bias

module.Block.2.mix_conv_2.0.1.0.weight
module.Block.2.mix_conv_2.0.1.1.weight
module.Block.2.mix_conv_2.0.1.1.bias

module.Block.2.mix_conv_2.0.2.0.weight
module.Block.2.mix_conv_2.0.2.1.weight
module.Block.2.mix_conv_2.0.2.1.bias

module.classifier.weight
module.classifier.bias

```