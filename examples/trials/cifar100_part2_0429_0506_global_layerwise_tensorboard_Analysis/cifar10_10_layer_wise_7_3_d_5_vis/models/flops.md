```markdown 3375
SuperNetwork_3(
  21.299 M, 100.000% Params, 0.615 GMac, 100.000% MACs, 
  (stem): Sequential(
    0.002 M, 0.009% Params, 0.002 GMac, 0.320% MACs, 
    (0): Conv2d(0.002 M, 0.008% Params, 0.002 GMac, 0.288% MACs, 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.021% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
  )
  (Block): ModuleList(
    21.246 M, 99.750% Params, 0.613 GMac, 99.671% MACs, 
    (0): Bottleneck(
      0.369 M, 1.734% Params, 0.151 GMac, 24.603% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        0.074 M, 0.347% Params, 0.075 GMac, 12.275% MACs, 
        (0): Sequential(
          0.074 M, 0.347% Params, 0.075 GMac, 12.275% MACs, 
          (0): Conv2d(0.074 M, 0.346% Params, 0.075 GMac, 12.275% MACs, 64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
        (0): ModuleList(
          0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
          (0): Sequential(
            0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
            (0): Conv2d(0.147 M, 0.692% Params, 0.038 GMac, 6.137% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.011% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
        (0): ModuleList(
          0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
          (0): Sequential(
            0.148 M, 0.694% Params, 0.038 GMac, 6.153% MACs, 
            (0): Conv2d(0.147 M, 0.692% Params, 0.038 GMac, 6.137% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.011% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      0.295 M, 1.387% Params, 0.076 GMac, 12.317% MACs, 
      (0): Conv2d(0.295 M, 1.385% Params, 0.075 GMac, 12.275% MACs, 128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.001 M, 0.002% Params, 0.0 GMac, 0.021% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
      (3): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Bottleneck(
      20.581 M, 96.629% Params, 0.386 GMac, 62.751% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        1.181 M, 5.543% Params, 0.075 GMac, 12.275% MACs, 
        (0): Sequential(
          1.181 M, 5.543% Params, 0.075 GMac, 12.275% MACs, 
          (0): Conv2d(1.18 M, 5.538% Params, 0.075 GMac, 12.275% MACs, 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.000% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        12.846 M, 60.312% Params, 0.206 GMac, 33.418% MACs, 
        (0): ModuleList(
          12.846 M, 60.312% Params, 0.206 GMac, 33.418% MACs, 
          (0): Sequential(
            12.846 M, 60.312% Params, 0.206 GMac, 33.418% MACs, 
            (0): Conv2d(12.845 M, 60.307% Params, 0.206 GMac, 33.414% MACs, 512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.003% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        6.555 M, 30.774% Params, 0.105 GMac, 17.052% MACs, 
        (0): ModuleList(
          6.555 M, 30.774% Params, 0.105 GMac, 17.052% MACs, 
          (0): Sequential(
            6.555 M, 30.774% Params, 0.105 GMac, 17.052% MACs, 
            (0): Conv2d(6.554 M, 30.769% Params, 0.105 GMac, 17.048% MACs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.003% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (avgpool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (classifier): Linear(0.051 M, 0.241% Params, 0.0 GMac, 0.008% MACs, in_features=512, out_features=100, bias=True)
)
Computational complexity:       0.62 GMac
Number of parameters:           21.3 M  


```

```markdown 3335
SuperNetwork_3(
  10.814 M, 100.000% Params, 0.447 GMac, 100.000% MACs, 
  (stem): Sequential(
    0.002 M, 0.017% Params, 0.002 GMac, 0.440% MACs, 
    (0): Conv2d(0.002 M, 0.016% Params, 0.002 GMac, 0.396% MACs, 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.029% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.015% MACs, inplace=True)
  )
  (Block): ModuleList(
    10.76 M, 99.508% Params, 0.445 GMac, 99.547% MACs, 
    (0): Bottleneck(
      0.369 M, 3.416% Params, 0.151 GMac, 33.831% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        0.074 M, 0.684% Params, 0.075 GMac, 16.879% MACs, 
        (0): Sequential(
          0.074 M, 0.684% Params, 0.075 GMac, 16.879% MACs, 
          (0): Conv2d(0.074 M, 0.682% Params, 0.075 GMac, 16.879% MACs, 64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
        (0): ModuleList(
          0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
          (0): Sequential(
            0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
            (0): Conv2d(0.147 M, 1.364% Params, 0.038 GMac, 8.439% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.015% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
        (0): ModuleList(
          0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
          (0): Sequential(
            0.148 M, 1.366% Params, 0.038 GMac, 8.461% MACs, 
            (0): Conv2d(0.147 M, 1.364% Params, 0.038 GMac, 8.439% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.015% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.029% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      0.295 M, 2.732% Params, 0.076 GMac, 16.937% MACs, 
      (0): Conv2d(0.295 M, 2.727% Params, 0.075 GMac, 16.879% MACs, 128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.029% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.015% MACs, inplace=True)
      (3): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.015% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Bottleneck(
      10.096 M, 93.360% Params, 0.218 GMac, 48.779% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        1.181 M, 10.918% Params, 0.075 GMac, 16.879% MACs, 
        (0): Sequential(
          1.181 M, 10.918% Params, 0.075 GMac, 16.879% MACs, 
          (0): Conv2d(1.18 M, 10.909% Params, 0.075 GMac, 16.879% MACs, 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.001 M, 0.009% Params, 0.0 GMac, 0.000% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        2.36 M, 21.827% Params, 0.038 GMac, 8.445% MACs, 
        (0): ModuleList(
          2.36 M, 21.827% Params, 0.038 GMac, 8.445% MACs, 
          (0): Sequential(
            2.36 M, 21.827% Params, 0.038 GMac, 8.445% MACs, 
            (0): Conv2d(2.359 M, 21.818% Params, 0.038 GMac, 8.439% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.009% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        6.555 M, 60.615% Params, 0.105 GMac, 23.448% MACs, 
        (0): ModuleList(
          6.555 M, 60.615% Params, 0.105 GMac, 23.448% MACs, 
          (0): Sequential(
            6.555 M, 60.615% Params, 0.105 GMac, 23.448% MACs, 
            (0): Conv2d(6.554 M, 60.605% Params, 0.105 GMac, 23.443% MACs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): BatchNorm2d(0.001 M, 0.009% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (avgpool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (classifier): Linear(0.051 M, 0.474% Params, 0.0 GMac, 0.011% MACs, in_features=512, out_features=100, bias=True)
)
Computational complexity:       0.45 GMac
Number of parameters:           10.81 M 
```

```markdown 3355
SuperNetwork_3(
  15.008 M, 100.000% Params, 0.514 GMac, 100.000% MACs, 
  (stem): Sequential(
    0.002 M, 0.012% Params, 0.002 GMac, 0.382% MACs, 
    (0): Conv2d(0.002 M, 0.012% Params, 0.002 GMac, 0.344% MACs, 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.025% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, inplace=True)
  )
  (Block): ModuleList(
    14.955 M, 99.646% Params, 0.512 GMac, 99.606% MACs, 
    (0): Bottleneck(
      0.369 M, 2.461% Params, 0.151 GMac, 29.417% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        0.074 M, 0.493% Params, 0.075 GMac, 14.677% MACs, 
        (0): Sequential(
          0.074 M, 0.493% Params, 0.075 GMac, 14.677% MACs, 
          (0): Conv2d(0.074 M, 0.491% Params, 0.075 GMac, 14.677% MACs, 64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
        (0): ModuleList(
          0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
          (0): Sequential(
            0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
            (0): Conv2d(0.147 M, 0.983% Params, 0.038 GMac, 7.338% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.013% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
        (0): ModuleList(
          0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
          (0): Sequential(
            0.148 M, 0.984% Params, 0.038 GMac, 7.357% MACs, 
            (0): Conv2d(0.147 M, 0.983% Params, 0.038 GMac, 7.338% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.013% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.025% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      0.295 M, 1.968% Params, 0.076 GMac, 14.728% MACs, 
      (0): Conv2d(0.295 M, 1.965% Params, 0.075 GMac, 14.677% MACs, 128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.025% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, inplace=True)
      (3): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Bottleneck(
      14.29 M, 95.216% Params, 0.285 GMac, 55.461% MACs, 
      (activation): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
      (conv_first): ModuleList(
        1.181 M, 7.867% Params, 0.075 GMac, 14.677% MACs, 
        (0): Sequential(
          1.181 M, 7.867% Params, 0.075 GMac, 14.677% MACs, 
          (0): Conv2d(1.18 M, 7.860% Params, 0.075 GMac, 14.677% MACs, 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(0.001 M, 0.007% Params, 0.0 GMac, 0.000% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        )
      )
      (mix_conv): ModuleList(
        6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
        (0): ModuleList(
          6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
          (0): Sequential(
            6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
            (0): Conv2d(6.554 M, 43.668% Params, 0.105 GMac, 20.384% MACs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): BatchNorm2d(0.001 M, 0.007% Params, 0.0 GMac, 0.003% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, inplace=True)
          )
        )
      )
      (mix_conv_2): ModuleList(
        6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
        (0): ModuleList(
          6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
          (0): Sequential(
            6.555 M, 43.674% Params, 0.105 GMac, 20.389% MACs, 
            (0): Conv2d(6.554 M, 43.668% Params, 0.105 GMac, 20.384% MACs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): BatchNorm2d(0.001 M, 0.007% Params, 0.0 GMac, 0.003% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, inplace=True)
          )
        )
      )
      (pool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (avgpool): MaxPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (classifier): Linear(0.051 M, 0.342% Params, 0.0 GMac, 0.010% MACs, in_features=512, out_features=100, bias=True)
)
Computational complexity:       0.51 GMac
Number of parameters:           15.01 M 
```