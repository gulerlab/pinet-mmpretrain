import torch.nn as nn
from mmpretrain.registry import MODELS

class PiNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False) -> None:
        super(PiNetConv2d, self).__init__()
        self.conv_01 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_02 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    
    def forward(self, x):
        out = self.conv_01(x)
        second = out * self.conv_02(x)
        return second + out

@MODELS.register_module()
class PiNetShallow_CIFAR(nn.Module):
    def __init__(self) -> None:
        super(PiNetShallow_CIFAR, self).__init__()
        self.piconv_1 = PiNetConv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.piconv_2 = PiNetConv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.piconv_3 = PiNetConv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.piconv_4 = PiNetConv2d(128, 256, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = self.piconv_1(x)
        out = self.piconv_2(out)
        out = self.piconv_3(out)
        out = self.piconv_4(out)
        return out