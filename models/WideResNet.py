# model/wide_resnet.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))

class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features: int, alpha: float = 0.1, eps: float = 1e-05, momentum: float = 0.001, affine: bool = True, track_running_stats: bool = True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.alpha

class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0, activate_before_residual: bool = False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.activate_before_residual = activate_before_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_in_out and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return (x if self.equal_in_out else self.conv_shortcut(x)) + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, out_planes: int, block: nn.Module, stride: int, drop_rate: float = 0.0, activate_before_residual: bool = False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block: nn.Module, in_planes: int, out_planes: int, nb_layers: int, stride: int, drop_rate: float, activate_before_residual: bool) -> nn.Sequential:
        layers = [block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate, activate_before_residual) for i in range(nb_layers)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes: int, depth: int = 28, widen_factor: int = 2, drop_rate: float = 0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)

def build_wideresnet(depth: int, widen_factor: int, dropout: float, num_classes: int) -> WideResNet:
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth, widen_factor=widen_factor, drop_rate=dropout, num_classes=num_classes)
