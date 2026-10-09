import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int = 10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR preact depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        self.depth = depth
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        feat = out.view(out.size(0), -1)
        return self.fc(feat), feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits


ARCHS = {
    "preact8": 8,
    "preact20": 20,
    "preact56": 56,
}


def build_model(arch: str, num_classes: int = 10) -> PreActResNet:
    if arch not in ARCHS:
        raise KeyError(f"unknown arch {arch}, expected one of {sorted(ARCHS)}")
    return PreActResNet(ARCHS[arch], num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
